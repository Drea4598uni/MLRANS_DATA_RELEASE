/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pimpleFoamReg (variante nativeML)

Group
    grpIncompressibleSolvers

Description
    Transient solver for incompressible, turbulent flow of Newtonian fluids
    on a fixed mesh, con correzione AI di nut (clustering KMeans + MLP
    residuale) eseguita in C++ NATIVO tramite nativeML.H: nessun embedding
    Python/TensorFlow a runtime.

    I parametri dei modelli vengono esportati una tantum con:
        <venv-python> export_models.py
    che genera models/native_ml.dat a partire da km, km-scaler,
    preprocessor.pkl, y_scaler_nut.pkl e best_model_nut.keras.

    Configurazione runtime (stesse variabili della versione Python):
        PIMPLEFOAMREG_SOLVER_DIR, PIMPLEFOAMREG_MODEL_DIR,
        PIMPLEFOAMREG_NATIVE_PARAMS (path diretto a native_ml.dat),
        NNTURBINE, CYL_ABS, NNCORR_ROTOR_DIAMETER,
        NNCORR_TARGET_MODE, NNCORR_TARGET_TRANSFORM,
        NNCLUSTER_REORDER_VAR, NNCLUSTER_REORDER_ASCENDING,
        NATIVE_ML_THREADS (thread OpenMP per l'inferenza, default 1)

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "HashSet.H"
#include "kOmegaSSTML/kOmegaSSTML.H"

// Inferenza ML nativa (KMeans + MLP residuale, senza Python)
#include "nativeML.H"

#include <unistd.h>
#include <memory>
#include <string>
#include <fstream>

static inline bool fileExists(const std::string& path)
{
    std::ifstream input(path.c_str());
    return input.good();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Transient solver for incompressible, turbulent flow"
        " of Newtonian fluids on a fixed mesh (native ML correction)."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "initContinuityErrs.H"
    #include "createTimeControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // Pipeline ML nativa (caricata al primo turbCorr)
    static std::unique_ptr<nativeML::Pipeline> mlPipeline;

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // CREA CAMPI CLUSTERING E NEURAL NETWORK ALL'INIZIO
    volScalarField clusterID
    (
        IOobject
        (
            "clusterID",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("clusterID", dimless, 0)
    );
    volScalarField axField
    (
        IOobject
        (
            "ax",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("ax", dimLength, 0)
    );
    volScalarField rhoField
    (
        IOobject
        (
            "rho",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("rho", dimLength, 0)
    );
    volScalarField thetaField
    (
        IOobject
        (
            "theta",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("theta", dimless, 0)
    );

    // 1. Prima dichiara le variabili statiche
    static volScalarField* nutFieldPtr = nullptr;
    static volScalarField* nutOriginalPtr = nullptr;
    static volScalarField* kFieldPtr = nullptr;
    static volScalarField* kOriginalPtr = nullptr;
    static volScalarField* omegaFieldPtr = nullptr;
    static volScalarField* omegaOriginalPtr = nullptr;
    static bool nutStandardReady = false;
    static bool firstIteration = true;
    static const bool applyNutCorrection = true;
    const char* solverDirEnv = getenv("PIMPLEFOAMREG_SOLVER_DIR");
    const std::string solverDir =
        (solverDirEnv && solverDirEnv[0])
      ? std::string(solverDirEnv)
      : std::string("/home/frank1/andrea/solvers_of/libOFRegression/pimpleFoamReg_native");
    const char* modelDirEnv = getenv("PIMPLEFOAMREG_MODEL_DIR");
    const std::string modelDir =
        (modelDirEnv && modelDirEnv[0])
      ? std::string(modelDirEnv)
      : solverDir + "/models";
    const char* paramsEnv = getenv("PIMPLEFOAMREG_NATIVE_PARAMS");
    const std::string paramsPath =
        (paramsEnv && paramsEnv[0])
      ? std::string(paramsEnv)
      : modelDir + "/native_ml.dat";
    static Foam::RASModels::kOmegaSSTML<Foam::incompressible::turbulenceModel>*
        kOmegaSSTMLPtr = nullptr;

    if (Pstream::master())
    {
        Info<< "AI correction setup (nativeML): applyNutCorrection="
            << applyNutCorrection
            << ", params=" << paramsPath.c_str()
            << endl;
    }

    // Campo di supporto: conserva la nut ricalcolata da
    // turbulence->correct(). Prima del turbulence correction successivo viene
    // ripristinata su nut, poi ricalcolata e rimandata in input alla rete.
    volScalarField nut_original
    (
        IOobject
        (
            "nut_original",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("nut_original", dimViscosity, 0)
    );
    nutOriginalPtr = &nut_original;

    // Campo di supporto: conserva la k appena ricalcolata da
    // turbulence->correct(), usata come input della rete.
    volScalarField k_original
    (
        IOobject
        (
            "k_original",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("k_original", dimVelocity*dimVelocity, 0)
    );
    kOriginalPtr = &k_original;

    // Campo di supporto: conserva la omega appena ricalcolata
    // da turbulence->correct(), usata per diagnostica.
    volScalarField omega_original
    (
        IOobject
        (
            "omega_original",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("omega_original", dimless/dimTime, 0)
    );
    omegaOriginalPtr = &omega_original;

    Info<< "\nStarting time loop\n" << endl;

    {
    while (runTime.run())
    {

        #include "readTimeControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (pimple.loop())
        {
            if (!nutFieldPtr) {
                nutFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("nut"));
            }
            if (!kFieldPtr) {
                kFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("k"));
            }
            if (!omegaFieldPtr) {
                omegaFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("omega"));
            }
            if (nutFieldPtr) {
                const scalarField& nutField = nutFieldPtr->primitiveField();
                Info<< "*** START PIMPLE - nut range: [" << min(nutField)
                    << ", " << max(nutField) << "]" << endl;
            }
            if (kFieldPtr) {
                const scalarField& kField = kFieldPtr->primitiveField();
                Info<< "*** START PIMPLE - k range: [" << min(kField)
                    << ", " << max(kField) << "]" << endl;
            }
            if (omegaFieldPtr) {
                const scalarField& omegaField = omegaFieldPtr->primitiveField();
                Info<< "*** START PIMPLE - omega range: [" << min(omegaField)
                    << ", " << max(omegaField) << "]" << endl;
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();

                if (!kOmegaSSTMLPtr)
                {
                    kOmegaSSTMLPtr =
                        dynamic_cast
                        <
                            Foam::RASModels::kOmegaSSTML
                            <
                                Foam::incompressible::turbulenceModel
                            >*
                        >(turbulence.get());
                }
                if (kOmegaSSTMLPtr)
                {
                    kOmegaSSTMLPtr->setSolveNut(true);
                }

                // --- INTEGRAZIONE CLUSTERING + RETE NEURALE (NATIVA) ---

                if (!nutFieldPtr) {
                    nutFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("nut"));
                    Info<< "✓ Persistent reference to nut field established" << endl;
                }
                if (!kFieldPtr) {
                    kFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("k"));
                    Info<< "✓ Persistent reference to k field established" << endl;
                }
                if (!omegaFieldPtr) {
                    omegaFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("omega"));
                    Info<< "✓ Persistent reference to omega field established" << endl;
                }

                const bool hasNutState = nutFieldPtr && nutOriginalPtr;
                const bool hasKState = kFieldPtr && kOriginalPtr;
                const bool hasOmegaState = omegaFieldPtr && omegaOriginalPtr;

                if (nutStandardReady && hasNutState)
                {
                    nutFieldPtr->primitiveFieldRef() = nutOriginalPtr->primitiveField();
                    nutFieldPtr->correctBoundaryConditions();
                }

                if (Pstream::master())
                {
                    if (nutStandardReady && hasNutState)
                    {
                        Info<< "*** turbulence->correct() uses restored RANS nut"
                            << " from previous turbulence correction ***"
                            << endl;
                    }
                    else
                    {
                        Info<< "*** turbulence->correct() bootstrap: no previous RANS nut to restore ***"
                            << endl;
                    }
                    Info<< "nut field supplied to turbulence->correct(): [" << min(nutFieldPtr->primitiveField())
                        << ", " << max(nutFieldPtr->primitiveField()) << "]" << endl;
                    Info<< "k field supplied to turbulence->correct(): [" << min(kFieldPtr->primitiveField())
                        << ", " << max(kFieldPtr->primitiveField()) << "]" << endl;
                    if (omegaFieldPtr)
                    {
                        Info<< "omega field supplied to turbulence->correct(): [" << min(omegaFieldPtr->primitiveField())
                            << ", " << max(omegaFieldPtr->primitiveField()) << "]" << endl;
                    }
                }

                turbulence->correct();

                scalarField& nutAfterTurbulence = nutFieldPtr->primitiveFieldRef();
                scalarField& kAfterTurbulence = kFieldPtr->primitiveFieldRef();
                scalarField& omegaAfterTurbulence = omegaFieldPtr->primitiveFieldRef();
                if (hasNutState)
                {
                    nutOriginalPtr->primitiveFieldRef() = nutAfterTurbulence;
                    nutOriginalPtr->correctBoundaryConditions();
                }
                nutStandardReady = hasNutState;
                if (hasKState)
                {
                    kOriginalPtr->primitiveFieldRef() = kAfterTurbulence;
                    kOriginalPtr->correctBoundaryConditions();
                }
                if (hasOmegaState)
                {
                    omegaOriginalPtr->primitiveFieldRef() = omegaAfterTurbulence;
                    omegaOriginalPtr->correctBoundaryConditions();
                }

                if (Pstream::master())
                {
                    Info<< "nut after turbulence->correct() before NN: ["
                        << min(nutAfterTurbulence)
                        << ", " << max(nutAfterTurbulence) << "]" << endl;
                    Info<< "k after turbulence->correct(): [" << min(kAfterTurbulence)
                        << ", " << max(kAfterTurbulence) << "]" << endl;
                    Info<< "omega after turbulence->correct(): [" << min(omegaAfterTurbulence)
                        << ", " << max(omegaAfterTurbulence) << "]" << endl;
                }

                // === INIZIALIZZAZIONE PIPELINE NATIVA (una volta sola) ===
                if (!mlPipeline && applyNutCorrection)
                {
                    Info<< "Loading native ML pipeline from "
                        << paramsPath.c_str() << endl;

                    if (!fileExists(paramsPath))
                    {
                        FatalErrorInFunction
                            << "File parametri non trovato: " << paramsPath.c_str() << nl
                            << "Generarlo con: <venv-python> "
                            << solverDir.c_str() << "/export_models.py"
                            << exit(FatalError);
                    }

                    // Il frame cilindrico viene letto da system/fvOptions e
                    // system/Trim.<turbina> nella directory di lavoro, come
                    // faceva la pipeline Python (os.getcwd()).
                    char cwdBuf[4096];
                    if (!::getcwd(cwdBuf, sizeof(cwdBuf)))
                    {
                        FatalErrorInFunction
                            << "getcwd() fallita" << exit(FatalError);
                    }

                    try
                    {
                        mlPipeline.reset
                        (
                            new nativeML::Pipeline(paramsPath, cwdBuf)
                        );
                    }
                    catch (const std::exception& e)
                    {
                        FatalErrorInFunction
                            << "Caricamento pipeline nativeML fallito: "
                            << e.what() << exit(FatalError);
                    }

                    if (Pstream::master())
                    {
                        const auto& g = mlPipeline->geometry;
                        const auto& km = mlPipeline->kmeans;
                        Info<< "✓ Native ML pipeline loaded" << nl
                            << "  turbine: "
                            << (g.turbineName.empty() ? "(default)" : g.turbineName.c_str()) << nl
                            << "  origin: (" << g.origin.x << " " << g.origin.y
                            << " " << g.origin.z << ")" << nl
                            << "  axis dir: (" << g.axisDir.x << " " << g.axisDir.y
                            << " " << g.axisDir.z << ")" << nl
                            << "  rotor diameter: " << g.rotorDiameter << " m" << nl
                            << "  yaw: " << g.yaw << " rad, tilt: " << g.tilt << " rad" << nl
                            << "  point shift: (" << g.pointShift.x << " "
                            << g.pointShift.y << " " << g.pointShift.z << ")" << nl
                            << "  KMeans clusters: " << km.nClusters
                            << " (riordino per '" << km.reorderVar.c_str() << "' "
                            << (km.reorderApplied ? "applicato" : "NON applicato") << ")" << nl
                            << "  target mode: " << mlPipeline->targetMode.c_str()
                            << ", transform: " << mlPipeline->targetTransform.c_str() << nl
                            << "  U-inf norm: "
                            << (mlPipeline->uinfNorm ? "ON" : "OFF")
                            << " (train=" << mlPipeline->uinfTrain
                            << ", infer=" << mlPipeline->uinfInfer
                            << ", r=" << mlPipeline->uinfR << ")"
                            << endl;

                        if (!km.reorderApplied)
                        {
                            WarningInFunction
                                << "Variabile di riordino cluster '"
                                << km.reorderVar.c_str()
                                << "' non nelle feature KMeans: uso ID 1-based"
                                << " nell'ordine grezzo (NESSUN riordino fisico!)"
                                << endl;
                        }
                    }
                }

                // ESEGUI CORREZIONE NN AD OGNI CICLO turbCorr
                if (mlPipeline && applyNutCorrection)
                {
                    const volVectorField& C = mesh.C();

                    // La rete usa i campi appena ricalcolati da turbulence->correct().
                    const volScalarField& kForNN = *kOriginalPtr;
                    const volScalarField& nutForNN = *nutOriginalPtr;

                    scalarField x = C.component(0)().primitiveField();
                    scalarField y = C.component(1)().primitiveField();
                    scalarField z = C.component(2)().primitiveField();
                    scalarField ux = U.component(0)().primitiveField();
                    scalarField uy = U.component(1)().primitiveField();
                    scalarField uz = U.component(2)().primitiveField();
                    scalarField k_field = kForNN.primitiveField();
                    scalarField nut_field = nutForNN.primitiveField();

                    label nCells = mesh.nCells();
                    scalar pipelineStart = runTime.elapsedClockTime();

                    if (Pstream::master())
                    {
                        Info<< "Native ML correction started (" << nCells
                            << " points)" << endl;
                    }

                    nativeML::PipelineResult mlResult;
                    nativeML::PipelineTimings mlTimings;
                    try
                    {
                        mlResult = mlPipeline->run
                        (
                            x.cdata(), y.cdata(), z.cdata(),
                            ux.cdata(), uy.cdata(), uz.cdata(),
                            k_field.cdata(), nut_field.cdata(),
                            static_cast<std::size_t>(nCells),
                            &mlTimings
                        );
                    }
                    catch (const std::exception& e)
                    {
                        FatalErrorInFunction
                            << "Inferenza nativeML fallita: " << e.what()
                            << exit(FatalError);
                    }

                    // Copia campi diagnostici
                    scalarField& clusterIDField = clusterID.primitiveFieldRef();
                    scalarField& axFieldVals = axField.primitiveFieldRef();
                    scalarField& rhoFieldVals = rhoField.primitiveFieldRef();
                    scalarField& thetaFieldVals = thetaField.primitiveFieldRef();
                    for (label i = 0; i < nCells; ++i)
                    {
                        clusterIDField[i] = scalar(mlResult.clusterId[i]);
                        axFieldVals[i] = mlResult.axCoord[i];
                        rhoFieldVals[i] = mlResult.rho[i];
                        thetaFieldVals[i] = mlResult.theta[i];
                    }
                    axField.correctBoundaryConditions();
                    rhoField.correctBoundaryConditions();
                    thetaField.correctBoundaryConditions();

                    // Diagnostica distribuzione cluster (solo master, come Python)
                    if (Pstream::master())
                    {
                        const int nk = mlPipeline->kmeans.nClusters;
                        List<label> counts(nk + 1, label(0));
                        for (label i = 0; i < nCells; ++i)
                        {
                            const int cid = mlResult.clusterId[i];
                            if (cid >= 1 && cid <= nk) counts[cid]++;
                        }
                        Info<< "[CLUSTERING] distribution:";
                        for (int c = 1; c <= nk; ++c)
                        {
                            Info<< " " << c << ":" << counts[c];
                        }
                        Info<< endl;
                    }

                    bool nutUpdated = false;

                    if (applyNutCorrection && nutFieldPtr)
                    {
                        scalarField& nutPrimitiveField = nutFieldPtr->primitiveFieldRef();
                        for (label i = 0; i < nCells; ++i)
                        {
                            double value = mlResult.nutCorrected[i];
                            if (value < 0.0)
                            {
                                value = 0.000015;
                            }
                            nutPrimitiveField[i] = value;
                        }
                        nutFieldPtr->correctBoundaryConditions();
                        nutUpdated = true;

                        if (Pstream::master())
                        {
                            Info<< "AI nut correction applied (native)" << endl;
                            Info<< "Final corrected nut range: [" << min(nutPrimitiveField)
                                << ", " << max(nutPrimitiveField) << "]" << endl;
                        }
                    }

                    scalar totalPipelineTime = runTime.elapsedClockTime() - pipelineStart;
                    if (Pstream::master())
                    {
                        Info<< "Native ML correction completed" << endl;
                        Info<< "Total AI pipeline time: " << totalPipelineTime
                            << " s" << endl;
                        // Breakdown per-fase (rank master, ad alta risoluzione)
                        Info<< "  ML phase breakdown [s]: geometry="
                            << mlTimings.geometry << ", clustering="
                            << mlTimings.clustering << ", features="
                            << mlTimings.features << ", mlp=" << mlTimings.mlp
                            << ", output=" << mlTimings.output << endl;
                    }

                    if (firstIteration && nutUpdated)
                    {
                        firstIteration = false;
                        Info<< "*** FIRST ITERATION COMPLETED - AI CORRECTION ACTIVE (nut, native) ***"
                            << endl;
                    }
                }

            } // End turbCorr
        } // End pimple.loop()

        // DEBUG: Verifica se deve scrivere
        if (runTime.writeTime())
        {
            Info<< "*** WRITING TIME STEP " << runTime.timeName() << " ***" << endl;
        }
        else
        {
            Info<< "*** NOT WRITING TIME STEP " << runTime.timeName() << " ***" << endl;
        }

        runTime.write();
        runTime.printExecutionTime(Info);
    } // End time loop

    Info<< "End\n" << endl;
    return 0;
    } // End AI correction scope
} // End main

// ************************************************************************* //
