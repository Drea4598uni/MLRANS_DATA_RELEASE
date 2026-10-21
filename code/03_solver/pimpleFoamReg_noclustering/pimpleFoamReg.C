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

    OpenFOAM is free software: you can redistribute it and/or modify itclear
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
    pimpleFoamReg.C

Group
    grpIncompressibleSolvers

Description
    Transient solver for incompressible, turbulent flow of Newtonian fluids
    on a fixed mesh.

    MACHINE-SPECIFIC CONFIGURATION:
    ===============================
    Current machine (frank1-21): Use complete environment at
    /home/frank1/andrea/acTestEnv/bin/python3.10
    
    To setup environment:
    source /home/frank1/andrea/acTestEnv/bin/activate
    pip install tensorflow==2.14.0 keras==2.14.0 numpy==1.24.3 scikit-learn==1.3.1

    \heading Solver details
    The solver uses the PIMPLE (merged PISO-SIMPLE) algorithm to solve the
    continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \ddt{\vec{U}} + \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    Sub-models include:
    - turbulence modelling, i.e. laminar, RAS or LES
    - run-time selectable MRF and finite volume options, e.g. explicit porosity

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

Note
   The motion frequency of this solver can be influenced by the presence
   of "updateControl" and "updateInterval" in the dynamicMeshDict.

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

// PYTHON HEADERS
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <signal.h>
#include <fenv.h>
#include <string>
#include <fstream>

// Copy OpenFOAM scalarField into a preallocated NumPy array (no allocation)
static inline void copyToNumpyArray(PyObject* array, const scalarField& data)
{
    PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(array);
    double* array_data = static_cast<double*>(PyArray_DATA(np_array));

    label size = min(data.size(), static_cast<label>(PyArray_SIZE(np_array)));
    for (label i = 0; i < size; i++) {
        array_data[i] = data[i];
    }
}

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
        " of Newtonian fluids on a fixed mesh."
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

    // Variabili Python - MEMORY LEAK SOLUTION
    static bool pythonInitialized = false;
    static PyObject* neuralNetworkModule = nullptr;
    
    // PERSISTENT OBJECTS FOR ZERO MEMORY LEAK
    static PyObject* persistent_corrector = nullptr;
    static constexpr int NUM_PERSISTENT_ARRAYS = 8;
    // Reusable NumPy arrays: x,y,z, ux,uy,uz, k,nut
    static PyObject* persistent_arrays[NUM_PERSISTENT_ARRAYS] = {nullptr};
    static bool objects_initialized = false;
    static label max_cells_allocated = 0; 

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // Campi diagnostici della trasformazione cilindrica usata dalla rete.
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
      : std::string("/home/frank1/andrea/solvers_of/libOFRegression/pimpleFoamReg_noclustering");
    const char* modelDirEnv = getenv("PIMPLEFOAMREG_MODEL_DIR");
    const std::string modelDir =
        (modelDirEnv && modelDirEnv[0])
      ? std::string(modelDirEnv)
      : solverDir + "/models";
    static Foam::RASModels::kOmegaSSTML<Foam::incompressible::turbulenceModel>*
        kOmegaSSTMLPtr = nullptr;

    if (Pstream::master())
    {
        Info<< "AI correction setup: applyNutCorrection=" << applyNutCorrection
            << ", modelDir=" << modelDir
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

                // --- INTEGRAZIONE RETE NEURALE SENZA CLUSTERING ---

                // ESTABLISH PERSISTENT REFERENCE TO NUT FIELD (se non già fatto)
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
                    if (nutOriginalPtr)
                    {
                        Info<< "nut stored for NN input: [" << min(nutOriginalPtr->primitiveField())
                            << ", " << max(nutOriginalPtr->primitiveField()) << "]" << endl;
                    }
                    Info<< "k after turbulence->correct(): [" << min(kAfterTurbulence)
                        << ", " << max(kAfterTurbulence) << "]" << endl;
                    if (kOriginalPtr)
                    {
                        Info<< "k stored for NN input: [" << min(kOriginalPtr->primitiveField())
                            << ", " << max(kOriginalPtr->primitiveField()) << "]" << endl;
                    }
                    Info<< "omega after turbulence->correct(): [" << min(omegaAfterTurbulence)
                        << ", " << max(omegaAfterTurbulence) << "]" << endl;
                    if (hasOmegaState)
                    {
                        Info<< "omega stored: [" << min(omegaOriginalPtr->primitiveField())
                            << ", " << max(omegaOriginalPtr->primitiveField()) << "]" << endl;
                    }
                }

                if (!pythonInitialized) {
                    Info<< "Initializing Python embedding..." << endl;
                    
                    // Environment setup (minimal debug)
                    setenv("PYTHONHOME", "/home/frank1/andrea/acTestEnv", 1);
                    const std::string pythonPath =
                        "/home/frank1/andrea/acTestEnv/lib/python3.10/site-packages:"
                        "/home/frank1/andrea/acTestEnv/lib/python3.10:"
                      + solverDir;
                    setenv("PYTHONPATH", pythonPath.c_str(), 1);
                    setenv("VIRTUAL_ENV", "/home/frank1/andrea/acTestEnv", 1);
                    
                    Py_SetProgramName(L"/home/frank1/andrea/acTestEnv/bin/python3");
                    Py_SetPythonHome(L"/home/frank1/andrea/acTestEnv");

                    signal(SIGFPE, SIG_IGN);
                    fedisableexcept(FE_ALL_EXCEPT);

                    Py_Initialize();
                    if (!Py_IsInitialized()) {
                        FatalErrorInFunction << "Python initialization failed" << exit(FatalError);
                    }

                    // Initialize NumPy C API
                    if (_import_array() < 0) {
                        PyErr_Print();
                        FatalErrorInFunction << "NumPy C API initialization failed" << exit(FatalError);
                    }

                    // Add solver directory to Python path
                    const std::string pythonPathSetup =
                        "import sys\n"
                        "import os\n"
                        "solver_dir = '" + solverDir + "'\n"
                        "if solver_dir not in sys.path:\n"
                        "    sys.path.insert(0, solver_dir)\n"
                        "print(f'✓ Added {solver_dir} to Python path')\n";
                    PyRun_SimpleString(pythonPathSetup.c_str());

                    // Load modules
                    neuralNetworkModule = PyImport_ImportModule("neural_network");
                    if (!neuralNetworkModule) {
                        PyErr_Print();
                        FatalErrorInFunction << "Failed to import neural_network module" << exit(FatalError);
                    }

                    pythonInitialized = true;
                    Info<< "✓ Python modules loaded successfully" << endl;
                }

                // === MEMORY LEAK SOLUTION: PERSISTENT OBJECTS ===
                if (!objects_initialized && pythonInitialized) {
                    Info<< "Creating persistent Python objects..." << endl;
                    
                    label nCells = mesh.nCells();
                    // Alloca esattamente nCells per evitare code di zeri e mismatch
                    max_cells_allocated = nCells;
                    
                    // Create persistent corrector (ONCE). Default model corrects nut only.
                    PyObject* correctorClass = PyObject_GetAttrString(neuralNetworkModule, "NeuralNetworkCorrector");
                    if (applyNutCorrection)
                    {
                        PyObject* correctorArgs = PyTuple_New(3);
                        const char* nnModelEnv = getenv("PIMPLEFOAMREG_NN_MODEL");
                        const char* nnPreprocessorEnv = getenv("PIMPLEFOAMREG_NN_PREPROCESSOR");
                        const char* nnScalerEnv = getenv("PIMPLEFOAMREG_NN_SCALER");
                        std::string nnModelPath =
                            (nnModelEnv && nnModelEnv[0])
                          ? std::string(nnModelEnv)
                          : modelDir + "/best_model_nut.keras";
                        std::string nnPreprocessorPath =
                            (nnPreprocessorEnv && nnPreprocessorEnv[0])
                          ? std::string(nnPreprocessorEnv)
                          : modelDir + "/preprocessor.pkl";
                        std::string nnScalerPath =
                            (nnScalerEnv && nnScalerEnv[0])
                          ? std::string(nnScalerEnv)
                          : modelDir + "/y_scaler_nut.pkl";
                        if (Pstream::master())
                        {
                            Info<< "NN model path: " << nnModelPath << nl
                                << "NN preprocessor path: " << nnPreprocessorPath << nl
                                << "NN target scaler path: " << nnScalerPath << endl;
                        }
                        PyTuple_SetItem(correctorArgs, 0, PyUnicode_FromString(nnModelPath.c_str()));
                        PyTuple_SetItem(correctorArgs, 1, PyUnicode_FromString(nnPreprocessorPath.c_str()));
                        PyTuple_SetItem(correctorArgs, 2, PyUnicode_FromString(nnScalerPath.c_str()));
                        persistent_corrector = PyObject_CallObject(correctorClass, correctorArgs);
                        Py_DECREF(correctorArgs);
                    }

                    Py_DECREF(correctorClass);
                    
                    if (applyNutCorrection && !persistent_corrector) {
                        PyErr_Print();
                        FatalErrorInFunction << "Failed to create persistent NN corrector" << exit(FatalError);
                    }
                    
                    // Pre-allocate NumPy arrays (ONCE)
                    npy_intp arrDims[1] = {static_cast<npy_intp>(max_cells_allocated)};
                    for (int i = 0; i < NUM_PERSISTENT_ARRAYS; i++) {
                        persistent_arrays[i] = PyArray_ZEROS(1, arrDims, NPY_DOUBLE, 0);
                        if (!persistent_arrays[i]) {
                            FatalErrorInFunction << "Failed to create persistent array " << i << exit(FatalError);
                        }
                    }
                    
                    objects_initialized = true;
                    Info<< "✓ Persistent objects created (allocated for " << max_cells_allocated << " cells)" << endl;
                }

                // ESEGUI CORREZIONE NN AD OGNI CICLO turbCorr (iterativo nel PIMPLE loop)
                const bool doNNCorrection = applyNutCorrection;

                if (pythonInitialized && objects_initialized && doNNCorrection) {
                    // ESTABLISH PERSISTENT REFERENCE TO NUT/K FIELD (first time only)
                    if (!nutFieldPtr) {
                        nutFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("nut"));
                    }
                    if (!kFieldPtr) {
                        kFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("k"));
                    }
                    if (!omegaFieldPtr) {
                        omegaFieldPtr = &const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("omega"));
                    }

                    // Extract mesh data
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
                    scalar timestepStart = runTime.elapsedClockTime();
                    scalar neuralNetworkStart, neuralNetworkTime;

                    // === UPDATE PERSISTENT ARRAYS (NO NEW ALLOCATIONS) ===
                    copyToNumpyArray(persistent_arrays[0], x);           // py_x
                    copyToNumpyArray(persistent_arrays[1], y);           // py_y
                    copyToNumpyArray(persistent_arrays[2], z);           // py_z
                    copyToNumpyArray(persistent_arrays[3], ux);          // py_ux
                    copyToNumpyArray(persistent_arrays[4], uy);          // py_uy
                    copyToNumpyArray(persistent_arrays[5], uz);          // py_uz
                    copyToNumpyArray(persistent_arrays[6], k_field);     // py_k
                    copyToNumpyArray(persistent_arrays[7], nut_field);   // py_nut
                    
                    neuralNetworkStart = runTime.elapsedClockTime();
                    if (Pstream::master())
                    {
                        Info<< "Neural network correction started (no clustering)" << endl;
                    }

                    bool nutUpdated = false;

                    if (applyNutCorrection && persistent_corrector) {
                        PyObject* predMethod = PyObject_GetAttrString(
                            persistent_corrector,
                            "predict_corrections_from_cartesian"
                        );
                        if (!predMethod) {
                            PyErr_Print();
                            FatalErrorInFunction
                                << "Unable to access predict_corrections_from_cartesian"
                                << exit(FatalError);
                        }

                        PyObject* predArgs = PyTuple_New(8);
                        PyTuple_SetItem(predArgs, 0, persistent_arrays[3]); Py_INCREF(persistent_arrays[3]);
                        PyTuple_SetItem(predArgs, 1, persistent_arrays[4]); Py_INCREF(persistent_arrays[4]);
                        PyTuple_SetItem(predArgs, 2, persistent_arrays[5]); Py_INCREF(persistent_arrays[5]);
                        PyTuple_SetItem(predArgs, 3, persistent_arrays[0]); Py_INCREF(persistent_arrays[0]);
                        PyTuple_SetItem(predArgs, 4, persistent_arrays[1]); Py_INCREF(persistent_arrays[1]);
                        PyTuple_SetItem(predArgs, 5, persistent_arrays[2]); Py_INCREF(persistent_arrays[2]);
                        PyTuple_SetItem(predArgs, 6, persistent_arrays[6]); Py_INCREF(persistent_arrays[6]);
                        PyTuple_SetItem(predArgs, 7, persistent_arrays[7]); Py_INCREF(persistent_arrays[7]);

                        PyObject* predResult = PyObject_CallObject(predMethod, predArgs);
                        Py_DECREF(predMethod);
                        Py_DECREF(predArgs);

                        if (!predResult) {
                            PyErr_Print();
                            FatalErrorInFunction << "predict_corrections call failed" << exit(FatalError);
                        }
                        if (!PyTuple_Check(predResult) || PyTuple_Size(predResult) < 1) {
                            Py_DECREF(predResult);
                            FatalErrorInFunction << "predict_corrections returned unexpected result" << exit(FatalError);
                        }

                        const Py_ssize_t predSize = PyTuple_Size(predResult);
                        PyObject* nut_corrected_py = PyTuple_GetItem(predResult, 0);

                        if (predSize >= 10)
                        {
                            PyObject* ax_py = PyTuple_GetItem(predResult, 7);
                            PyObject* rho_py = PyTuple_GetItem(predResult, 8);
                            PyObject* theta_py = PyTuple_GetItem(predResult, 9);
                            if (!PyArray_Check(ax_py) || !PyArray_Check(rho_py) || !PyArray_Check(theta_py))
                            {
                                Py_DECREF(predResult);
                                FatalErrorInFunction
                                    << "predict_corrections returned invalid geometry arrays"
                                    << exit(FatalError);
                            }

                            PyArrayObject* ax_np = reinterpret_cast<PyArrayObject*>(ax_py);
                            PyArrayObject* rho_np = reinterpret_cast<PyArrayObject*>(rho_py);
                            PyArrayObject* theta_np = reinterpret_cast<PyArrayObject*>(theta_py);
                            label axSize = static_cast<label>(PyArray_SIZE(ax_np));
                            label rhoSize = static_cast<label>(PyArray_SIZE(rho_np));
                            label thetaSize = static_cast<label>(PyArray_SIZE(theta_np));
                            if (axSize != nCells || rhoSize != nCells || thetaSize != nCells)
                            {
                                Py_DECREF(predResult);
                                FatalErrorInFunction
                                    << "Geometry result size mismatch with mesh cells"
                                    << exit(FatalError);
                            }

                            double* axData = static_cast<double*>(PyArray_DATA(ax_np));
                            double* rhoData = static_cast<double*>(PyArray_DATA(rho_np));
                            double* thetaData = static_cast<double*>(PyArray_DATA(theta_np));
                            scalarField& axFieldVals = axField.primitiveFieldRef();
                            scalarField& rhoFieldVals = rhoField.primitiveFieldRef();
                            scalarField& thetaFieldVals = thetaField.primitiveFieldRef();
                            for (label i = 0; i < nCells; ++i) {
                                axFieldVals[i] = scalar(axData[i]);
                                rhoFieldVals[i] = scalar(rhoData[i]);
                                thetaFieldVals[i] = scalar(thetaData[i]);
                            }
                            axField.correctBoundaryConditions();
                            rhoField.correctBoundaryConditions();
                            thetaField.correctBoundaryConditions();
                        }

                        if (applyNutCorrection && nutFieldPtr && PyArray_Check(nut_corrected_py)) {
                            PyArrayObject* nutArray = reinterpret_cast<PyArrayObject*>(nut_corrected_py);
                            double* nutCorrectedData = static_cast<double*>(PyArray_DATA(nutArray));
                            label nutSize = static_cast<label>(PyArray_SIZE(nutArray));

                            scalarField& nutPrimitiveField = nutFieldPtr->primitiveFieldRef();
                            for (label i = 0; i < min(nCells, nutSize); ++i) {
                                double value = nutCorrectedData[i];
                                if (value < 0.0) {
                                    value = 0.000015;
                                }
                                nutPrimitiveField[i] = value;
                            }
                            nutFieldPtr->correctBoundaryConditions();
                            nutUpdated = true;

                            if (Pstream::master())
                            {
                                Info<< "AI nut correction applied" << endl;
                                Info<< "Final corrected nut range: [" << min(nutPrimitiveField)
                                    << ", " << max(nutPrimitiveField) << "]" << endl;
                            }
                        }

                        Py_DECREF(predResult);
                    }

                    neuralNetworkTime = runTime.elapsedClockTime() - neuralNetworkStart;
                    scalar totalTimestepTime = runTime.elapsedClockTime() - timestepStart;
                    if (Pstream::master())
                    {
                        Info<< "Neural network correction completed" << endl;
                        Info<< "Neural network time: " << neuralNetworkTime << " s" << endl;
                        Info<< "Total AI pipeline time: " << totalTimestepTime << " s" << endl;
                    }

                    if (firstIteration && nutUpdated) {
                        firstIteration = false;
                        Info<< "*** FIRST ITERATION COMPLETED - AI CORRECTION ACTIVE (nut) ***" << endl;
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

    // Cleanup Python - MEMORY LEAK SAFE
    if (pythonInitialized) {
        Info<< "Cleaning up Python objects..." << endl;

        // Cleanup persistent objects
        Py_XDECREF(persistent_corrector);

        // Cleanup persistent arrays
        for (int i = 0; i < NUM_PERSISTENT_ARRAYS; i++) {
            Py_XDECREF(persistent_arrays[i]);
        }

        // Cleanup modules
        Py_XDECREF(neuralNetworkModule);

        Py_Finalize();
        Info<< "✓ Python cleanup completed" << endl;
    }

    Info<< "End\n" << endl;
    return 0;
    } // End AI correction scope
} // End main

// ************************************************************************* 
