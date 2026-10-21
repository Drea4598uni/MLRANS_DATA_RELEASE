/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

Class
    Foam::RASModels::kOmegaSSTML

Group
    grpRASTurbulence

Description
    k-omega-SST turbulence model with optional frozen nut correction.

SourceFiles
    kOmegaSSTML.C

SeeAlso
    kOmegaSSTBase.H

\*---------------------------------------------------------------------------*/

#ifndef kOmegaSSTML_H
#define kOmegaSSTML_H

#include "fvCFD.H"
#include "RASModel.H"
#include "eddyViscosity.H"
#include "kOmegaSSTBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                       Class kOmegaSSTML Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kOmegaSSTML
:
    public kOmegaSSTBase<eddyViscosity<RASModel<BasicTurbulenceModel>>>
{
    // Private Member Functions

        //- No copy construct
        kOmegaSSTML(const kOmegaSSTML&) = delete;

        //- No copy assignment
        void operator=(const kOmegaSSTML&) = delete;


protected:

    // Protected Data

        //- Flag to correct nut
        Switch solveNut_;


    // Protected Member Functions

        virtual void correctNut(const volScalarField& S2);
        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("kOmegaSSTML");


    // Constructors

        //- Construct from components
        kOmegaSSTML
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~kOmegaSSTML() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- Control whether nut is corrected
        void setSolveNut(const bool solveNut) noexcept
        {
            solveNut_ = solveNut;
        }

        //- Return whether nut is corrected
        bool solveNut() const noexcept
        {
            return solveNut_;
        }

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#ifdef NoRepository
    #include "kOmegaSSTML/kOmegaSSTML.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#endif

// ************************************************************************* //
