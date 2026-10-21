// Stampa la geometria calcolata da nativeML::Geometry per un case dir:
//   geomCheck <case_dir>
// Output confrontabile con la geometria della pipeline Python.

#include "../nativeML.H"

#include <cstdio>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::fprintf(stderr, "uso: geomCheck <case_dir>\n");
        return 2;
    }

    try
    {
        nativeML::Geometry g;
        g.initialise(argv[1]);
        std::printf("turbine %s\n",
                    g.turbineName.empty() ? "-" : g.turbineName.c_str());
        std::printf("rotor_radius %.17g\n", g.rotorRadius);
        std::printf("axis_dir %.17g %.17g %.17g\n",
                    g.axisDir.x, g.axisDir.y, g.axisDir.z);
        std::printf("ref_axis %.17g %.17g %.17g\n",
                    g.refAxis.x, g.refAxis.y, g.refAxis.z);
        std::printf("point_shift %.17g %.17g %.17g\n",
                    g.pointShift.x, g.pointShift.y, g.pointShift.z);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERRORE: %s\n", e.what());
        return 1;
    }

    return 0;
}
