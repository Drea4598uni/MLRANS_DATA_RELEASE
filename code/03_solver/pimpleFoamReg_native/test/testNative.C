// Test standalone della pipeline nativeML (senza OpenFOAM):
//   testNative <param_file> <case_dir> <inputs.txt> <output.txt>
// Legge gli input generati da gen_reference.py, esegue la pipeline C++ e
// scrive le stesse colonne di expected.txt per il confronto numerico.

#include "../nativeML.H"

#include <chrono>
#include <cstdio>
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "uso: testNative <param_file> <case_dir> <inputs> <output>\n";
        return 2;
    }

    try
    {
        const std::string paramFile = argv[1];
        const std::string caseDir = argv[2];

        std::ifstream in(argv[3]);
        if (!in.good())
        {
            std::cerr << "impossibile aprire " << argv[3] << "\n";
            return 2;
        }
        std::size_t n = 0;
        in >> n;
        std::vector<double> x(n), y(n), z(n), ux(n), uy(n), uz(n), k(n), nut(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            in >> x[i] >> y[i] >> z[i] >> ux[i] >> uy[i] >> uz[i]
               >> k[i] >> nut[i];
        }

        const auto t0 = std::chrono::steady_clock::now();
        nativeML::Pipeline pipeline(paramFile, caseDir);
        const auto t1 = std::chrono::steady_clock::now();

        const auto res = pipeline.run
        (
            x.data(), y.data(), z.data(),
            ux.data(), uy.data(), uz.data(),
            k.data(), nut.data(), n
        );
        const auto t2 = std::chrono::steady_clock::now();

        // seconda run per il timing a caldo
        const auto res2 = pipeline.run
        (
            x.data(), y.data(), z.data(),
            ux.data(), uy.data(), uz.data(),
            k.data(), nut.data(), n
        );
        const auto t3 = std::chrono::steady_clock::now();

        using ms = std::chrono::duration<double, std::milli>;
        std::cout << "load: " << ms(t1 - t0).count() << " ms, "
                  << "run(" << n << " celle): " << ms(t2 - t1).count()
                  << " ms, run a caldo: " << ms(t3 - t2).count() << " ms\n";
        std::cout << "target_mode=" << pipeline.targetMode
                  << ", transform=" << pipeline.targetTransform
                  << ", turbina=" << pipeline.geometry.turbineName << "\n";

        std::FILE* out = std::fopen(argv[4], "w");
        if (!out)
        {
            std::cerr << "impossibile aprire " << argv[4] << "\n";
            return 2;
        }
        std::fprintf(out, "%zu\n", n);
        for (std::size_t i = 0; i < n; ++i)
        {
            std::fprintf
            (
                out,
                "%.17g %.17g %.17g %d %.17g %.17g %.17g %.17g %.17g %.17g\n",
                res.uAx[i], res.uRad[i], res.uTan[i],
                res.clusterId[i], res.centroidDist[i],
                res.axCoord[i], res.rho[i], res.theta[i],
                res.nutCorrected[i], res.deltaNut[i]
            );
        }
        std::fclose(out);
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERRORE: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
