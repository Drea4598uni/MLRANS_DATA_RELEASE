// Benchmark phase-resolved della pipeline nativeML.
//   benchPipeline <param_file> <case_dir> [n_cells] [n_reps]
// Genera dati sintetici realistici e cronometra ogni fase di Pipeline::run
// (geometria, clustering, feature, MLP, output) mediando su n_reps ripetizioni
// a caldo. Legge NATIVE_ML_THREADS come il solver.

#include "../nativeML.H"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::fprintf(stderr,
            "uso: benchPipeline <param_file> <case_dir> [n_cells] [n_reps]\n");
        return 2;
    }
    const std::string paramFile = argv[1];
    const std::string caseDir = argv[2];
    const std::size_t n = (argc > 3) ? std::strtoul(argv[3], nullptr, 10) : 265000;
    const int reps = (argc > 4) ? std::atoi(argv[4]) : 20;

    try
    {
        nativeML::Pipeline pipeline(paramFile, caseDir);

        // Dati sintetici nel range fisico tipico della scia
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> ux(4, 14), uyz(-2, 2);
        std::uniform_real_distribution<double> px(-200, 800), py(-200, 200), pz(0, 300);
        std::uniform_real_distribution<double> dk(1e-4, 2.0), dnut(1e-5, 20.0);
        std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n), kk(n), nn(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] = px(rng); y[i] = py(rng); z[i] = pz(rng);
            vx[i] = ux(rng); vy[i] = uyz(rng); vz[i] = uyz(rng);
            kk[i] = dk(rng); nn[i] = dnut(rng);
        }

        // warm-up
        pipeline.run(x.data(), y.data(), z.data(), vx.data(), vy.data(),
                     vz.data(), kk.data(), nn.data(), n);

        nativeML::PipelineTimings tm;
        const double wall0 = nativeML::nowSeconds();
        for (int r = 0; r < reps; ++r)
        {
            pipeline.run(x.data(), y.data(), z.data(), vx.data(), vy.data(),
                         vz.data(), kk.data(), nn.data(), n, &tm);
        }
        const double wall = (nativeML::nowSeconds() - wall0)/reps*1e3;

        const double us = 1e6/reps;   // s cumulati -> us medi
        const double geo = tm.geometry*us, clu = tm.clustering*us;
        const double fea = tm.features*us, ml = tm.mlp*us, out = tm.output*us;
        const double tot = geo + clu + fea + ml + out;

        int threads = 1;
        #ifdef _OPENMP
        threads = std::atoi(nativeML::envOr("NATIVE_ML_THREADS", "1").c_str());
        if (threads < 1) threads = 1;
        #endif

        std::printf("=== benchPipeline  n=%zu  reps=%d  threads=%d ===\n",
                    n, reps, threads);
        std::printf("  %-12s %10s %8s %10s\n", "fase", "us/rep", "%", "ns/cella");
        auto row = [&](const char* nm, double v)
        {
            std::printf("  %-12s %10.1f %7.1f%% %10.2f\n",
                        nm, v, 100.0*v/tot, v*1000.0/n);
        };
        row("geometria", geo);
        row("clustering", clu);
        row("feature", fea);
        row("mlp", ml);
        row("output", out);
        std::printf("  %-12s %10.1f %7.1f%% %10.2f\n", "TOTALE", tot, 100.0, tot*1000.0/n);
        std::printf("  wall run() medio: %.3f ms  (%.2f ns/cella)\n",
                    wall, wall*1e6/n);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERRORE: %s\n", e.what());
        return 1;
    }
    return 0;
}
