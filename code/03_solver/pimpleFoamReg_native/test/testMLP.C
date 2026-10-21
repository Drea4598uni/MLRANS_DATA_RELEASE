// Harness di test per il solo grafo MLP nativo (robustezza architetture).
//   testMLP <mlp_only.dat> <inputs.txt> <out.txt>
// inputs.txt: prima riga "N D", poi N righe da D float. Scrive N righe da
// outputDim. Legge un file esportato con export_models.py --mlp-only.

#include "../nativeML.H"

#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::fprintf(stderr, "uso: testMLP <mlp.dat> <inputs.txt> <out.txt>\n");
        return 2;
    }
    try
    {
        nativeML::ParamReader r(argv[1]);
        r.expect("NATIVE_ML_EXPORT"); r.word();
        r.expect("target_mode_hint"); r.word();
        r.expect("target_transform_hint"); r.word();

        nativeML::GraphMLP mlp;
        mlp.read(r);

        std::ifstream in(argv[2]);
        std::size_t N = 0; int D = 0;
        in >> N >> D;
        if (D != mlp.inputDim)
        {
            std::fprintf(stderr, "input dim %d != mlp.inputDim %d\n", D, mlp.inputDim);
            return 1;
        }
        std::vector<float> X(N*D);
        for (std::size_t i = 0; i < N*D; ++i) in >> X[i];

        std::vector<double> y(N*mlp.outputDim);
        mlp.forward(X.data(), N, y.data());

        std::FILE* out = std::fopen(argv[3], "w");
        std::fprintf(out, "%zu %d\n", N, mlp.outputDim);
        for (std::size_t i = 0; i < N; ++i)
        {
            for (int o = 0; o < mlp.outputDim; ++o)
                std::fprintf(out, "%.9g ", y[i*mlp.outputDim + o]);
            std::fprintf(out, "\n");
        }
        std::fclose(out);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERRORE: %s\n", e.what());
        return 1;
    }
    return 0;
}
