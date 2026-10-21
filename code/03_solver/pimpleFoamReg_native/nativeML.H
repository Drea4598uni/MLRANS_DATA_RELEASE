/*---------------------------------------------------------------------------*\
    nativeML.H

    Inferenza C++ nativa della pipeline ML del solver pimpleFoamReg:
      - frame cilindrico turbina (parsing di system/fvOptions e system/Trim.*)
      - MinMaxScaler + KMeans (clustering con riordino per media Ux)
      - StandardScaler + one-hot cluster (preprocessor)
      - MLP residuale (Dense -> LayerNorm -> GELU + skip Dense) in float32
      - StandardScaler inverso sul target (+ exp opzionale, modo delta/absolute)

    Replica esattamente il comportamento di clustering.py e neural_network.py
    in inferenza, leggendo i parametri esportati da export_models.py
    (models/native_ml.dat). Nessuna dipendenza da Python/TensorFlow/OpenFOAM:
    solo C++ standard (+ OpenMP se disponibile).

    Variabili d'ambiente rispettate (stesse dei moduli Python):
      NNTURBINE, CYL_ABS,
      NNCLUSTER_REORDER_VAR (default "Ux"),
      NNCLUSTER_REORDER_ASCENDING (default false),
      NNCORR_TARGET_MODE (delta|absolute), NNCORR_TARGET_TRANSFORM (none|log),
      NNCORR_ROTOR_DIAMETER (default 126.0),
      NATIVE_ML_THREADS (default 1; thread OpenMP per l'inferenza per-rank)
\*---------------------------------------------------------------------------*/

#ifndef nativeML_H
#define nativeML_H

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace nativeML
{

// * * * * * * * * * * * * * * * utility  * * * * * * * * * * * * * * * * * //

inline std::string envOr(const char* name, const std::string& fallback)
{
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::string(v) : fallback;
}

// Orologio monotono ad alta risoluzione, in un helper cosi' da non esporre
// <chrono> nel resto dell'interfaccia.
inline double nowSeconds()
{
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count())*1e-9;
}

inline std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

inline bool envFlag(const char* name)
{
    const std::string v = toLower(envOr(name, ""));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

inline std::string strip(const std::string& s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

inline std::vector<std::string> splitWs(const std::string& s)
{
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

inline bool contains(const std::string& s, const std::string& sub)
{
    return s.find(sub) != std::string::npos;
}

inline double toDouble(std::string tok)
{
    while (!tok.empty() && (tok.back() == ';')) tok.pop_back();
    return std::stod(tok);
}

// Estrae un vettore "( a b c )" da una riga, come fa il parser Python
inline std::vector<double> parseParenVector(const std::string& line)
{
    const auto open = line.find('(');
    const auto close = line.find(')', open == std::string::npos ? 0 : open);
    if (open == std::string::npos || close == std::string::npos)
    {
        throw std::runtime_error("nativeML: vettore '( ... )' non trovato: " + line);
    }
    std::string body = line.substr(open + 1, close - open - 1);
    for (auto& c : body) if (c == ';') c = ' ';
    std::vector<double> out;
    for (const auto& t : splitWs(body)) out.push_back(std::stod(t));
    return out;
}

// erf float32 con la stessa approssimazione razionale di Eigen
// (generic_fast_erf_float), cioe' il kernel usato da TensorFlow su CPU:
// polinomio puro senza branch, auto-vettorizzabile (accuratezza ~1e-7,
// verificata in test/testNative.C contro std::erf).
inline float fastErf(float x)
{
    x = std::min(std::max(x, -4.0f), 4.0f);
    const float x2 = x*x;

    float p = -2.72614225801306e-10f;
    p = p*x2 + 2.77068142495902e-08f;
    p = p*x2 - 2.10102402082508e-06f;
    p = p*x2 - 5.69250639462346e-05f;
    p = p*x2 - 7.34990630326855e-04f;
    p = p*x2 - 2.95459980854025e-03f;
    p = p*x2 - 1.60960333262415e-02f;
    p = p*x;

    float q = -1.45660718464996e-05f;
    q = q*x2 - 2.13374055278905e-04f;
    q = q*x2 - 1.68282697438203e-03f;
    q = q*x2 - 7.37332916720468e-03f;
    q = q*x2 - 1.42647390514189e-02f;

    return p/q;
}

struct Vec3
{
    double x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(double a, double b, double c) : x(a), y(b), z(c) {}

    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const
    {
        return Vec3(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);
    }
    double norm() const { return std::sqrt(dot(*this)); }
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(double a) const { return Vec3(x*a, y*a, z*a); }
    Vec3 normalised() const
    {
        const double n = norm();
        if (n == 0.0)
        {
            throw std::runtime_error("nativeML: vettore a norma nulla");
        }
        return Vec3(x/n, y/n, z/n);
    }
};

// * * * * * * * * geometria turbina / frame cilindrico * * * * * * * * * * //

class Geometry
{
public:
    std::string turbineName;
    Vec3 origin{0, 0, 0};
    Vec3 axisInput{-1, 0, 0};
    Vec3 refAxisBase{0, -1, 0};
    double rotorRadius = 63.0;
    double rotorDiameter = 126.0;
    double hubAxDist = 0.0;
    double zTower = 0.0;
    double yaw = 0.0;
    double tilt = 0.0;
    bool cylAbs = false;
    double uInfInfer = 0.0;   // |freeStreamVelocity| da fvOptions (0 = assente)

    // derivati
    Vec3 axisDir, refAxis, pointShift;

    // Replica ClusteringProcessor._load_case_config + _setup_rotated_frame
    void initialise(const std::string& caseDir)
    {
        rotorDiameter = std::stod(envOr("NNCORR_ROTOR_DIAMETER", "126.0"));
        if (rotorDiameter <= 0.0)
        {
            throw std::runtime_error("NNCORR_ROTOR_DIAMETER deve essere > 0");
        }
        rotorRadius = 0.5*rotorDiameter;

        loadCaseConfig(caseDir);
        setupRotatedFrame();
    }

    void computeCylindrical
    (
        const double* x, const double* y, const double* z,
        const double* ux, const double* uy, const double* uz,
        std::size_t n,
        double* uAx, double* uRad, double* uTan,
        double* axCoord, double* rho, double* theta
    ) const
    {
        const Vec3 a = axisDir;
        const Vec3 r = refAxis;
        const Vec3 shift = pointShift;
        const bool useAbs = cylAbs;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (long i = 0; i < static_cast<long>(n); ++i)
        {
            const Vec3 p = Vec3(x[i], y[i], z[i]) + shift;
            const double axp = p.dot(a);
            const Vec3 proj = p - a*axp;
            const double rd = proj.norm();

            const Vec3 cr = r.cross(proj);
            const double th = std::atan2(cr.dot(a), proj.dot(r));

            const double ct = std::cos(th);
            const double st = std::sin(th);

            const double ua = Vec3(ux[i], uy[i], uz[i]).dot(a);
            double ur = ct*uy[i] + st*uz[i];
            double ut = -st*uy[i] + ct*uz[i];
            if (useAbs)
            {
                ur = std::fabs(ur);
                ut = std::fabs(ut);
            }

            uAx[i] = ua;
            uRad[i] = ur;
            uTan[i] = ut;
            axCoord[i] = axp;
            rho[i] = rd;
            theta[i] = th;
        }
    }

private:
    struct TurbineEntry
    {
        bool hasRadius = false, hasOrigin = false, hasAxis = false;
        bool hasFreeStream = false;
        double radius = 0.0;
        Vec3 origin, axis, freeStream;
    };

    void loadCaseConfig(const std::string& caseDir)
    {
        std::map<std::string, TurbineEntry> turbines;

        // Come in Python: qualunque errore di parsing interrompe la lettura
        // del file mantenendo i valori gia' acquisiti (try/except esterno).
        const std::string fvPath = caseDir + "/system/fvOptions";
        std::ifstream fv(fvPath);
        if (fv.good())
        {
            try
            {
                std::string currentTurbine;
                std::string raw;
                while (std::getline(fv, raw))
                {
                    const std::string line = strip(raw);
                    if (line.empty() || line.rfind("//", 0) == 0
                     || line.rfind("/*", 0) == 0 || line.rfind("*", 0) == 0)
                    {
                        continue;
                    }
                    if (line.rfind("turbine", 0) == 0)
                    {
                        currentTurbine = splitWs(line)[0];
                        turbines[currentTurbine];  // crea entry
                    }
                    if (currentTurbine.empty()) continue;

                    TurbineEntry& t = turbines[currentTurbine];
                    if (contains(line, "rotorRadius"))
                    {
                        t.radius = toDouble(splitWs(line).at(1));
                        t.hasRadius = true;
                    }
                    else if (contains(line, "origin"))
                    {
                        const auto v = parseParenVector(line);
                        t.origin = Vec3(v.at(0), v.at(1), v.at(2));
                        t.hasOrigin = true;
                    }
                    else if (contains(line, "axis"))
                    {
                        const auto v = parseParenVector(line);
                        t.axis = Vec3(v.at(0), v.at(1), v.at(2));
                        t.hasAxis = true;
                    }
                    else if (contains(line, "freeStreamVelocity"))
                    {
                        const auto v = parseParenVector(line);
                        t.freeStream = Vec3(v.at(0), v.at(1), v.at(2));
                        t.hasFreeStream = true;
                    }
                }
            }
            catch (const std::exception&)
            {
                // parsing parziale: prosegui con quanto letto finora
            }
        }

        const std::string envTurbine = envOr("NNTURBINE", "");
        if (!envTurbine.empty() && turbines.count(envTurbine))
        {
            turbineName = envTurbine;
        }
        else if (!turbines.empty())
        {
            turbineName = turbines.begin()->first;  // std::map: gia' ordinata
        }

        if (!turbineName.empty())
        {
            const TurbineEntry& t = turbines[turbineName];
            if (t.hasOrigin) origin = t.origin;
            if (t.hasAxis) axisInput = t.axis;
            if (t.hasRadius) rotorRadius = t.radius;
            if (t.hasFreeStream) uInfInfer = t.freeStream.norm();
        }

        if (!turbineName.empty())
        {
            const std::string trimPath =
                caseDir + "/system/Trim." + turbineName;
            std::ifstream trim(trimPath);
            if (trim.good())
            {
                // Come in Python: nessuno skip dei commenti e un errore di
                // parsing interrompe la lettura mantenendo i valori acquisiti
                try
                {
                    std::string raw;
                    while (std::getline(trim, raw))
                    {
                        const std::string line = strip(raw);
                        // chiave = primo token (non sottostringa: 'rotorYaw_phase'
                        // non deve sovrascrivere 'rotorYaw')
                        const std::vector<std::string> tok = splitWs(line);
                        const std::string key = tok.empty() ? "" : tok[0];
                        if (key == "towerHeight")
                        {
                            zTower = toDouble(tok.at(1));
                        }
                        else if (key == "hubAxDist")
                        {
                            hubAxDist = -toDouble(tok.at(1));
                        }
                        else if (key == "rotorYaw")
                        {
                            yaw = toDouble(tok.at(1));
                        }
                        else if (key == "rotorTilt")
                        {
                            tilt = toDouble(tok.at(1));
                        }
                    }
                }
                catch (const std::exception&)
                {
                    // parsing parziale: prosegui con quanto letto finora
                }
            }
        }

        cylAbs = envFlag("CYL_ABS");
    }

    void setupRotatedFrame()
    {
        const double cy = std::cos(yaw), sy = std::sin(yaw);
        const double ct = std::cos(tilt), st = std::sin(tilt);

        // R = Ryaw * Rtilt
        const double R[3][3] =
        {
            {cy*ct, -sy, cy*st},
            {sy*ct,  cy, sy*st},
            {-st,   0.0, ct}
        };

        auto mul = [&R](const Vec3& v)
        {
            return Vec3
            (
                R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
                R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
                R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z
            );
        };

        axisDir = (mul(axisInput)*(-1.0)).normalised();
        refAxis = mul(refAxisBase).normalised();
        // Il training (sim_clustering, utils.points_CART2CYL) centra il frame
        // nel mozzo: x - origin - hubAxDist, z - origin - towerHeight.
        const Vec3 hub(origin.x + hubAxDist, origin.y, origin.z + zTower);
        pointShift = hub*(-1.0);
    }
};

// * * * * * * * * * * * * * lettura file parametri * * * * * * * * * * * * //

class ParamReader
{
public:
    explicit ParamReader(const std::string& path) : in_(path), path_(path)
    {
        if (!in_.good())
        {
            throw std::runtime_error("nativeML: impossibile aprire " + path);
        }
    }

    std::string word()
    {
        std::string w;
        if (!(in_ >> w))
        {
            throw std::runtime_error("nativeML: EOF inatteso in " + path_);
        }
        return w;
    }

    void expect(const std::string& w)
    {
        const std::string got = word();
        if (got != w)
        {
            throw std::runtime_error
            (
                "nativeML: atteso '" + w + "', trovato '" + got + "' in " + path_
            );
        }
    }

    long integer() { return std::stol(word()); }
    double number() { return std::stod(word()); }

    std::vector<std::string> names(std::size_t n)
    {
        std::vector<std::string> out(n);
        for (auto& s : out) s = word();
        return out;
    }

    template<class T>
    std::vector<T> vec(const std::string& key)
    {
        expect(key);
        const long n = integer();
        std::vector<T> out(n);
        for (auto& v : out) v = static_cast<T>(number());
        return out;
    }

    template<class T>
    std::vector<T> mat(const std::string& key, long& rows, long& cols)
    {
        expect(key);
        rows = integer();
        cols = integer();
        std::vector<T> out(rows*cols);
        for (auto& v : out) v = static_cast<T>(number());
        return out;
    }

private:
    std::ifstream in_;
    std::string path_;
};

// * * * * * * * * * * * * * * * * KMeans * * * * * * * * * * * * * * * * * //

class KMeansClassifier
{
public:
    bool enabled = true;                    // false = modello senza clustering
    int nClusters = 0;
    int nFeatures = 0;
    std::vector<std::string> featureNames;
    std::vector<double> mul, add;           // scaler affine: s = x*mul + add
    std::vector<double> centers;            // nClusters x nFeatures
    std::vector<int> remap;                 // label grezza -> new_ID 1-based
    std::string reorderVar;
    bool reorderApplied = false;

    void read(ParamReader& r)
    {
        r.expect("section"); r.expect("kmeans");
        r.expect("n_clusters"); nClusters = static_cast<int>(r.integer());
        if (nClusters == 0)
        {
            // modello no-clustering: nessun km, sezione vuota
            enabled = false;
            return;
        }
        r.expect("n_features"); nFeatures = static_cast<int>(r.integer());
        if (nFeatures < 1 || nFeatures > 64)
        {
            throw std::runtime_error("nativeML: n_features KMeans fuori range [1,64]");
        }
        r.expect("feature_names"); featureNames = r.names(nFeatures);
        mul = r.vec<double>("scaler_mul");
        add = r.vec<double>("scaler_add");
        long rows = 0, cols = 0;
        centers = r.mat<double>("centers", rows, cols);
        if (rows != nClusters || cols != nFeatures)
        {
            throw std::runtime_error("nativeML: dimensioni centroidi KMeans errate");
        }
        computeRemap();
    }

    // Replica il riordino di clustering.py: new_ID 1-based ordinati per la
    // media della variabile di riordino (default Ux, discendente)
    void computeRemap()
    {
        remap.assign(nClusters, 0);
        reorderVar = envOr("NNCLUSTER_REORDER_VAR", "Ux");
        const bool ascending = envFlag("NNCLUSTER_REORDER_ASCENDING");

        const auto it =
            std::find(featureNames.begin(), featureNames.end(), reorderVar);
        if (it == featureNames.end())
        {
            for (int i = 0; i < nClusters; ++i) remap[i] = i + 1;
            reorderApplied = false;
            return;
        }
        const int varIdx = static_cast<int>(it - featureNames.begin());

        std::vector<int> order(nClusters);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort
        (
            order.begin(), order.end(),
            [this, varIdx](int a, int b)
            {
                return centers[a*nFeatures + varIdx]
                     < centers[b*nFeatures + varIdx];
            }
        );
        if (!ascending) std::reverse(order.begin(), order.end());

        for (int rank = 0; rank < nClusters; ++rank)
        {
            remap[order[rank]] = rank + 1;
        }
        reorderApplied = true;
    }

    // features: n x nFeatures NON scalate (ordine featureNames).
    // Output: newID 1-based e distanza euclidea dal centroide (spazio scalato)
    void predict
    (
        const double* features, std::size_t n,
        int* newIds, double* centroidDist
    ) const
    {
        const int k = nClusters;
        const int f = nFeatures;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (long i = 0; i < static_cast<long>(n); ++i)
        {
            double s[64];  // nFeatures atteso piccolo
            for (int j = 0; j < f; ++j)
            {
                s[j] = features[i*f + j]*mul[j] + add[j];
            }

            int best = 0;
            double bestD2 = 1e300;
            for (int c = 0; c < k; ++c)
            {
                double d2 = 0.0;
                const double* ctr = &centers[c*f];
                for (int j = 0; j < f; ++j)
                {
                    const double d = s[j] - ctr[j];
                    d2 += d*d;
                }
                if (d2 < bestD2)
                {
                    bestD2 = d2;
                    best = c;
                }
            }
            newIds[i] = remap[best];
            centroidDist[i] = std::sqrt(bestD2);
        }
    }
};

// * * * * * * * * * * preprocessor e scaler target  * * * * * * * * * * * * //

struct Preprocessor
{
    // colonne numeriche: out = x*mul + add (scaler affine generico)
    std::vector<std::string> numNames;
    std::vector<double> numMul, numAdd;
    // one-hot: colonna j attiva quando la feature sorgente == oneHotIds[j]
    std::string oneHotSource;
    std::vector<int> oneHotIds;

    void read(ParamReader& r)
    {
        r.expect("section"); r.expect("preprocessor");
        r.expect("n_num");
        const long nn = r.integer();
        if (nn > 0)
        {
            r.expect("num_names"); numNames = r.names(nn);
            numMul = r.vec<double>("num_mul");
            numAdd = r.vec<double>("num_add");
        }
        r.expect("n_onehot");
        const long noh = r.integer();
        if (noh > 0)
        {
            r.expect("onehot_source"); oneHotSource = r.word();
            r.expect("onehot_categories");
            oneHotIds.resize(noh);
            for (auto& v : oneHotIds) v = static_cast<int>(r.integer());
        }
    }

    std::size_t nOutputs() const { return numNames.size() + oneHotIds.size(); }
};

struct YScaler
{
    // physical = scaled*mul + add (inversa affine generica)
    std::vector<double> mul, add;

    void read(ParamReader& r)
    {
        r.expect("section"); r.expect("y_scaler");
        r.expect("n_targets");
        const long nt = r.integer();
        mul = r.vec<double>("y_mul");
        add = r.vec<double>("y_add");
        if (static_cast<long>(mul.size()) != nt)
        {
            throw std::runtime_error("nativeML: y_scaler incoerente");
        }
    }
};

// * * * * * * * * * * * *  kernel vettorizzati (attivazioni) * * * * * * * * * //

constexpr float INV_SQRT2 = 0.70710678118654752f;

#ifdef __AVX512F__
// erf float32 vettoriale (stessa approssimazione razionale di fastErf/Eigen)
inline __m512 fastErf512(__m512 x)
{
    x = _mm512_max_ps(_mm512_set1_ps(-4.0f),
                      _mm512_min_ps(_mm512_set1_ps(4.0f), x));
    const __m512 x2 = _mm512_mul_ps(x, x);

    __m512 p = _mm512_set1_ps(-2.72614225801306e-10f);
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(2.77068142495902e-08f));
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(-2.10102402082508e-06f));
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(-5.69250639462346e-05f));
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(-7.34990630326855e-04f));
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(-2.95459980854025e-03f));
    p = _mm512_fmadd_ps(p, x2, _mm512_set1_ps(-1.60960333262415e-02f));
    p = _mm512_mul_ps(p, x);

    __m512 q = _mm512_set1_ps(-1.45660718464996e-05f);
    q = _mm512_fmadd_ps(q, x2, _mm512_set1_ps(-2.13374055278905e-04f));
    q = _mm512_fmadd_ps(q, x2, _mm512_set1_ps(-1.68282697438203e-03f));
    q = _mm512_fmadd_ps(q, x2, _mm512_set1_ps(-7.37332916720468e-03f));
    q = _mm512_fmadd_ps(q, x2, _mm512_set1_ps(-1.42647390514189e-02f));

    return _mm512_div_ps(p, q);
}
#endif

// Attivazioni supportate. Le calde (usate nei modelli reali) sono
// vettorizzate in AVX-512; le altre restano corrette ma scalari.
enum class Act
{
    Linear, Relu, Relu6, Gelu, GeluTanh, Elu, EluAlpha, Selu, Tanh,
    Sigmoid, HardSigmoid, Swish, Softplus, Softsign, Exponential,
    LeakyRelu, Prelu
};

inline Act parseAct(const std::string& s)
{
    if (s == "linear")       return Act::Linear;
    if (s == "relu")         return Act::Relu;
    if (s == "relu6")        return Act::Relu6;
    if (s == "gelu")         return Act::Gelu;
    if (s == "gelu_tanh")    return Act::GeluTanh;
    if (s == "elu")          return Act::Elu;
    if (s == "elu_alpha")    return Act::EluAlpha;
    if (s == "selu")         return Act::Selu;
    if (s == "tanh")         return Act::Tanh;
    if (s == "sigmoid")      return Act::Sigmoid;
    if (s == "hard_sigmoid") return Act::HardSigmoid;
    if (s == "swish" || s == "silu") return Act::Swish;
    if (s == "softplus")     return Act::Softplus;
    if (s == "softsign")     return Act::Softsign;
    if (s == "exponential")  return Act::Exponential;
    if (s == "leaky_relu")   return Act::LeakyRelu;
    if (s == "prelu")        return Act::Prelu;
    throw std::runtime_error("nativeML: attivazione '" + s + "' non supportata");
}

inline float actScalar(float x, Act a, float alpha, const float* av, int j)
{
    switch (a)
    {
        case Act::Linear: return x;
        case Act::Relu:   return x > 0.0f ? x : 0.0f;
        case Act::Relu6:  return x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x);
        case Act::Gelu:   return 0.5f*x*(1.0f + fastErf(x*INV_SQRT2));
        case Act::GeluTanh:
        {
            const float c = 0.7978845608028654f;   // sqrt(2/pi)
            const float t = c*(x + 0.044715f*x*x*x);
            return 0.5f*x*(1.0f + std::tanh(t));
        }
        case Act::Elu:      return x >= 0.0f ? x : std::expm1(x);
        case Act::EluAlpha: return x >= 0.0f ? x : alpha*std::expm1(x);
        case Act::Selu:
        {
            const float al = 1.6732632423543772f, sc = 1.0507009873554805f;
            return sc*(x >= 0.0f ? x : al*std::expm1(x));
        }
        case Act::Tanh:    return std::tanh(x);
        case Act::Sigmoid: return 1.0f/(1.0f + std::exp(-x));
        case Act::HardSigmoid:
        {
            const float r = 0.2f*x + 0.5f;
            return r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
        }
        case Act::Swish:    return x/(1.0f + std::exp(-x));
        case Act::Softplus: return x > 20.0f ? x : std::log1p(std::exp(x));
        case Act::Softsign: return x/(1.0f + std::fabs(x));
        case Act::Exponential: return std::exp(x);
        case Act::LeakyRelu: return x >= 0.0f ? x : alpha*x;
        case Act::Prelu:     return x >= 0.0f ? x : av[j]*x;
    }
    return x;
}

// Applica l'attivazione in-place su una riga di d elementi contigui.
inline void applyAct(float* v, int d, Act a, float alpha, const float* av)
{
    if (a == Act::Linear) return;
#ifdef __AVX512F__
    if (a == Act::Relu || a == Act::Relu6 || a == Act::Gelu || a == Act::LeakyRelu)
    {
        const __m512 zero = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= d; j += 16)
        {
            __m512 x = _mm512_loadu_ps(v + j), r;
            if (a == Act::Relu)
            {
                r = _mm512_max_ps(zero, x);
            }
            else if (a == Act::Relu6)
            {
                r = _mm512_min_ps(_mm512_set1_ps(6.0f), _mm512_max_ps(zero, x));
            }
            else if (a == Act::Gelu)
            {
                const __m512 e = fastErf512(_mm512_mul_ps(x, _mm512_set1_ps(INV_SQRT2)));
                r = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x),
                                  _mm512_add_ps(_mm512_set1_ps(1.0f), e));
            }
            else // LeakyRelu
            {
                const __mmask16 m = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
                r = _mm512_mask_mul_ps(x, m, x, _mm512_set1_ps(alpha));
            }
            _mm512_storeu_ps(v + j, r);
        }
        for (; j < d; ++j) v[j] = actScalar(v[j], a, alpha, av, j);
        return;
    }
#endif
    for (int j = 0; j < d; ++j) v[j] = actScalar(v[j], a, alpha, av, j);
}

// LayerNormalization su una riga: out = (in-mean)/sqrt(var+eps)*gamma + beta
inline void layerNormRow
(
    float* out, const float* in, int d,
    const float* gamma, const float* beta, float eps
)
{
    float mean, var;
#ifdef __AVX512F__
    {
        __m512 s = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= d; j += 16) s = _mm512_add_ps(s, _mm512_loadu_ps(in + j));
        float sm = _mm512_reduce_add_ps(s);
        for (; j < d; ++j) sm += in[j];
        mean = sm/d;

        const __m512 mv = _mm512_set1_ps(mean);
        __m512 vs = _mm512_setzero_ps();
        j = 0;
        for (; j + 16 <= d; j += 16)
        {
            const __m512 dd = _mm512_sub_ps(_mm512_loadu_ps(in + j), mv);
            vs = _mm512_fmadd_ps(dd, dd, vs);
        }
        float sv = _mm512_reduce_add_ps(vs);
        for (; j < d; ++j) { const float dd = in[j] - mean; sv += dd*dd; }
        var = sv/d;
    }
#else
    {
        float sm = 0.0f;
        for (int j = 0; j < d; ++j) sm += in[j];
        mean = sm/d;
        float sv = 0.0f;
        for (int j = 0; j < d; ++j) { const float dd = in[j] - mean; sv += dd*dd; }
        var = sv/d;
    }
#endif
    const float rinv = 1.0f/std::sqrt(var + eps);
#ifdef __AVX512F__
    {
        const __m512 mv = _mm512_set1_ps(mean), rv = _mm512_set1_ps(rinv);
        int j = 0;
        for (; j + 16 <= d; j += 16)
        {
            __m512 x = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(in + j), mv), rv);
            x = _mm512_fmadd_ps(x, _mm512_loadu_ps(gamma + j), _mm512_loadu_ps(beta + j));
            _mm512_storeu_ps(out + j, x);
        }
        for (; j < d; ++j) out[j] = (in[j] - mean)*rinv*gamma[j] + beta[j];
    }
#else
    for (int j = 0; j < d; ++j) out[j] = (in[j] - mean)*rinv*gamma[j] + beta[j];
#endif
}

// GEMM lineare register-blocked a 8 righe: out[b][j] = bias[j] + sum_i in[b][i]*W[i][j].
// Accumulatori ESPLICITI (a0..a7) per garantire l'allocazione in registro
// (un array acc[8] verrebbe spillato da GCC, dimezzando il throughput).
// Le righe oltre nb sono comunque calcolate (input azzerato a monte) e i loro
// output ignorati: nessun branch nel loop caldo. W row-major (din x dout).
template<int NB>
inline void denseLinear
(
    float* out, const float* in, int ld, int /*nb*/,
    const float* W, const float* bias, int din, int dout
)
{
    static_assert(NB == 8, "denseLinear: microkernel per 8 righe");
#ifdef __AVX512F__
    for (int j0 = 0; j0 < dout; j0 += 16)
    {
        const int jl = std::min(16, dout - j0);
        const __mmask16 mask = static_cast<__mmask16>((1u << jl) - 1u);
        const __m512 b0 = _mm512_maskz_loadu_ps(mask, bias + j0);
        __m512 a0 = b0, a1 = b0, a2 = b0, a3 = b0, a4 = b0, a5 = b0, a6 = b0, a7 = b0;
        for (int i = 0; i < din; ++i)
        {
            const __m512 w = _mm512_maskz_loadu_ps(mask, W + static_cast<std::size_t>(i)*dout + j0);
            const float* xi = in + i;
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(xi[0*ld]), w, a0);
            a1 = _mm512_fmadd_ps(_mm512_set1_ps(xi[1*ld]), w, a1);
            a2 = _mm512_fmadd_ps(_mm512_set1_ps(xi[2*ld]), w, a2);
            a3 = _mm512_fmadd_ps(_mm512_set1_ps(xi[3*ld]), w, a3);
            a4 = _mm512_fmadd_ps(_mm512_set1_ps(xi[4*ld]), w, a4);
            a5 = _mm512_fmadd_ps(_mm512_set1_ps(xi[5*ld]), w, a5);
            a6 = _mm512_fmadd_ps(_mm512_set1_ps(xi[6*ld]), w, a6);
            a7 = _mm512_fmadd_ps(_mm512_set1_ps(xi[7*ld]), w, a7);
        }
        _mm512_mask_storeu_ps(out + 0*ld + j0, mask, a0);
        _mm512_mask_storeu_ps(out + 1*ld + j0, mask, a1);
        _mm512_mask_storeu_ps(out + 2*ld + j0, mask, a2);
        _mm512_mask_storeu_ps(out + 3*ld + j0, mask, a3);
        _mm512_mask_storeu_ps(out + 4*ld + j0, mask, a4);
        _mm512_mask_storeu_ps(out + 5*ld + j0, mask, a5);
        _mm512_mask_storeu_ps(out + 6*ld + j0, mask, a6);
        _mm512_mask_storeu_ps(out + 7*ld + j0, mask, a7);
    }
#else
    for (int b = 0; b < NB; ++b)
    {
        float* o = out + b*ld;
        std::copy(bias, bias + dout, o);
        const float* xi = in + b*ld;
        for (int i = 0; i < din; ++i)
        {
            const float v = xi[i];
            const float* w = W + static_cast<std::size_t>(i)*dout;
            for (int j = 0; j < dout; ++j) o[j] += v*w[j];
        }
    }
#endif
}

// * * * * * * * * * * * * *  MLP come grafo generico  * * * * * * * * * * * * //
//
// Interpreta una rete Keras arbitraria esportata da export_models.py (formato
// grafo): nodi in ordine topologico, connessioni arbitrarie (skip/residui),
// LayerNorm/BatchNorm opzionali, qualsiasi attivazione supportata. I buffer
// intermedi di ogni nodo vivono in uno scratch per-thread a stride uniforme.
class GraphMLP
{
public:
    enum class Op
    {
        Input, Dense, LayerNorm, BatchNorm, Activation, Identity,
        Add, Mul, Sub, Avg, Concat
    };

    struct Node
    {
        Op op = Op::Identity;
        int outDim = 0;
        std::vector<int> inputs;
        int bufOffset = 0;      // colonna di partenza nello scratch
        bool ownsBuffer = true;
        // dense
        int din = 0;
        std::vector<float> W, b;
        // norm
        float eps = 1e-3f;
        std::vector<float> gamma, beta;        // layernorm
        std::vector<float> normA, normB;       // batchnorm pre-fuso: x*A+B
        // attivazione
        Act act = Act::Linear;
        float alpha = 0.0f;
        std::vector<float> alphaVec;           // prelu
    };

    struct InputSpec { int nodeId; std::string name; int dim; std::string role; };

    int inputDim = 0;       // dim input principale (main)
    int outputDim = 0;
    int outputNode = -1;
    int mainInputNode = -1;
    int nutInputNode = -1;
    std::vector<Node> nodes;
    std::vector<InputSpec> inputs;

    #ifdef NATIVE_ML_PROFILE
    mutable double profDense_ = 0.0, profNorm_ = 0.0, profAct_ = 0.0, profElt_ = 0.0;
    #endif

    void read(ParamReader& r)
    {
        r.expect("section"); r.expect("mlp");
        r.expect("format");
        const std::string fmt = r.word();
        if (fmt != "graph")
        {
            throw std::runtime_error(
                "nativeML: formato MLP '" + fmt + "' non supportato (atteso 'graph'; "
                "rigenerare native_ml.dat con export_models.py aggiornato)");
        }

        r.expect("n_inputs");
        const int ni = static_cast<int>(r.integer());
        inputs.resize(ni);
        for (int k = 0; k < ni; ++k)
        {
            r.expect("input");
            InputSpec s;
            s.nodeId = static_cast<int>(r.integer());
            s.name = r.word();
            s.dim = static_cast<int>(r.integer());
            s.role = r.word();
            inputs[k] = s;
        }

        r.expect("n_nodes");
        const int N = static_cast<int>(r.integer());
        nodes.resize(N);
        for (int idx = 0; idx < N; ++idx)
        {
            r.expect("node");
            const int id = static_cast<int>(r.integer());
            if (id != idx)
            {
                throw std::runtime_error("nativeML: id nodo non sequenziale nel grafo");
            }
            const std::string type = r.word();
            Node& nd = nodes[idx];
            nd.outDim = static_cast<int>(r.integer());
            r.expect("nin");
            const int m = static_cast<int>(r.integer());
            nd.inputs.resize(m);
            for (int t = 0; t < m; ++t) nd.inputs[t] = static_cast<int>(r.integer());
            readNodeParams(r, type, nd);
        }

        r.expect("output");
        outputNode = static_cast<int>(r.integer());
        if (outputNode < 0 || outputNode >= N)
        {
            throw std::runtime_error("nativeML: nodo di output fuori range");
        }
        outputDim = nodes[outputNode].outDim;

        assignBuffers();
        wireInputs();
    }

    // X: n righe di inputDim (input principale). nutCol opzionale per un
    // eventuale secondo input di dimensione 1 con ruolo 'nut'.
    void forward(const float* X, std::size_t n, double* y,
                 const double* nutCol = nullptr) const
    {
        const int ld = totalWidth_;
        const long nTiles = (static_cast<long>(n) + NB - 1)/NB;

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            std::vector<float> scratch(static_cast<std::size_t>(NB)*ld);

            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (long t = 0; t < nTiles; ++t)
            {
                const long i0 = t*NB;
                const int nb = static_cast<int>(std::min<long>(NB, static_cast<long>(n) - i0));

                // riempi input principale
                float* mainBuf = scratch.data() + nodes[mainInputNode].bufOffset;
                for (int b = 0; b < nb; ++b)
                {
                    const float* srcRow = X + (i0 + b)*inputDim;
                    std::copy(srcRow, srcRow + inputDim, mainBuf + b*ld);
                }
                for (int b = nb; b < NB; ++b)
                    std::fill(mainBuf + b*ld, mainBuf + b*ld + inputDim, 0.0f);

                if (nutInputNode >= 0)
                {
                    float* nBuf = scratch.data() + nodes[nutInputNode].bufOffset;
                    for (int b = 0; b < NB; ++b)
                        nBuf[b*ld] = (b < nb && nutCol) ? static_cast<float>(nutCol[i0 + b]) : 0.0f;
                }

                for (int id = 0; id < static_cast<int>(nodes.size()); ++id)
                {
                    computeNode(id, scratch.data(), ld, nb);
                }

                const float* ob = scratch.data() + nodes[outputNode].bufOffset;
                for (int b = 0; b < nb; ++b)
                    for (int o = 0; o < outputDim; ++o)
                        y[(i0 + b)*outputDim + o] = static_cast<double>(ob[b*ld + o]);
            }
        }
    }

private:
    static constexpr int NB = 8;    // celle per blocco (microkernel a 8 righe)
    int totalWidth_ = 0;

    void readNodeParams(ParamReader& r, const std::string& type, Node& nd)
    {
        if (type == "input")            { nd.op = Op::Input; return; }
        if (type == "identity")         { nd.op = Op::Identity; return; }
        if (type == "add")              { nd.op = Op::Add; return; }
        if (type == "mul")              { nd.op = Op::Mul; return; }
        if (type == "sub")              { nd.op = Op::Sub; return; }
        if (type == "avg")              { nd.op = Op::Avg; return; }
        if (type == "concat")           { nd.op = Op::Concat; return; }

        if (type == "dense")
        {
            nd.op = Op::Dense;
            r.expect("act");
            nd.act = parseAct(r.word());
            long rows = 0, cols = 0;
            nd.W = r.mat<float>("W", rows, cols);
            nd.din = static_cast<int>(rows);
            nd.b = r.vec<float>("b");
            if (cols != nd.outDim || static_cast<long>(nd.b.size()) != nd.outDim)
                throw std::runtime_error("nativeML: dimensioni Dense incoerenti");
            return;
        }
        if (type == "layernorm")
        {
            nd.op = Op::LayerNorm;
            r.expect("eps"); nd.eps = static_cast<float>(r.number());
            nd.gamma = r.vec<float>("gamma");
            nd.beta = r.vec<float>("beta");
            return;
        }
        if (type == "batchnorm")
        {
            nd.op = Op::BatchNorm;
            r.expect("eps"); nd.eps = static_cast<float>(r.number());
            const auto g = r.vec<float>("gamma");
            const auto be = r.vec<float>("beta");
            const auto mean = r.vec<float>("mean");
            const auto var = r.vec<float>("var");
            const int d = nd.outDim;
            nd.normA.resize(d); nd.normB.resize(d);
            for (int j = 0; j < d; ++j)
            {
                const float a = g[j]/std::sqrt(var[j] + nd.eps);
                nd.normA[j] = a;
                nd.normB[j] = be[j] - mean[j]*a;
            }
            return;
        }
        if (type == "activation" || type == "activation_p")
        {
            nd.op = Op::Activation;
            r.expect("act");
            const std::string an = r.word();
            nd.act = parseAct(an);
            if (an == "leaky_relu" || an == "elu_alpha")
            {
                r.expect("alpha"); nd.alpha = static_cast<float>(r.number());
            }
            else if (an == "prelu")
            {
                nd.alphaVec = r.vec<float>("alpha");
            }
            return;
        }
        throw std::runtime_error("nativeML: tipo nodo '" + type + "' sconosciuto");
    }

    void assignBuffers()
    {
        totalWidth_ = 0;
        for (auto& nd : nodes)
        {
            if (nd.op == Op::Identity && nd.inputs.size() == 1)
            {
                nd.bufOffset = nodes[nd.inputs[0]].bufOffset;   // alias
                nd.ownsBuffer = false;
            }
            else
            {
                nd.bufOffset = totalWidth_;
                nd.ownsBuffer = true;
                totalWidth_ += nd.outDim;
            }
        }
    }

    void wireInputs()
    {
        mainInputNode = -1; nutInputNode = -1;
        for (const auto& s : inputs)
        {
            if (s.role == "main") { mainInputNode = s.nodeId; inputDim = s.dim; }
            else if (s.role == "nut") { nutInputNode = s.nodeId; }
            else if (s.role == "aux")
                throw std::runtime_error(
                    "nativeML: input ausiliario '" + s.name + "' non gestito");
        }
        if (mainInputNode < 0)
        {
            if (inputs.size() == 1)   // singolo input = principale
            {
                mainInputNode = inputs[0].nodeId;
                inputDim = inputs[0].dim;
            }
            else
            {
                throw std::runtime_error("nativeML: input principale non identificato");
            }
        }
    }

    void computeNode(int id, float* scratch, int ld, int nb) const
    {
        const Node& nd = nodes[id];
        float* out = scratch + nd.bufOffset;

        switch (nd.op)
        {
            case Op::Input:
            case Op::Identity:
                return;   // riempiti/aliasati altrove

            case Op::Dense:
            {
                const float* in = scratch + nodes[nd.inputs[0]].bufOffset;
                #ifdef NATIVE_ML_PROFILE
                const double t0 = nowSeconds();
                #endif
                denseLinear<NB>(out, in, ld, nb, nd.W.data(), nd.b.data(),
                                nd.din, nd.outDim);
                #ifdef NATIVE_ML_PROFILE
                profDense_ += nowSeconds() - t0;
                const double t1 = nowSeconds();
                #endif
                if (nd.act != Act::Linear)
                {
                    for (int b = 0; b < nb; ++b)
                        applyAct(out + b*ld, nd.outDim, nd.act, nd.alpha,
                                 nd.alphaVec.empty() ? nullptr : nd.alphaVec.data());
                }
                #ifdef NATIVE_ML_PROFILE
                profAct_ += nowSeconds() - t1;
                #endif
                return;
            }

            case Op::LayerNorm:
            {
                const float* in = scratch + nodes[nd.inputs[0]].bufOffset;
                #ifdef NATIVE_ML_PROFILE
                const double t0 = nowSeconds();
                #endif
                for (int b = 0; b < nb; ++b)
                    layerNormRow(out + b*ld, in + b*ld, nd.outDim,
                                 nd.gamma.data(), nd.beta.data(), nd.eps);
                #ifdef NATIVE_ML_PROFILE
                profNorm_ += nowSeconds() - t0;
                #endif
                return;
            }

            case Op::BatchNorm:
            {
                const float* in = scratch + nodes[nd.inputs[0]].bufOffset;
                const int d = nd.outDim;
                for (int b = 0; b < nb; ++b)
                {
                    const float* xi = in + b*ld;
                    float* o = out + b*ld;
                    for (int j = 0; j < d; ++j) o[j] = xi[j]*nd.normA[j] + nd.normB[j];
                }
                return;
            }

            case Op::Activation:
            {
                const float* in = scratch + nodes[nd.inputs[0]].bufOffset;
                #ifdef NATIVE_ML_PROFILE
                const double t0 = nowSeconds();
                #endif
                for (int b = 0; b < nb; ++b)
                {
                    std::copy(in + b*ld, in + b*ld + nd.outDim, out + b*ld);
                    applyAct(out + b*ld, nd.outDim, nd.act, nd.alpha,
                             nd.alphaVec.empty() ? nullptr : nd.alphaVec.data());
                }
                #ifdef NATIVE_ML_PROFILE
                profAct_ += nowSeconds() - t0;
                #endif
                return;
            }

            case Op::Add:
            case Op::Mul:
            case Op::Sub:
            case Op::Avg:
            {
                #ifdef NATIVE_ML_PROFILE
                const double t0 = nowSeconds();
                #endif
                const int d = nd.outDim;
                const int nin = static_cast<int>(nd.inputs.size());
                for (int b = 0; b < nb; ++b)
                {
                    float* o = out + b*ld;
                    const float* a0 = scratch + nodes[nd.inputs[0]].bufOffset + b*ld;
                    std::copy(a0, a0 + d, o);
                    for (int t = 1; t < nin; ++t)
                    {
                        const float* ai = scratch + nodes[nd.inputs[t]].bufOffset + b*ld;
                        if (nd.op == Op::Add || nd.op == Op::Avg)
                            for (int j = 0; j < d; ++j) o[j] += ai[j];
                        else if (nd.op == Op::Sub)
                            for (int j = 0; j < d; ++j) o[j] -= ai[j];
                        else // Mul
                            for (int j = 0; j < d; ++j) o[j] *= ai[j];
                    }
                    if (nd.op == Op::Avg)
                    {
                        const float inv = 1.0f/nin;
                        for (int j = 0; j < d; ++j) o[j] *= inv;
                    }
                }
                #ifdef NATIVE_ML_PROFILE
                profElt_ += nowSeconds() - t0;
                #endif
                return;
            }

            case Op::Concat:
            {
                for (int b = 0; b < nb; ++b)
                {
                    float* o = out + b*ld;
                    int col = 0;
                    for (int t = 0; t < static_cast<int>(nd.inputs.size()); ++t)
                    {
                        const Node& src = nodes[nd.inputs[t]];
                        const float* ai = scratch + src.bufOffset + b*ld;
                        std::copy(ai, ai + src.outDim, o + col);
                        col += src.outDim;
                    }
                }
                return;
            }
        }
    }
};

// * * * * * * * * * * * * * * pipeline completa  * * * * * * * * * * * * * //

struct PipelineResult
{
    std::vector<double> uAx, uRad, uTan;
    std::vector<int> clusterId;          // new_ID 1-based
    std::vector<double> centroidDist;
    std::vector<double> axCoord, rho, theta;
    std::vector<double> nutCorrected, deltaNut;
};

// Breakdown dei tempi (secondi) delle fasi di run(); opzionale.
struct PipelineTimings
{
    double geometry = 0.0;    // frame cilindrico
    double clustering = 0.0;  // scaler + KMeans
    double features = 0.0;    // standardizzazione + one-hot
    double mlp = 0.0;         // forward MLP
    double output = 0.0;      // inverse transform + nut corretta
    double total() const { return geometry + clustering + features + mlp + output; }
};

class Pipeline
{
public:
    Geometry geometry;
    KMeansClassifier kmeans;
    Preprocessor preprocessor;
    YScaler yScaler;
    GraphMLP mlp;

    std::string targetMode;       // delta | absolute
    std::string targetTransform;  // none | log

    // Ponte U∞: normalizzazione per velocita' (opt-in via NNCLUSTER_UINF_NORM).
    // Riporta il campo corrente nello spazio di velocita' del training, cosi'
    // il clustering e il MLP (addestrati a U∞_train) restano validi a qualunque
    // velocita'. r = U∞_train/U∞_infer; a U∞=U∞_train -> r=1 (identita').
    bool   uinfNorm = false;
    double uinfTrain = 0.0;
    double uinfInfer = 0.0;
    double uinfR = 1.0;

    Pipeline(const std::string& paramFile, const std::string& caseDir)
    {
        ParamReader r(paramFile);
        r.expect("NATIVE_ML_EXPORT");
        const std::string ver = r.word();
        if (ver != "v2")
        {
            throw std::runtime_error(
                "nativeML: versione file parametri '" + ver + "' non supportata "
                "(attesa v2; rigenerare con export_models.py aggiornato)");
        }
        r.expect("target_mode_hint");
        const std::string modeHint = r.word();
        r.expect("target_transform_hint");
        const std::string transformHint = r.word();

        kmeans.read(r);
        preprocessor.read(r);
        yScaler.read(r);
        mlp.read(r);
        r.expect("end");

        if (mlp.inputDim != static_cast<int>(preprocessor.nOutputs()))
        {
            throw std::runtime_error
            (
                "nativeML: input MLP != feature del preprocessor"
            );
        }

        targetMode = toLower(envOr("NNCORR_TARGET_MODE", modeHint));
        if (targetMode != "delta" && targetMode != "absolute")
        {
            throw std::runtime_error
            (
                "NNCORR_TARGET_MODE deve essere 'delta' o 'absolute'"
            );
        }
        targetTransform = toLower(envOr("NNCORR_TARGET_TRANSFORM", transformHint));
        if (targetTransform != "none" && targetTransform != "log")
        {
            throw std::runtime_error
            (
                "NNCORR_TARGET_TRANSFORM deve essere 'none' o 'log'"
            );
        }

        geometry.initialise(caseDir);

        // Ponte U∞ (opt-in). U∞_infer da NNCLUSTER_UINF_INFER o, se assente,
        // da freeStreamVelocity in fvOptions. Se mancano i dati, disattiva.
        uinfNorm = envFlag("NNCLUSTER_UINF_NORM");
        if (uinfNorm)
        {
            uinfTrain = std::stod(envOr("NNCLUSTER_UINF_TRAIN", "0"));
            uinfInfer = std::stod(envOr("NNCLUSTER_UINF_INFER", "0"));
            if (uinfInfer <= 0.0) uinfInfer = geometry.uInfInfer;
            if (uinfTrain > 0.0 && uinfInfer > 0.0)
            {
                uinfR = uinfTrain/uinfInfer;
            }
            else
            {
                uinfNorm = false;
                uinfR = 1.0;
            }
        }

        #ifdef _OPENMP
        const long nThreads = std::stol(envOr("NATIVE_ML_THREADS", "1"));
        omp_set_num_threads(static_cast<int>(std::max(1L, nThreads)));
        #endif
    }

    // Pipeline completa: coordinate/velocita' cartesiane -> nut corretta.
    // Tutti gli array di input hanno n elementi.
    PipelineResult run
    (
        const double* x, const double* y, const double* z,
        const double* ux, const double* uy, const double* uz,
        const double* k, const double* nut,
        std::size_t n,
        PipelineTimings* timings = nullptr
    ) const
    {
        PipelineResult res;
        res.uAx.resize(n); res.uRad.resize(n); res.uTan.resize(n);
        res.axCoord.resize(n); res.rho.resize(n); res.theta.resize(n);
        res.clusterId.resize(n); res.centroidDist.resize(n);
        res.nutCorrected.resize(n); res.deltaNut.resize(n);

        // Ponte U∞: scala gli input nello spazio del training (r=1 => no-op).
        // velocita' x r, k x r^2, nut x r; posizioni invariate. Gli output di
        // velocita' e nut vengono ri-portati alla velocita' fisica (/r) in coda.
        const double r = uinfR;
        const double* uxP = ux; const double* uyP = uy; const double* uzP = uz;
        const double* kP = k;   const double* nutP = nut;
        std::vector<double> uxN, uyN, uzN, kN, nutN;
        if (r != 1.0)
        {
            const double r2 = r*r;
            uxN.resize(n); uyN.resize(n); uzN.resize(n); kN.resize(n); nutN.resize(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                uxN[i] = ux[i]*r; uyN[i] = uy[i]*r; uzN[i] = uz[i]*r;
                kN[i]  = k[i]*r2; nutN[i] = nut[i]*r;
            }
            uxP = uxN.data(); uyP = uyN.data(); uzP = uzN.data();
            kP  = kN.data();  nutP = nutN.data();
        }

        double t0 = timings ? nowSeconds() : 0.0;

        // 1. frame cilindrico
        geometry.computeCylindrical
        (
            x, y, z, uxP, uyP, uzP, n,
            res.uAx.data(), res.uRad.data(), res.uTan.data(),
            res.axCoord.data(), res.rho.data(), res.theta.data()
        );

        if (timings) { const double t = nowSeconds(); timings->geometry += t - t0; t0 = t; }

        // 2. clustering (saltato per modelli no-clustering: clusterId/dist = 0)
        if (kmeans.enabled)
        {
            const int nf = kmeans.nFeatures;
            std::vector<double> clusterFeatures(n*nf);
            for (int j = 0; j < nf; ++j)
            {
                const double* src = resolveFeature
                (
                    kmeans.featureNames[j], res, x, y, z, uxP, uyP, uzP, kP, nutP,
                    nullptr, nullptr
                );
                for (std::size_t i = 0; i < n; ++i)
                {
                    clusterFeatures[i*nf + j] = src[i];
                }
            }
            kmeans.predict
            (
                clusterFeatures.data(), n,
                res.clusterId.data(), res.centroidDist.data()
            );
        }

        if (timings) { const double t = nowSeconds(); timings->clustering += t - t0; t0 = t; }

        // 3. feature MLP: numeriche affini + one-hot cluster (row-major, parallelo)
        const std::size_t nNum = preprocessor.numNames.size();
        const std::size_t nOh = preprocessor.oneHotIds.size();
        const std::size_t width = nNum + nOh;

        std::vector<double> zd(n), axd(n);
        const double invD = 1.0/geometry.rotorDiameter;
        for (std::size_t i = 0; i < n; ++i)
        {
            zd[i] = z[i]*invD;
            axd[i] = res.axCoord[i]*invD;
        }

        // Puntatori sorgente per ogni feature numerica (una volta sola)
        std::vector<const double*> numSrc(nNum);
        for (std::size_t j = 0; j < nNum; ++j)
        {
            numSrc[j] = resolveFeature
            (
                preprocessor.numNames[j], res, x, y, z, uxP, uyP, uzP, kP, nutP,
                zd.data(), axd.data()
            );
        }
        if (nOh && preprocessor.oneHotSource != "clusterID")
        {
            throw std::runtime_error(
                "nativeML: one-hot su feature '" + preprocessor.oneHotSource
                + "' non supportata (solo 'clusterID')");
        }

        std::vector<float> X(n*width);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (long i = 0; i < static_cast<long>(n); ++i)
        {
            float* row = X.data() + static_cast<std::size_t>(i)*width;
            for (std::size_t j = 0; j < nNum; ++j)
            {
                row[j] = static_cast<float>(
                    numSrc[j][i]*preprocessor.numMul[j] + preprocessor.numAdd[j]);
            }
            const int cid = res.clusterId[i];
            for (std::size_t j = 0; j < nOh; ++j)
            {
                row[nNum + j] = (cid == preprocessor.oneHotIds[j]) ? 1.0f : 0.0f;
            }
        }

        if (timings) { const double t = nowSeconds(); timings->features += t - t0; t0 = t; }

        // 4. MLP (grafo generico). nut passato come eventuale input secondario.
        std::vector<double> yPred(n*mlp.outputDim);
        mlp.forward(X.data(), n, yPred.data(), nutP);

        if (timings) { const double t = nowSeconds(); timings->mlp += t - t0; t0 = t; }

        // 5. inverse transform target: physical = scaled*mul + add
        const double yMul = yScaler.mul[0];
        const double yAdd = yScaler.add[0];
        const bool useLog = (targetTransform == "log");
        const bool deltaMode = (targetMode == "delta");
        // Ponte U∞: l'output del modello e' nello spazio del training (nut ~ U∞);
        // riportalo alla velocita' fisica dividendo per r. Anche le velocita'
        // cilindriche in res sono scalate x r -> ripristina il valore fisico.
        const double invR = 1.0/r;
        const bool denorm = (r != 1.0);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (long i = 0; i < static_cast<long>(n); ++i)
        {
            double t = yPred[i*mlp.outputDim]*yMul + yAdd;
            if (useLog)
            {
                t = std::exp(std::min(std::max(t, -745.0), 709.0));
            }
            double nutC, dNut;
            if (deltaMode)
            {
                dNut = t;
                nutC = nutP[i] + t;
            }
            else
            {
                nutC = t;
                dNut = t - nutP[i];
            }
            res.nutCorrected[i] = nutC*invR;
            res.deltaNut[i]     = dNut*invR;
            if (denorm)
            {
                res.uAx[i]  *= invR;
                res.uRad[i] *= invR;
                res.uTan[i] *= invR;
            }
        }

        if (timings) { const double t = nowSeconds(); timings->output += t - t0; }

        return res;
    }

private:
    // Mappa nome feature -> array sorgente (stessa feature_map di
    // clustering.py/neural_network.py per i nomi usati dai modelli correnti)
    const double* resolveFeature
    (
        const std::string& name,
        const PipelineResult& res,
        const double* x, const double* y, const double* z,
        const double* ux, const double* uy, const double* uz,
        const double* k, const double* nut,
        const double* zd, const double* axd
    ) const
    {
        if (name == "Ux" || name == "u_axial") return res.uAx.data();
        if (name == "Ur" || name == "u_radial") return res.uRad.data();
        if (name == "Ut" || name == "u_tangential") return res.uTan.data();
        if (name == "Uy") return uy;
        if (name == "Uz") return uz;
        if (name == "x") return x;
        if (name == "y") return y;
        if (name == "z" || name == "z_original") return z;
        if (name == "k" || name == "kMean") return k;
        if (name == "nut" || name == "nutMean") return nut;
        if (name == "ax") return res.axCoord.data();
        if (name == "rho") return res.rho.data();
        if (name == "theta") return res.theta.data();
        if (name == "centroid_dist") return res.centroidDist.data();
        if (name == "z_d" && zd) return zd;
        if (name == "ax_d" && axd) return axd;
        throw std::runtime_error
        (
            "nativeML: feature '" + name + "' non supportata"
        );
    }
};

} // namespace nativeML

#endif

// ************************************************************************* //
