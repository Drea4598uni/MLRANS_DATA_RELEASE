#!/usr/bin/env python3
"""
Esporta la pipeline ML (KMeans + scaler + rete Keras) in un unico file di
testo (models/native_ml.dat, formato v2) leggibile dalla libreria C++
nativeML.H, eliminando Python/TensorFlow a runtime nel solver.

Robustezza (obiettivo di progetto): l'export NON assume una architettura
fissa. La rete Keras viene esportata come GRAFO generico di nodi
topologicamente ordinati, quindi tollera cambi di:
  - numero di layer, neuroni, profondita';
  - presenza/assenza di LayerNorm/BatchNorm, connessioni residue (Add/…);
  - funzioni di attivazione (relu, gelu, elu, selu, tanh, sigmoid, swish,
    softplus, softsign, hard_sigmoid, exponential, relu6, leaky_relu, prelu);
  - pesi e bias (letti dai layer);
  - ordine e numero delle feature (guidato dai nomi del preprocessor);
  - tipo di scaler (Standard/MinMax/MaxAbs/Robust: gestiti come affini,
    verificati numericamente; scaler non affini -> errore esplicito);
  - numero di cluster e feature del KMeans.
Qualunque cosa non riconosciuta produce un errore chiaro in fase di export
(mai un modello nativo silenziosamente sbagliato).

Uso (con il venv del solver):
    /home/frank1/andrea/acTestEnv/bin/python3.10 export_models.py \
        [--model-dir models] [--model-file best_model_nut.keras] \
        [--scaler-file y_scaler_nut.pkl] [--output models/native_ml.dat]
"""
import argparse
import os
import sys

import numpy as np

# numeri: float64 (17 cifre) per sklearn/scaler, float32 (9 cifre) per la rete
def f64(x):
    return "%.17g" % float(x)


def f32(x):
    return "%.9g" % float(np.float32(x))


def write_vec(fh, name, arr, fmt=f64):
    arr = np.asarray(arr).ravel()
    fh.write(f"{name} {arr.size}\n")
    fh.write(" ".join(fmt(v) for v in arr) + "\n")


def write_mat(fh, name, arr, fmt=f64):
    arr = np.asarray(arr)
    assert arr.ndim == 2, f"{name}: attesa matrice 2D, trovato {arr.shape}"
    fh.write(f"{name} {arr.shape[0]} {arr.shape[1]}\n")
    for row in arr:
        fh.write(" ".join(fmt(v) for v in row) + "\n")


# --------------------------------------------------------------------------- #
# Scaler affini generici (Standard/MinMax/MaxAbs/Robust) trattati come y = x*mul+add
# per colonna, ricavando mul/add per valutazione diretta e VERIFICANDO
# l'ipotesi affine numericamente. Cosi' l'export non dipende dal tipo esatto.
# --------------------------------------------------------------------------- #
def _assert_affine_forward(scaler, nfeat):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, nfeat))
    b = rng.normal(size=(4, nfeat))
    lhs = scaler.transform(0.5 * (a + b))
    rhs = 0.5 * (scaler.transform(a) + scaler.transform(b))
    if not np.allclose(lhs, rhs, rtol=1e-6, atol=1e-9):
        raise SystemExit(
            f"Scaler {type(scaler).__name__} non affine: non convertibile "
            f"in forma nativa x*mul+add"
        )


def affine_forward(scaler, nfeat):
    """mul, add per colonna tali che scaler.transform(x)[j] = x[j]*mul[j]+add[j]."""
    _assert_affine_forward(scaler, nfeat)
    add = scaler.transform(np.zeros((1, nfeat)))[0]
    diag = scaler.transform(np.eye(nfeat))
    mul = np.array([diag[j, j] - add[j] for j in range(nfeat)])
    return mul.astype(np.float64), add.astype(np.float64)


def affine_inverse(scaler, ntargets):
    """mul, add tali che scaler.inverse_transform(s)[j] = s[j]*mul[j]+add[j]."""
    add = scaler.inverse_transform(np.zeros((1, ntargets)))[0]
    diag = scaler.inverse_transform(np.eye(ntargets))
    mul = np.array([diag[j, j] - add[j] for j in range(ntargets)])
    # verifica affinita' inversa
    rng = np.random.default_rng(1)
    s = rng.normal(size=(4, ntargets))
    ref = scaler.inverse_transform(s)
    got = s * mul + add
    if not np.allclose(ref, got, rtol=1e-6, atol=1e-9):
        raise SystemExit(
            f"y-scaler {type(scaler).__name__} non affine invertibile"
        )
    return mul.astype(np.float64), add.astype(np.float64)


def export_kmeans(fh, model_dir):
    import joblib

    km_path = os.path.join(model_dir, "km")
    if not os.path.exists(km_path):
        # modello senza clustering: sezione vuota (n_clusters 0)
        fh.write("section kmeans\n")
        fh.write("n_clusters 0\n")
        print("[kmeans] assente -> modello no-clustering")
        return

    km = joblib.load(km_path)
    sc = joblib.load(os.path.join(model_dir, "km-scaler"))

    feat = [str(n) for n in getattr(sc, "feature_names_in_", [])]
    if not feat:
        raise SystemExit("km-scaler senza feature_names_in_: export impossibile")

    centers = np.asarray(km.cluster_centers_, dtype=np.float64)
    if centers.shape[1] != len(feat):
        raise SystemExit("Mismatch dimensioni centroidi/feature scaler")

    mul, add = affine_forward(sc, len(feat))

    fh.write("section kmeans\n")
    fh.write(f"n_clusters {int(km.n_clusters)}\n")
    fh.write(f"n_features {len(feat)}\n")
    fh.write("feature_names " + " ".join(feat) + "\n")
    # scaled = x*mul + add
    write_vec(fh, "scaler_mul", mul)
    write_vec(fh, "scaler_add", add)
    write_mat(fh, "centers", centers)
    print(f"[kmeans] n_clusters={km.n_clusters}, features={feat}, "
          f"scaler={type(sc).__name__}")


def export_preprocessor(fh, model_dir):
    import pickle

    with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
        prep = pickle.load(f)

    if not hasattr(prep, "transformers_"):
        raise SystemExit("preprocessor senza transformers_: export impossibile")

    num_names, num_mul, num_add = [], None, None
    onehot_source = None
    onehot_categories = []   # valore che attiva la colonna, in ordine

    for name, tr, cols in prep.transformers_:
        if name == "remainder":
            continue
        cols = [str(c) for c in cols]
        if isinstance(tr, str):
            if tr == "passthrough":
                # colonne pre-costruite (es. cluster_0..cluster_k) => one-hot su
                # clusterID: la colonna cluster_c si attiva per new_ID == c+1
                for c in cols:
                    if not c.startswith("cluster_"):
                        raise SystemExit(
                            f"Colonna passthrough '{c}' non e' una one-hot "
                            f"cluster_*: gestione non supportata"
                        )
                    onehot_source = "clusterID"
                    onehot_categories.append(int(c.split("_")[1]) + 1)
            elif tr == "drop":
                pass
            else:
                raise SystemExit(f"Transformer stringa '{tr}' non supportato")
            continue

        cls = type(tr).__name__
        if cls in ("StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler"):
            num_names = cols
            num_mul, num_add = affine_forward(tr, len(cols))
        elif cls == "OneHotEncoder":
            if len(cols) != 1:
                raise SystemExit("OneHotEncoder atteso su una sola colonna")
            onehot_source = cols[0]
            cats = tr.categories_[0]
            onehot_categories = [int(round(float(v))) for v in cats]
        elif cls == "Pipeline":
            raise SystemExit(
                "Pipeline annidata nel ColumnTransformer non supportata: "
                "appiattirla o esportarne gli step separatamente"
            )
        else:
            raise SystemExit(f"Transformer '{name}' di tipo {cls} non supportato")

    if num_mul is None and not onehot_categories:
        raise SystemExit("Preprocessor senza colonne numeriche ne one-hot")

    fh.write("section preprocessor\n")
    n_num = 0 if num_mul is None else len(num_names)
    fh.write(f"n_num {n_num}\n")
    if n_num:
        fh.write("num_names " + " ".join(num_names) + "\n")
        write_vec(fh, "num_mul", num_mul)   # out = x*mul + add
        write_vec(fh, "num_add", num_add)
    fh.write(f"n_onehot {len(onehot_categories)}\n")
    if onehot_categories:
        fh.write(f"onehot_source {onehot_source}\n")
        fh.write("onehot_categories "
                 + " ".join(str(c) for c in onehot_categories) + "\n")
    print(f"[preprocessor] num={num_names}, "
          f"onehot(source={onehot_source})={onehot_categories}")
    return n_num + len(onehot_categories)


def export_y_scaler(fh, model_dir, scaler_file):
    import pickle

    with open(os.path.join(model_dir, scaler_file), "rb") as f:
        ys = pickle.load(f)

    if hasattr(ys, "n_features_in_"):
        nt = int(ys.n_features_in_)
    elif hasattr(ys, "scale_"):
        nt = int(np.asarray(ys.scale_).size)
    else:
        nt = 1
    mul, add = affine_inverse(ys, nt)
    fh.write("section y_scaler\n")
    fh.write(f"n_targets {nt}\n")
    write_vec(fh, "y_mul", mul)    # physical = scaled*mul + add
    write_vec(fh, "y_add", add)
    print(f"[y_scaler] n_targets={nt}, type={type(ys).__name__}")


# --------------------------------------------------------------------------- #
# Rete Keras -> grafo generico
# --------------------------------------------------------------------------- #
# Attivazioni supportate lato C++ (nativeML.H). L'export fallisce se ne trova
# una fuori da questo insieme.
SUPPORTED_ACT = {
    "linear", "relu", "relu6", "gelu", "gelu_tanh", "elu", "selu", "tanh",
    "sigmoid", "hard_sigmoid", "swish", "silu", "softplus", "softsign",
    "exponential",
}


def norm_activation(act):
    """Normalizza l'attivazione Keras (stringa o serializzata) a un nome noto."""
    if act is None:
        return "linear"
    if isinstance(act, dict):
        # attivazione serializzata (es. {'class_name':..., 'config':...})
        cls = act.get("class_name", "").lower()
        if cls:
            act = cls
        else:
            act = act.get("config", {}).get("activation", "linear")
    name = str(act).lower()
    if name in ("none", ""):
        name = "linear"
    if name == "silu":
        name = "swish"
    if name not in SUPPORTED_ACT:
        raise SystemExit(
            f"Attivazione '{name}' non supportata da nativeML.H. "
            f"Supportate: {sorted(SUPPORTED_ACT)}"
        )
    return name


def _tensor_key(t):
    try:
        return t.ref()
    except Exception:
        return id(t)


def build_graph_spec(model):
    """
    Costruisce la topologia in modo robusto per modelli Functional E Sequential.
    Ritorna:
      input_specs : lista (name, dim) dei nodi input (sintetizzati da model.inputs)
      real_layers : lista di layer Keras non-InputLayer
      inbound     : dict name -> lista nomi sorgente
    """
    model_inputs = model.inputs if isinstance(model.inputs, (list, tuple)) \
        else [model.inputs]

    input_specs = []
    out2node = {}
    for i, t in enumerate(model_inputs):
        nm = f"__input_{i}"
        input_specs.append((nm, int(t.shape[-1])))
        out2node[_tensor_key(t)] = nm

    real_layers = [L for L in model.layers if type(L).__name__ != "InputLayer"]
    for L in real_layers:
        try:
            outs = L.output
            outs = outs if isinstance(outs, (list, tuple)) else [outs]
            for o in outs:
                out2node[_tensor_key(o)] = L.name
        except Exception:
            pass

    inbound = {nm: [] for nm, _ in input_specs}
    ok = True
    for L in real_layers:
        try:
            ins = L.input
            ins = ins if isinstance(ins, (list, tuple)) else [ins]
            names = [out2node.get(_tensor_key(t)) for t in ins]
            if any(s is None for s in names):
                ok = False
            inbound[L.name] = names
        except Exception:
            ok = False
            inbound[L.name] = [None]

    if not ok:
        # Fallback sequenziale: catena, primo layer sull'input sintetico
        first_in = input_specs[0][0] if input_specs else None
        inbound = {nm: [] for nm, _ in input_specs}
        prev = first_in
        for L in real_layers:
            inbound[L.name] = [prev] if prev is not None else []
            prev = L.name

    return input_specs, real_layers, inbound


def topo_order(names, inbound):
    order, visited, temp = [], set(), set()

    def visit(nm):
        if nm in visited:
            return
        if nm in temp:
            raise SystemExit(f"Ciclo nel grafo della rete a '{nm}'")
        temp.add(nm)
        for src in inbound.get(nm, []):
            if src is not None:
                visit(src)
        temp.discard(nm)
        visited.add(nm)
        order.append(nm)

    for nm in names:
        visit(nm)
    return order


def layer_output_dim(layer):
    shape = layer.output.shape
    dim = shape[-1]
    return int(dim) if dim is not None else -1


def extract_node(layer):
    """Ritorna (type, params_writer) per un layer supportato."""
    cls = type(layer).__name__
    cfg = layer.get_config()

    if cls == "InputLayer":
        return ("input", None)

    if cls == "Dense":
        w = layer.get_weights()
        W = w[0]
        b = w[1] if len(w) > 1 else np.zeros(W.shape[1], dtype=W.dtype)
        act = norm_activation(cfg.get("activation"))

        def writer(fh):
            fh.write(f"act {act}\n")
            write_mat(fh, "W", W, f32)
            write_vec(fh, "b", b, f32)

        return ("dense", writer)

    if cls == "LayerNormalization":
        axis = cfg.get("axis", -1)
        if isinstance(axis, (list, tuple)):
            axis = list(axis)
        else:
            axis = [axis]
        if axis not in ([-1], [1]):
            raise SystemExit(f"LayerNorm axis {axis} non gestito")
        eps = float(cfg.get("epsilon", 1e-3))
        d = layer_output_dim(layer)
        weights = layer.get_weights()
        gamma = np.ones(d) if cfg.get("scale", True) else np.ones(d)
        beta = np.zeros(d) if cfg.get("center", True) else np.zeros(d)
        wi = 0
        if cfg.get("scale", True):
            gamma = weights[wi]; wi += 1
        if cfg.get("center", True):
            beta = weights[wi]; wi += 1

        def writer(fh):
            fh.write(f"eps {f64(eps)}\n")
            write_vec(fh, "gamma", gamma, f32)
            write_vec(fh, "beta", beta, f32)

        return ("layernorm", writer)

    if cls == "BatchNormalization":
        eps = float(cfg.get("epsilon", 1e-3))
        gamma, beta, mean, var = None, None, None, None
        weights = layer.get_weights()
        d = layer_output_dim(layer)
        idx = 0
        if cfg.get("scale", True):
            gamma = weights[idx]; idx += 1
        else:
            gamma = np.ones(d)
        if cfg.get("center", True):
            beta = weights[idx]; idx += 1
        else:
            beta = np.zeros(d)
        mean = weights[idx]; idx += 1
        var = weights[idx]; idx += 1

        def writer(fh):
            fh.write(f"eps {f64(eps)}\n")
            write_vec(fh, "gamma", gamma, f32)
            write_vec(fh, "beta", beta, f32)
            write_vec(fh, "mean", mean, f32)
            write_vec(fh, "var", var, f32)

        return ("batchnorm", writer)

    if cls == "Activation":
        act = norm_activation(cfg.get("activation"))
        return ("activation", lambda fh: fh.write(f"act {act}\n"))

    if cls == "ReLU":
        neg = float(cfg.get("negative_slope", 0.0) or 0.0)
        maxv = cfg.get("max_value", None)
        thr = float(cfg.get("threshold", 0.0) or 0.0)
        if neg == 0.0 and thr == 0.0 and (maxv is None):
            return ("activation", lambda fh: fh.write("act relu\n"))
        if neg == 0.0 and thr == 0.0 and float(maxv) == 6.0:
            return ("activation", lambda fh: fh.write("act relu6\n"))
        raise SystemExit(
            f"ReLU con (neg_slope={neg}, max={maxv}, thr={thr}) non gestita"
        )

    if cls == "LeakyReLU":
        alpha = float(cfg.get("alpha", cfg.get("negative_slope", 0.3)))

        def writer(fh):
            fh.write("act leaky_relu\n")
            fh.write(f"alpha {f32(alpha)}\n")

        return ("activation_p", writer)

    if cls == "ELU":
        alpha = float(cfg.get("alpha", 1.0))

        def writer(fh):
            fh.write("act elu_alpha\n")
            fh.write(f"alpha {f32(alpha)}\n")

        return ("activation_p", writer)

    if cls == "PReLU":
        alpha = np.asarray(layer.get_weights()[0]).ravel()

        def writer(fh):
            fh.write("act prelu\n")
            write_vec(fh, "alpha", alpha, f32)

        return ("activation_p", writer)

    if cls in ("Dropout", "SpatialDropout1D", "GaussianDropout",
               "GaussianNoise", "ActivityRegularization"):
        return ("identity", None)  # no-op in inferenza

    if cls in ("Add", "Multiply", "Subtract", "Average", "Concatenate"):
        kind = {"Add": "add", "Multiply": "mul", "Subtract": "sub",
                "Average": "avg", "Concatenate": "concat"}[cls]
        return (kind, None)

    raise SystemExit(
        f"Layer '{layer.name}' di tipo {cls} non supportato dall'export nativo"
    )


def export_mlp(fh, model_dir, model_file, expected_inputs):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras

    model = keras.models.load_model(
        os.path.join(model_dir, model_file), compile=False
    )

    input_specs, real_layers, inbound = build_graph_spec(model)
    by_name = {L.name: L for L in real_layers}
    input_dims = {nm: d for nm, d in input_specs}
    input_name_set = set(input_dims)

    all_names = [nm for nm, _ in input_specs] + [L.name for L in real_layers]
    order = topo_order(all_names, inbound)
    id_of = {nm: i for i, nm in enumerate(order)}

    # output del modello
    model_outputs = model.outputs if isinstance(model.outputs, (list, tuple)) \
        else [model.outputs]
    out_names = []
    for t in model_outputs:
        for L in real_layers:
            try:
                if _tensor_key(L.output) == _tensor_key(t):
                    out_names.append(L.name)
            except Exception:
                pass
    if not out_names:                 # fallback: ultimo layer in ordine topo
        out_names = [order[-1]]
    if len(out_names) != 1:
        raise SystemExit(
            f"Rete con {len(out_names)} output: solo output singolo supportato")

    fh.write("section mlp\n")
    fh.write("format graph\n")

    # dichiarazione input (ruoli: main/nut/aux)
    inp_list = [nm for nm in order if nm in input_name_set]
    fh.write(f"n_inputs {len(inp_list)}\n")
    n_main = 0
    for nm in inp_list:
        d = input_dims[nm]
        if "nut" in nm.lower() and d == 1:
            role = "nut"
        elif d == expected_inputs:
            role = "main"; n_main += 1
        else:
            role = "aux"
        fh.write(f"input {id_of[nm]} {nm} {d} {role}\n")

    if len(inp_list) == 1:
        # con un solo input, e' sempre il principale
        d = input_dims[inp_list[0]]
        if d != expected_inputs:
            raise SystemExit(
                f"Input rete={d} ma preprocessor produce {expected_inputs} feature")
    elif n_main == 0:
        raise SystemExit(
            f"Nessun input di dimensione {expected_inputs} (feature preprocessor)")

    # nodi
    fh.write(f"n_nodes {len(order)}\n")
    total_params = 0
    for nm in order:
        if nm in input_name_set:
            fh.write(f"node {id_of[nm]} input {input_dims[nm]} nin 0 \n")
            continue
        layer = by_name[nm]
        ntype, writer = extract_node(layer)
        odim = layer_output_dim(layer)
        ins = [id_of[s] for s in inbound.get(nm, []) if s is not None]
        fh.write(f"node {id_of[nm]} {ntype} {odim} nin {len(ins)} "
                 + " ".join(str(i) for i in ins) + "\n")
        if writer is not None:
            writer(fh)
        for w in layer.get_weights():
            total_params += np.asarray(w).size

    fh.write(f"output {id_of[out_names[0]]}\n")
    print(f"[mlp] nodi={len(order)}, input={len(inp_list)}, "
          f"output_dim={layer_output_dim(by_name[out_names[0]])}, "
          f"parametri={total_params}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--model-dir", default=os.path.join(here, "models"))
    ap.add_argument("--model-file", default="best_model_nut.keras")
    ap.add_argument("--scaler-file", default="y_scaler_nut.pkl")
    ap.add_argument("--output", default=None)
    ap.add_argument("--mlp-only", action="store_true",
                    help="esporta solo la sezione MLP (per test di robustezza)")
    args = ap.parse_args()

    out = args.output or os.path.join(args.model_dir, "native_ml.dat")

    with open(out, "w") as fh:
        fh.write("NATIVE_ML_EXPORT v2\n")
        name_hint = (args.model_file + " " + args.scaler_file).lower()
        mode = "absolute" if ("absolute" in name_hint or "direct" in name_hint) else "delta"
        transform = "log" if "log" in name_hint else "none"
        fh.write(f"target_mode_hint {mode}\n")
        fh.write(f"target_transform_hint {transform}\n")
        if args.mlp_only:
            # ricava la dim input dal modello e marca quell'input come 'main'
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from tensorflow import keras
            m = keras.models.load_model(
                os.path.join(args.model_dir, args.model_file), compile=False)
            inp = m.inputs if isinstance(m.inputs, (list, tuple)) else [m.inputs]
            in_dim = int(inp[0].shape[-1])
            export_mlp(fh, args.model_dir, args.model_file, in_dim)
        else:
            export_kmeans(fh, args.model_dir)
            n_features = export_preprocessor(fh, args.model_dir)
            export_y_scaler(fh, args.model_dir, args.scaler_file)
            export_mlp(fh, args.model_dir, args.model_file, n_features)
        fh.write("end\n")

    size_mb = os.path.getsize(out) / 1e6
    print(f"\n✓ Export v2{' (mlp-only)' if args.mlp_only else ''} "
          f"completato: {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
