#!/usr/bin/env python3
"""
Test di ROBUSTEZZA dell'export/loader nativo del grafo MLP.

Costruisce numerose architetture Keras diverse (profondita', neuroni,
attivazioni, con/senza LayerNorm/BatchNorm, residui, feature diverse,
multi-target), le esporta con export_models.py --mlp-only e confronta
l'inferenza nativa (testMLP) con model.predict di Keras. Se tutte le
architetture combaciano entro tolleranza, l'export e' robusto ai cambi di
iperparametri/feature/pesi/layer/attivazioni.

Uso:
    /home/frank1/andrea/acTestEnv/bin/python3.10 test_robustness.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
L = keras.layers

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.path.join(HERE, "..", "export_models.py")
TESTMLP = os.path.join(HERE, "testMLP")
PY = sys.executable


def dense_block(x, units, act, norm=None):
    h = L.Dense(units)(x)
    if norm == "layer":
        h = L.LayerNormalization()(h)
    elif norm == "batch":
        h = L.BatchNormalization()(h)
    h = L.Activation(act)(h)
    return h


def build_models():
    models = {}

    # 1. MLP semplice relu, 3 layer
    x = keras.Input(shape=(9,))
    h = dense_block(x, 64, "relu")
    h = dense_block(h, 32, "relu")
    y = L.Dense(1)(h)
    models["mlp_relu_3"] = keras.Model(x, y)

    # 2. tanh + sigmoid finale, feature diverse (12)
    x = keras.Input(shape=(12,))
    h = dense_block(x, 48, "tanh")
    h = dense_block(h, 24, "tanh")
    y = L.Dense(2, activation="sigmoid")(h)
    models["mlp_tanh_2out"] = keras.Model(x, y)

    # 3. residuale con LayerNorm + gelu (come il modello reale, ridotto)
    x = keras.Input(shape=(14,))
    h = L.Dense(40)(x)
    for _ in range(3):
        a = L.Dense(40)(h)
        a = L.LayerNormalization()(a)
        a = L.Activation("gelu")(a)
        a = L.Dropout(0.1)(a)
        s = L.Dense(40)(h)
        h = L.Add()([a, s])
    y = L.Dense(1)(h)
    models["resnet_ln_gelu"] = keras.Model(x, y)

    # 4. BatchNorm + elu + swish
    x = keras.Input(shape=(7,))
    h = dense_block(x, 32, "elu", norm="batch")
    h = dense_block(h, 16, "swish", norm="batch")
    y = L.Dense(1)(h)
    m = keras.Model(x, y)
    # randomizza le statistiche BN (mean/var non banali)
    for lay in m.layers:
        if isinstance(lay, L.BatchNormalization):
            w = lay.get_weights()
            w[2] = np.random.randn(*w[2].shape).astype(np.float32) * 0.5
            w[3] = np.abs(np.random.randn(*w[3].shape).astype(np.float32)) + 0.5
            lay.set_weights(w)
    models["bn_elu_swish"] = m

    # 5. profondo, selu + softplus, 6 layer
    x = keras.Input(shape=(10,))
    h = x
    for u in (80, 64, 48, 32, 24, 16):
        h = dense_block(h, u, "selu")
    y = L.Dense(1, activation="softplus")(h)
    models["deep_selu"] = keras.Model(x, y)

    # 6. LeakyReLU + Concatenate (due rami)
    x = keras.Input(shape=(11,))
    a = dense_block(x, 32, "relu")
    b = L.Dense(32)(x)
    b = L.LeakyReLU(alpha=0.15)(b)
    h = L.Concatenate()([a, b])
    h = dense_block(h, 24, "gelu")
    y = L.Dense(3)(h)
    models["concat_leaky"] = keras.Model(x, y)

    # 7. Dense con attivazione integrata (no layer Activation separato)
    x = keras.Input(shape=(8,))
    h = L.Dense(40, activation="relu")(x)
    h = L.Dense(20, activation="tanh")(h)
    y = L.Dense(1, activation="linear")(h)
    models["dense_builtin_act"] = keras.Model(x, y)

    # 8. Sequential (non funzionale) hard_sigmoid + relu6
    m = keras.Sequential([
        keras.Input(shape=(6,)),
        L.Dense(24, activation="relu6"),
        L.Dense(12, activation="hard_sigmoid"),
        L.Dense(1),
    ])
    models["sequential_mixed"] = m

    return models


def run_native(model, name, X):
    with tempfile.TemporaryDirectory() as td:
        model.save(os.path.join(td, "m.keras"))
        dat = os.path.join(td, "mlp.dat")
        r = subprocess.run(
            [PY, EXPORT, "--model-dir", td, "--model-file", "m.keras",
             "--mlp-only", "--output", dat],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"export fallito:\n{r.stdout}\n{r.stderr}")

        inp = os.path.join(td, "in.txt")
        with open(inp, "w") as f:
            f.write(f"{X.shape[0]} {X.shape[1]}\n")
            for row in X:
                f.write(" ".join("%.9g" % v for v in row) + "\n")
        outp = os.path.join(td, "out.txt")
        r = subprocess.run([TESTMLP, dat, inp, outp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"testMLP fallito:\n{r.stdout}\n{r.stderr}")
        data = np.loadtxt(outp, skiprows=1)
        return data.reshape(X.shape[0], -1)


def main():
    if not os.path.exists(TESTMLP):
        print(f"manca {TESTMLP}: compilarlo prima")
        return 2

    rng = np.random.default_rng(0)
    models = build_models()
    all_ok = True
    print(f"{'modello':22s} {'in':>3s} {'out':>3s} {'max_abs_err':>12s} "
          f"{'max_rel_err':>12s}  esito")
    for name, model in models.items():
        d = model.inputs[0].shape[-1]
        X = rng.normal(size=(512, d)).astype(np.float32)
        ref = np.asarray(model(X, training=False)).reshape(512, -1)
        got = run_native(model, name, X)
        abs_err = np.abs(got - ref)
        denom = np.maximum(np.abs(ref), 1e-3)
        rel_err = abs_err / denom
        ok = (abs_err.max() < 2e-3) or (rel_err.max() < 2e-3)
        all_ok &= ok
        print(f"{name:22s} {d:3d} {ref.shape[1]:3d} {abs_err.max():12.3e} "
              f"{rel_err.max():12.3e}  {'OK' if ok else 'FAIL'}")

    print("\n" + ("✓ EXPORT ROBUSTO: tutte le architetture combaciano col nativo"
                  if all_ok else "✗ ALCUNE ARCHITETTURE NON COMBACIANO"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
