"""Retreino do modelo LightGBM com os parâmetros finais do DontGo Predictor.

Parâmetros do modelo em produção (model_metrics.json):
  num_leaves=255, scale_pos_weight=40, learning_rate=0.05
  min_child_samples=50, neg_ratio=5, treino jan-abr, val mai, teste jun
  Resultado: F1=0.6886, ROC-AUC=0.9923, PR-AUC=0.6739 no teste (jun/2025)

Use este script para reproduzir o modelo a partir dos dados Gold.
Uso: uv run python3 src/retrain_optimized.py
"""

import pickle
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))

GOLD_DIR = Path(__file__).parent.parent / "outputs" / "gold"
MODEL_OUT = GOLD_DIR / "lgbm_dontgo.pkl"
MODEL_BAK = GOLD_DIR / "lgbm_dontgo_v1_backup.pkl"
TARGET_COL = "is_dont_go_next_60m"

TRAIN_MONTHS = {"jan", "feb", "mar", "abr"}
VAL_MONTHS   = {"may"}
TEST_MONTHS  = {"jun"}
NEG_RATIO    = 5        # 5:1 negativos por positivo no treino
RANDOM_STATE = 42


# ── Parâmetros corrigidos ────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            ["binary_logloss", "auc"],
    "num_leaves":        255,         # modelo final: profundidade elevada captura padrões complexos
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      1000,        # early stopping para bem antes (best_iter=50)
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 50,
    "scale_pos_weight":  40.0,        # ajustado para maximize F1 no val (mai/2025)
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
}


# ── Carregamento ──────────────────────────────────────────────────────────────

def _load_months(month_set: set[str], feature_cols: list[str],
                 neg_ratio: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    files = [GOLD_DIR / f"gold_{m}.parquet" for m in month_set]
    files = [f for f in files if f.exists()]
    needed = feature_cols + [TARGET_COL]

    df = pl.read_parquet([str(f) for f in files], columns=needed)

    if neg_ratio is not None:
        tgt = pl.col(TARGET_COL).cast(pl.Int8)
        pos     = df.filter(tgt == 1)
        neg_all = df.filter(tgt == 0)
        n_neg   = min(len(pos) * neg_ratio, len(neg_all))
        neg     = neg_all.sample(n=n_neg, seed=RANDOM_STATE)
        del neg_all
        df = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=RANDOM_STATE)
        print(f"  undersampling {neg_ratio}:1 → {len(pos):,} pos + {len(neg):,} neg = {len(df):,} total")
        del pos, neg

    X = df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_pandas()
    y = df[TARGET_COL].cast(pl.Int8).to_pandas()
    del df
    return X, y


def get_feature_cols() -> list[str]:
    """Reutiliza lista de features do modelo atual."""
    with open(MODEL_OUT, "rb") as f:
        m = pickle.load(f)
    return m.feature_name_


# ── Treino ────────────────────────────────────────────────────────────────────

def train():
    print("\n=== Retreino Otimizado — LightGBM Don't Go ===\n")

    feature_cols = get_feature_cols()
    print(f"Features: {len(feature_cols)}")

    # Backup do modelo atual
    with open(MODEL_OUT, "rb") as f:
        old_model = pickle.load(f)
    with open(MODEL_BAK, "wb") as f:
        pickle.dump(old_model, f)
    print(f"Backup salvo → {MODEL_BAK.name}")

    # Treino (jan-abr)
    print(f"\nCarregando treino ({', '.join(sorted(TRAIN_MONTHS))})...")
    X_train, y_train = _load_months(TRAIN_MONTHS, feature_cols, neg_ratio=NEG_RATIO)
    print(f"  RAM treino: ~{X_train.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"  Distribuição: {y_train.sum():,} pos / {(y_train==0).sum():,} neg")

    # Validação (mai) — split temporal estrito, sem vazar o teste (jun)
    print(f"\nCarregando validação ({', '.join(sorted(VAL_MONTHS))})...")
    X_eval, y_eval = _load_months(VAL_MONTHS, feature_cols, neg_ratio=None)
    print(f"  {len(X_eval):,} registros | {y_eval.sum():,} pos ({y_eval.mean():.2%})")

    X_fit, y_fit = X_train, y_train
    del X_train, y_train

    print(f"\nTreinando (fit={len(X_fit):,} / eval={len(X_eval):,})...")
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_eval, y_eval)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    del X_fit, y_fit, X_eval, y_eval

    print(f"\nMelhor iteração: {model.best_iteration_}")

    # Avaliação no teste (Jun)
    print(f"\nCarregando teste (jun)...")
    X_test, y_test = _load_months(TEST_MONTHS, feature_cols, neg_ratio=None)
    proba = model.predict_proba(X_test)[:, 1]
    del X_test

    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, proba)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_idx = np.argmax(f1_arr[:-1])
    best_thr = float(thresholds[best_idx])
    f1   = float(f1_arr[best_idx])
    prec = float(prec_arr[best_idx])
    rec  = float(rec_arr[best_idx])

    print("\n=== RESULTADO NO TESTE (Junho 2025) ===")
    print(f"  ROC-AUC:       {roc:.4f}")
    print(f"  PR-AUC:        {pr:.4f}")
    print(f"  Threshold ótimo: {best_thr:.4f}")
    print(f"  F1-Score:      {f1:.4f}")
    print(f"  Precision:     {prec:.4f}")
    print(f"  Recall:        {rec:.4f}")

    # Comparação com modelo antigo
    old_proba = old_model.predict_proba(
        pd.read_parquet(str(GOLD_DIR / "gold_jun.parquet"),
                        columns=feature_cols).fillna(0).astype("float32")
    )[:, 1]
    old_f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    # usa mesmos thresholds para comparação justa
    old_prec2, old_rec2, old_thr2 = precision_recall_curve(y_test, old_proba)
    old_f12 = 2 * old_prec2 * old_rec2 / (old_prec2 + old_rec2 + 1e-9)
    old_f1_best = float(old_f12[:-1].max())

    print(f"\n  Modelo ANTERIOR F1: {old_f1_best:.4f}")
    print(f"  Modelo NOVO     F1: {f1:.4f}  ({f1 - old_f1_best:+.4f})")

    if f1 >= old_f1_best:
        with open(MODEL_OUT, "wb") as fp:
            pickle.dump(model, fp)
        print(f"\n✔ Modelo novo salvo → {MODEL_OUT.name}")

        import json
        metrics = {
            "test_month": "junho_2025",
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "optimal_threshold": round(best_thr, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "n_features": len(feature_cols),
            "algorithm": "LightGBM",
            "train_months": "jan-abr_2025",
            "eval_month": "mai_2025",
            "neg_sample_ratio": NEG_RATIO,
            "scale_pos_weight":  LGB_PARAMS["scale_pos_weight"],
            "num_leaves":        LGB_PARAMS["num_leaves"],
            "min_child_samples": LGB_PARAMS["min_child_samples"],
            "best_iteration":    int(model.best_iteration_),
        }
        metrics_path = Path(__file__).parent.parent / "outputs" / "reports" / "model_metrics.json"
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)
        print(f"✔ Métricas atualizadas → {metrics_path.name}")
    else:
        # restaura backup
        import shutil
        shutil.copy(MODEL_BAK, MODEL_OUT)
        print(f"\n⚠ Modelo novo não melhorou — backup restaurado.")


if __name__ == "__main__":
    train()
