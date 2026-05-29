"""⚠️ DEPRECATED — NÃO USAR EM PRODUÇÃO.

Experimento que adiciona features de perfil por equipamento (TAG-level).
Concluiu-se empiricamente que essas features causam *distribution shift* entre
meses e PIORAM recall/F1 no teste (overfitting de identidade do equipamento).
Mantido apenas como registro histórico do experimento negativo.

➡️  Para retreinar o modelo final reproduzível, use: src/retrain_optimized.py

----------------------------------------------------------------------------
Retreino v2 — adiciona features de perfil por equipamento (TAG-level).

Novas features (sem re-rodar pipeline gold):
  tag_dg_rate         — taxa histórica de DG por TAG (jan-mai)
  tag_alarm_intensity — média de n_alarmes_240m por TAG
  tag_critical_rate   — fração de alarmes críticos por TAG
  tag_dg_per_krow     — DGs por 1000 linhas por TAG (normalizado por atividade)
  tag_log_dg_total    — log1p do total de DG rows por TAG

Estratégia: computa perfil por TAG APENAS nos meses de treino (sem leakage),
depois merge estático em treino e teste.
"""

import json
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
    precision_recall_curve,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))

GOLD_DIR = Path(__file__).parent.parent / "outputs" / "gold"
MODEL_OUT = GOLD_DIR / "lgbm_dontgo.pkl"
MODEL_BAK = GOLD_DIR / "lgbm_dontgo_v1_backup.pkl"
TARGET_COL = "is_dont_go_next_60m"

TRAIN_MONTHS = {"jan", "feb", "mar", "abr", "may"}
TEST_MONTHS  = {"jun"}
NEG_RATIO    = 5
RANDOM_STATE = 42

# Features de perfil adicionadas por este script
TAG_FEATURE_COLS = [
    "tag_dg_rate",
    "tag_alarm_intensity",
    "tag_critical_rate",
    "tag_dg_per_krow",
    "tag_log_dg_total",
]

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            ["binary_logloss", "auc"],
    "num_leaves":        63,
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      1000,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_samples": 100,
    "reg_lambda":        1.0,
    "scale_pos_weight":  40.0,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
}


def get_feature_cols() -> list[str]:
    with open(MODEL_OUT, "rb") as f:
        m = pickle.load(f)
    return m.feature_name_


def compute_tag_profiles(train_months: set[str]) -> pd.DataFrame:
    """Perfil histórico por equipamento — calculado APENAS nos meses de treino."""
    files = [str(GOLD_DIR / f"gold_{m}.parquet") for m in sorted(train_months)]
    files_str = "', '".join(files)

    df = duckdb.execute(f"""
        SELECT
            TAG,
            AVG(CAST(is_dont_go_next_60m AS DOUBLE)) AS tag_dg_rate,
            AVG(n_alarmes_240m)                       AS tag_alarm_intensity,
            AVG(CASE WHEN n_alarmes_240m > 0
                     THEN CAST(n_criticos_240m AS DOUBLE) / n_alarmes_240m
                     ELSE 0 END)                      AS tag_critical_rate,
            SUM(CAST(is_dont_go_next_60m AS DOUBLE))  AS tag_dg_total,
            COUNT(*)                                   AS tag_n_rows
        FROM read_parquet(['{files_str}'])
        GROUP BY TAG
    """).df()

    df["tag_dg_per_krow"]    = (df["tag_dg_total"] / (df["tag_n_rows"] / 1000.0)).fillna(0)
    df["tag_log_dg_total"]   = np.log1p(df["tag_dg_total"])

    print("  Perfis por TAG:")
    top = df.sort_values("tag_dg_rate", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"    {r['TAG']:15s}  dg_rate={r['tag_dg_rate']:.4f}  dg/krow={r['tag_dg_per_krow']:.2f}")

    return df[["TAG"] + TAG_FEATURE_COLS].copy()


def _load_months(
    month_set: set[str],
    base_feature_cols: list[str],
    tag_profiles: pd.DataFrame,
    neg_ratio: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    files = [GOLD_DIR / f"gold_{m}.parquet" for m in month_set]
    files = [f for f in files if f.exists()]
    needed = ["TAG"] + base_feature_cols + [TARGET_COL]

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

    # Merge com perfil TAG (em pandas para simplicidade)
    pdf = df.to_pandas()
    del df
    pdf = pdf.merge(tag_profiles, on="TAG", how="left")

    # TAGs no teste que não apareceram no treino recebem valores medianos
    for col in TAG_FEATURE_COLS:
        pdf[col] = pdf[col].fillna(tag_profiles[col].median())

    y = pdf[TARGET_COL].astype("int8")
    all_feature_cols = base_feature_cols + TAG_FEATURE_COLS
    X = pdf[all_feature_cols].astype("float32")
    del pdf
    return X, y


def train():
    print("\n=== Retreino v2 — Features de Perfil por TAG ===\n")

    base_feature_cols = get_feature_cols()
    print(f"Features base: {len(base_feature_cols)}")
    print(f"Features novas: {TAG_FEATURE_COLS}")
    total_features = len(base_feature_cols) + len(TAG_FEATURE_COLS)
    print(f"Total features: {total_features}")

    with open(MODEL_OUT, "rb") as f:
        old_model = pickle.load(f)
    with open(MODEL_BAK, "wb") as f:
        pickle.dump(old_model, f)
    print(f"\nBackup salvo → {MODEL_BAK.name}")

    # Computa perfil por TAG (apenas meses de treino — sem leakage)
    print(f"\nComputando perfis por equipamento ({', '.join(sorted(TRAIN_MONTHS))})...")
    tag_profiles = compute_tag_profiles(TRAIN_MONTHS)

    # Treino
    print(f"\nCarregando treino ({', '.join(sorted(TRAIN_MONTHS))})...")
    X_train, y_train = _load_months(TRAIN_MONTHS, base_feature_cols, tag_profiles, neg_ratio=NEG_RATIO)
    print(f"  RAM treino: ~{X_train.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"  Distribuição: {y_train.sum():,} pos / {(y_train==0).sum():,} neg")

    # Eval set interno (15%)
    n_eval = max(int(len(X_train) * 0.15), 1000)
    eval_idx   = np.random.default_rng(RANDOM_STATE).choice(len(X_train), n_eval, replace=False)
    train_idx  = np.setdiff1d(np.arange(len(X_train)), eval_idx)
    X_eval, y_eval = X_train.iloc[eval_idx], y_train.iloc[eval_idx]
    X_fit,  y_fit  = X_train.iloc[train_idx], y_train.iloc[train_idx]
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
    X_test, y_test = _load_months(TEST_MONTHS, base_feature_cols, tag_profiles, neg_ratio=None)
    proba = model.predict_proba(X_test)[:, 1]
    del X_test

    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, proba)
    f1_arr  = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_idx = np.argmax(f1_arr[:-1])
    best_thr = float(thresholds[best_idx])
    f1   = float(f1_arr[best_idx])
    prec = float(prec_arr[best_idx])
    rec  = float(rec_arr[best_idx])

    print("\n=== RESULTADO NO TESTE (Junho 2025) ===")
    print(f"  ROC-AUC:         {roc:.4f}")
    print(f"  PR-AUC:          {pr:.4f}")
    print(f"  Threshold ótimo: {best_thr:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  Precision:       {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")

    # Comparação com modelo anterior
    old_pdf = pd.read_parquet(str(GOLD_DIR / "gold_jun.parquet"), columns=["TAG"] + base_feature_cols)
    old_pdf = old_pdf.merge(tag_profiles, on="TAG", how="left")
    for col in TAG_FEATURE_COLS:
        old_pdf[col] = old_pdf[col].fillna(tag_profiles[col].median())

    # O modelo antigo não tem as novas features — usar só base
    old_proba = old_model.predict_proba(
        old_pdf[base_feature_cols].fillna(0).astype("float32")
    )[:, 1]
    del old_pdf

    old_prec2, old_rec2, _ = precision_recall_curve(y_test, old_proba)
    old_f12 = 2 * old_prec2 * old_rec2 / (old_prec2 + old_rec2 + 1e-9)
    old_f1_best = float(old_f12[:-1].max())

    print(f"\n  Modelo ANTERIOR F1: {old_f1_best:.4f}")
    print(f"  Modelo NOVO     F1: {f1:.4f}  ({f1 - old_f1_best:+.4f})")

    # Top features novas no modelo
    if hasattr(model, "feature_importances_"):
        feat_names = base_feature_cols + TAG_FEATURE_COLS
        importance = dict(zip(feat_names, model.feature_importances_))
        print("\n  Importância das features de TAG:")
        for col in TAG_FEATURE_COLS:
            print(f"    {col}: {importance.get(col, 0):.1f}")

    if f1 >= old_f1_best:
        with open(MODEL_OUT, "wb") as fp:
            pickle.dump(model, fp)
        print(f"\n✔ Modelo novo salvo → {MODEL_OUT.name}")

        metrics = {
            "test_month": "junho_2025",
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "optimal_threshold": round(best_thr, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "n_features": total_features,
            "algorithm": "LightGBM",
            "train_months": "jan-mai_2025",
            "neg_sample_ratio": NEG_RATIO,
            "scale_pos_weight": LGB_PARAMS["scale_pos_weight"],
            "best_iteration": int(model.best_iteration_),
            "tag_features_added": TAG_FEATURE_COLS,
        }
        metrics_path = Path(__file__).parent.parent / "outputs" / "reports" / "model_metrics.json"
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)
        print(f"✔ Métricas atualizadas → {metrics_path.name}")
    else:
        import shutil
        shutil.copy(MODEL_BAK, MODEL_OUT)
        print(f"\n⚠ Modelo não melhorou — backup restaurado.")


if __name__ == "__main__":
    raise SystemExit(
        "⚠️ retrain_v2_features.py está DEPRECATED — as features de perfil por TAG "
        "causam distribution shift e pioram o F1.\n"
        "   Use 'uv run python3 src/retrain_optimized.py' para o modelo final.\n"
        "   (Para rodar este experimento histórico mesmo assim, chame train() "
        "explicitamente em um script/REPL.)"
    )
