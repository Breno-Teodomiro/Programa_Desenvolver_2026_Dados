"""Sprint 1 — Comparação de modelos: Baseline naive, Baseline regra, LogReg L1, RandomForest, LightGBM.

Atende CM 4.2-4.3 (dois modelos distintos documentados + baseline).
Salva resultados em outputs/gold/comparison_sprint1.json e figuras em outputs/figures/.
"""

from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))

from models import (
    _load_split,
    TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS, TARGET_COL,
    load_model,
    evaluate_majority_baseline,
    evaluate_rule_baseline,
    find_best_rule_threshold,
    train_logistic_baseline,
    train_random_forest,
    evaluate_sklearn_model,
    evaluate_model,
    find_best_threshold,
    build_comparison_table,
)
from features import get_feature_columns

GOLD_DIR = Path(__file__).parent.parent / "outputs" / "gold"
FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    print(f"Gold files: {[f.name for f in gold_files]}")

    # Features
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)
    print(f"Features: {len(feature_cols)}")

    # ── Carregamento ────────────────────────────────────────────────────────
    # Treino: undersampled 5:1 para Logistic e RF caberem em RAM e treinarem rápido
    print("\n== Loading splits ==")
    print("Treino (undersampled 5:1):")
    X_tr_us, y_tr_us = _load_split(gold_files, feature_cols, TRAIN_MONTHS, neg_sample_ratio=5)
    print(f"  {len(X_tr_us):,} linhas, {y_tr_us.sum():,} positivos")

    print("Validação (Mai):")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS)
    print(f"  {len(X_val):,} linhas, {y_val.sum():,} positivos")

    print("Teste (Jun):")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    print(f"  {len(X_test):,} linhas, {y_test.sum():,} positivos")

    # Para baselines de regra: precisamos da coluna n_criticos_30m no val/test
    print("\nCarregando n_criticos_30m para baselines de regra...")
    val_files = [f for f in gold_files if any(m in f.stem for m in VAL_MONTHS)]
    test_files = [f for f in gold_files if any(m in f.stem for m in TEST_MONTHS)]
    val_rule = pl.read_parquet([str(f) for f in val_files], columns=["n_criticos_30m"]).to_pandas()
    test_rule = pl.read_parquet([str(f) for f in test_files], columns=["n_criticos_30m"]).to_pandas()

    results = []

    # ── 1. Baseline naive — majority class ─────────────────────────────────
    print("\n== Baseline 1: Majority class ==")
    m_naive = evaluate_majority_baseline(y_test)
    results.append(m_naive)

    # ── 2. Baseline regra de negócio ───────────────────────────────────────
    print("\n== Baseline 2: Rule (n_criticos_30m >= N) ==")
    best_n = find_best_rule_threshold(val_rule, y_val, "n_criticos_30m")
    m_rule = evaluate_rule_baseline(test_rule, y_test, "n_criticos_30m", threshold=best_n)
    results.append(m_rule)

    # ── 3. Logistic Regression L1 ──────────────────────────────────────────
    print("\n== Model A: Logistic Regression L1 ==")
    # scale_pos_weight=1.0 porque já fizemos undersampling 5:1
    logreg = train_logistic_baseline(X_tr_us, y_tr_us, scale_pos_weight=5.0, C=0.1)
    # Threshold ótimo no val
    y_prob_val = logreg.predict_proba(X_val.fillna(0))[:, 1]
    thr_log = float(np.linspace(0.05, 0.95, 91)[np.argmax([
        np.mean(((y_prob_val >= t).astype(int) == 1) & (y_val == 1)) /
        max(np.mean((y_prob_val >= t).astype(int) == 1), 1e-9) +
        np.mean(((y_prob_val >= t).astype(int) == 1) & (y_val == 1)) /
        max(y_val.mean(), 1e-9)
        for t in np.linspace(0.05, 0.95, 91)
    ])])
    # Fallback simples: usar find_best_threshold padrão
    from sklearn.metrics import f1_score
    thrs = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (y_prob_val >= t).astype(int), zero_division=0) for t in thrs]
    thr_log = float(thrs[int(np.argmax(f1s))])
    print(f"LogReg threshold ótimo (val): {thr_log:.3f} (F1={max(f1s):.4f})")
    m_log = evaluate_sklearn_model(logreg, X_test, y_test, threshold=thr_log, name="logistic_l1")
    m_log["threshold"] = thr_log
    results.append(m_log)

    # ── 4. Random Forest ───────────────────────────────────────────────────
    print("\n== Model B: Random Forest ==")
    rf = train_random_forest(X_tr_us, y_tr_us, n_estimators=200, max_depth=20, scale_pos_weight=5.0)
    y_prob_val_rf = rf.predict_proba(X_val.fillna(0))[:, 1]
    f1s_rf = [f1_score(y_val, (y_prob_val_rf >= t).astype(int), zero_division=0) for t in thrs]
    thr_rf = float(thrs[int(np.argmax(f1s_rf))])
    print(f"RF threshold ótimo (val): {thr_rf:.3f} (F1={max(f1s_rf):.4f})")
    m_rf = evaluate_sklearn_model(rf, X_test, y_test, threshold=thr_rf, name="random_forest")
    m_rf["threshold"] = thr_rf
    results.append(m_rf)

    # ── 5. LightGBM (existente) ────────────────────────────────────────────
    print("\n== Model C: LightGBM (existente) ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", feature_cols))
    print(f"LightGBM espera {len(lgbm_features)} features (modelo treinado anteriormente).")
    missing = [c for c in lgbm_features if c not in X_val.columns]
    if missing:
        print(f"  ⚠️  Features ausentes no gold atual: {missing[:5]}{'...' if len(missing)>5 else ''}")
        # preenche com 0 para manter dimensionalidade
        for c in missing:
            X_val[c] = 0.0
            X_test[c] = 0.0
    X_val_lgbm = X_val[lgbm_features]
    X_test_lgbm = X_test[lgbm_features]
    thr_lgbm = find_best_threshold(lgbm, X_val_lgbm, y_val, metric="f1")
    m_lgbm = evaluate_model(lgbm, X_test_lgbm, y_test, threshold=thr_lgbm)
    m_lgbm["model"] = "lightgbm"
    results.append(m_lgbm)

    # ── Tabela comparativa ─────────────────────────────────────────────────
    print("\n\n══════════════ TABELA COMPARATIVA ══════════════")
    comp = build_comparison_table(results)
    print(comp.to_string(index=False))

    # ── Figuras ────────────────────────────────────────────────────────────
    print("\n== Generating figures ==")

    # 1. Barchart F1 por modelo
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#9CA3AF", "#F59E0B", "#3B82F6", "#8B5CF6", "#10B981"]
    ax.bar(comp["Modelo"], comp["F1"], color=colors[:len(comp)])
    ax.set_ylabel("F1-Score")
    ax.set_title("Comparação de Modelos — F1-Score no Conjunto de Teste (Jun/2025)")
    ax.set_ylim(0, max(comp["F1"].max() * 1.15, 0.05))
    for i, v in enumerate(comp["F1"]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_f1_bar.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'comparison_f1_bar.png'}")

    # 2. ROC overlay para modelos com prob
    fig, ax = plt.subplots(figsize=(8, 7))
    from sklearn.metrics import roc_auc_score as _rauc
    for model_obj, name, color, X_for in [
        (logreg, "Logistic L1", "#3B82F6", X_test),
        (rf, "Random Forest", "#8B5CF6", X_test),
        (lgbm, "LightGBM", "#10B981", X_test_lgbm),
    ]:
        if hasattr(model_obj, "predict_proba"):
            yp = model_obj.predict_proba(X_for.fillna(0))[:, 1]
            fpr, tpr, _ = roc_curve(y_test, yp)
            ax.plot(fpr, tpr, label=f"{name} (AUC={_rauc(y_test, yp):.4f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC — Modelos Treinados (Teste Jun/2025)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_roc_curves.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'comparison_roc_curves.png'}")

    # 3. PR overlay
    fig, ax = plt.subplots(figsize=(8, 7))
    from sklearn.metrics import average_precision_score as _ap
    for model_obj, name, color, X_for in [
        (logreg, "Logistic L1", "#3B82F6", X_test),
        (rf, "Random Forest", "#8B5CF6", X_test),
        (lgbm, "LightGBM", "#10B981", X_test_lgbm),
    ]:
        yp = model_obj.predict_proba(X_for.fillna(0))[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, yp)
        ap = _ap(y_test, yp)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=color, linewidth=2)
    base_rate = float(y_test.mean())
    ax.axhline(base_rate, color="k", linestyle="--", alpha=0.4, label=f"Baseline ({base_rate:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — Modelos Treinados (Teste Jun/2025)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_pr_curves.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'comparison_pr_curves.png'}")

    # ── Persistência ───────────────────────────────────────────────────────
    out_json = GOLD_DIR / "comparison_sprint1.json"
    payload = {
        "results": results,
        "comparison_table": comp.to_dict(orient="records"),
        "best_rule_threshold": int(best_n),
        "n_features": len(feature_cols),
        "train_months": sorted(TRAIN_MONTHS),
        "val_month": sorted(VAL_MONTHS),
        "test_month": sorted(TEST_MONTHS),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Resultados salvos em {out_json}")

    # Salvar modelos alternativos
    import pickle
    with open(GOLD_DIR / "logistic_l1.pkl", "wb") as f:
        pickle.dump(logreg, f)
    with open(GOLD_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    print(f"✓ Modelos alternativos salvos em {GOLD_DIR}")

    print("\n══════════════ SPRINT 1 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
