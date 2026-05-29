"""Sprint 3 — Antecedência multi-horizonte (BONUS-1) + calibração (P-2).

1. Avalia o mesmo modelo LightGBM contra targets de 60/120/240min para
   demonstrar a janela de antecedência operacional prometida no PRD.
2. Calibration plot (Brier score) + curva precision/recall vs threshold
   para justificar matematicamente o threshold ótimo.

Salva: outputs/gold/horizon_calibration.json + figuras em outputs/figures/.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))
from models import load_model, _load_split, TEST_MONTHS, TARGET_COL
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"

HORIZONS = [60, 120, 240]  # minutos


def main():
    print("== Carregando modelo e dados ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", []))
    print(f"Modelo: {len(lgbm_features)} features (target original: {TARGET_COL})")

    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)

    # Teste (Jun) — features + 3 targets de horizonte
    test_files = [f for f in gold_files if any(m in f.stem for m in TEST_MONTHS)]
    target_cols = [f"is_dont_go_next_{h}m" for h in HORIZONS]
    missing_targets = [c for c in target_cols if c not in schema.columns]
    if missing_targets:
        print(f"⚠ Targets ausentes no gold: {missing_targets}")
        # Sem fallback prático — abortamos
        return None

    print("\n== Carregando teste (Jun) com todos os horizontes ==")
    needed = feature_cols + target_cols
    df_test = pl.read_parquet([str(f) for f in test_files], columns=needed).to_pandas()
    print(f"  {len(df_test):,} linhas")

    # Subset features para o modelo treinado
    for c in lgbm_features:
        if c not in df_test.columns:
            df_test[c] = 0.0
    X_test = df_test[lgbm_features].fillna(0).astype("float32")
    print("Computando probabilidades...")
    y_prob = lgbm.predict_proba(X_test)[:, 1]
    print("✓ probabilidades calculadas")

    # ── 1. Métricas por horizonte ──────────────────────────────────────────
    print("\n== Métricas por horizonte (threshold F1-ótimo do val) ==")
    # Threshold ótimo do modelo original
    try:
        err = json.loads((GOLD_DIR / "error_cost_analysis.json").read_text())
        thr_f1 = err.get("best_f1_threshold_val", 0.93)
        thr_cost = err.get("best_cost_threshold_val", 0.51)
    except Exception:
        thr_f1, thr_cost = 0.93, 0.51

    rows = []
    for h, tcol in zip(HORIZONS, target_cols):
        y_true = df_test[tcol].astype(int).values
        n_pos = int(y_true.sum())
        if n_pos == 0:
            continue
        for label, thr in [("F1-ótimo", thr_f1), ("Custo-ótimo", thr_cost)]:
            y_pred = (y_prob >= thr).astype(int)
            rows.append({
                "horizonte_min": h,
                "criterio_threshold": label,
                "threshold": thr,
                "N_positivos": n_pos,
                "Taxa_positivos_pct": 100.0 * n_pos / len(y_true),
                "F1": f1_score(y_true, y_pred, zero_division=0),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_true, y_prob),
                "PR_AUC": average_precision_score(y_true, y_prob),
                "Brier": brier_score_loss(y_true, y_prob),
            })
    horizon_df = pd.DataFrame(rows)
    print(horizon_df.to_string(index=False))

    # ── 2. Figura: degradação F1 vs horizonte ──────────────────────────────
    print("\n== Gerando figuras ==")
    fig, ax = plt.subplots(figsize=(10, 6))
    for criterio, sub in horizon_df.groupby("criterio_threshold"):
        sub = sub.sort_values("horizonte_min")
        color = "#3B82F6" if criterio == "F1-ótimo" else "#EF4444"
        ax.plot(sub["horizonte_min"], sub["F1"], marker="o", linewidth=2,
                label=f"F1 ({criterio} thr={sub['threshold'].iloc[0]:.2f})",
                color=color)
        ax.plot(sub["horizonte_min"], sub["Recall"], marker="s", linewidth=2,
                linestyle="--", label=f"Recall ({criterio})", color=color, alpha=0.6)
    ax.set_xlabel("Horizonte de antecedência (minutos)")
    ax.set_ylabel("Métrica")
    ax.set_title("Antecedência Preditiva — Degradação por Horizonte (Teste Jun/2025)\n"
                 "Modelo treinado com target=60min, avaliado contra targets 60/120/240min")
    ax.set_xticks(HORIZONS)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_degradation.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'horizon_degradation.png'}")

    # ── 3. Calibration plot (Brier) — para target 60min ────────────────────
    y_true_60 = df_test["is_dont_go_next_60m"].astype(int).values
    prob_pred, prob_obs = calibration_curve(y_true_60, y_prob, n_bins=15, strategy="quantile")
    brier = brier_score_loss(y_true_60, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibração perfeita")
    ax1.plot(prob_pred, prob_obs, "o-", color="#10B981", linewidth=2, markersize=8,
             label=f"LightGBM (Brier={brier:.4f})")
    ax1.set_xlabel("Probabilidade prevista")
    ax1.set_ylabel("Frequência observada")
    ax1.set_title("Curva de Calibração — Target 60min (Jun/2025)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Histograma das probabilidades previstas (rugplot ajuda interpretação)
    ax2.hist(y_prob, bins=80, color="#3B82F6", alpha=0.7, edgecolor="white")
    ax2.set_yscale("log")
    ax2.axvline(thr_f1, color="black", linestyle="--", linewidth=1.5, label=f"F1-ótimo @ {thr_f1:.2f}")
    ax2.axvline(thr_cost, color="red", linestyle=":", linewidth=1.5, label=f"Custo-ótimo @ {thr_cost:.2f}")
    ax2.set_xlabel("Probabilidade prevista")
    ax2.set_ylabel("Contagem (escala log)")
    ax2.set_title("Distribuição das Probabilidades Previstas (Teste Jun/2025)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_plot.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'calibration_plot.png'}")

    # ── 4. Curva precision/recall vs threshold ─────────────────────────────
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thresholds_grid:
        yp = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true_60, yp, zero_division=0))
        recs.append(recall_score(y_true_60, yp, zero_division=0))
        f1s.append(f1_score(y_true_60, yp, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds_grid, precs, label="Precision", color="#3B82F6", linewidth=2)
    ax.plot(thresholds_grid, recs, label="Recall", color="#EF4444", linewidth=2)
    ax.plot(thresholds_grid, f1s, label="F1-Score", color="#10B981", linewidth=2.5)
    ax.axvline(thr_f1, color="black", linestyle="--", alpha=0.6, label=f"F1-ótimo @ {thr_f1:.2f}")
    ax.axvline(thr_cost, color="purple", linestyle=":", alpha=0.6, label=f"Custo-ótimo @ {thr_cost:.2f}")
    ax.set_xlabel("Threshold de decisão")
    ax.set_ylabel("Métrica")
    ax.set_title("Curvas Precision / Recall / F1 vs Threshold (Teste Jun/2025, target 60min)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_curves.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'threshold_curves.png'}")

    # ── 5. Persistência ────────────────────────────────────────────────────
    payload = {
        "horizons_min": HORIZONS,
        "thresholds_used": {"f1_optimo": thr_f1, "custo_otimo": thr_cost},
        "metrics_by_horizon": horizon_df.to_dict(orient="records"),
        "calibration": {
            "brier_score": brier,
            "prob_predicted_bins": prob_pred.tolist(),
            "prob_observed_bins": prob_obs.tolist(),
        },
    }
    out = GOLD_DIR / "horizon_calibration.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Salvo em {out}")
    print(f"\nBrier Score = {brier:.5f}  (perfeito=0, baseline majoritário={y_true_60.mean()*(1-y_true_60.mean()):.5f})")
    print("══════════════ SPRINT 3 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
