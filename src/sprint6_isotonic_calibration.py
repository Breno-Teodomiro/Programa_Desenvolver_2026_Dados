"""Sprint 6 — Calibração isotônica pós-treino do LightGBM.

Motivação: Sprint 3 mostrou Brier=0.020 com sobre-estimativa em
probabilidades intermediárias (efeito do scale_pos_weight=40). Calibração
isotônica pós-treino preserva o ranking (logo F1/Recall/AUC ficam
preservados ou similares) mas ajusta os valores absolutos para que
P(y=1|score=p) = p de fato.

Aplicação: precificação de manutenção, dimensionamento de capacidade
operacional, comunicação executiva ("probabilidade de 0.7 significa 70%").

Salva: outputs/gold/isotonic_calibration.json + modelo + figuras.
"""

from pathlib import Path
import json
import sys
import pickle
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score, f1_score
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))
from models import load_model, _load_split, VAL_MONTHS, TEST_MONTHS, TARGET_COL
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"


def main():
    print("== Carregando modelo principal ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", []))
    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)

    # Val para AJUSTAR a calibração (não vista pelo modelo no treino — usada para early stop)
    # Idealmente seria um holdout próprio. Aqui usamos val como aproximação prática.
    print("\n== Carregando validação (Mai) ==")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS)
    for c in lgbm_features:
        if c not in X_val.columns:
            X_val[c] = 0.0
    X_val_lgbm = X_val[lgbm_features].fillna(0)
    p_val = lgbm.predict_proba(X_val_lgbm)[:, 1]
    print(f"  {len(X_val):,} linhas | pos={y_val.sum():,}")

    # ── Ajustar isotonic regression ────────────────────────────────────────
    print("\n== Ajustando IsotonicRegression no val ==")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)

    # ── Avaliar no teste ──────────────────────────────────────────────────
    print("\n== Carregando teste (Jun) ==")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    for c in lgbm_features:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test_lgbm = X_test[lgbm_features].fillna(0)
    p_raw = lgbm.predict_proba(X_test_lgbm)[:, 1]
    p_iso = iso.predict(p_raw)
    print(f"  {len(X_test):,} linhas | pos={y_test.sum():,}")

    # ── Métricas comparativas ─────────────────────────────────────────────
    metrics = {
        "raw": {
            "Brier": brier_score_loss(y_test, p_raw),
            "LogLoss": log_loss(y_test, np.clip(p_raw, 1e-15, 1-1e-15)),
            "ROC_AUC": roc_auc_score(y_test, p_raw),
            "PR_AUC": average_precision_score(y_test, p_raw),
            "prob_mean": float(p_raw.mean()),
            "prob_at_0_5": float((p_raw >= 0.5).mean()),
        },
        "isotonic": {
            "Brier": brier_score_loss(y_test, p_iso),
            "LogLoss": log_loss(y_test, np.clip(p_iso, 1e-15, 1-1e-15)),
            "ROC_AUC": roc_auc_score(y_test, p_iso),
            "PR_AUC": average_precision_score(y_test, p_iso),
            "prob_mean": float(p_iso.mean()),
            "prob_at_0_5": float((p_iso >= 0.5).mean()),
        },
    }
    base_rate = float(y_test.mean())
    metrics["baseline_brier"] = base_rate * (1 - base_rate)
    metrics["test_base_rate"] = base_rate

    print(f"\n══ Comparação Bruto vs Isotônico ══")
    print(f"  Métrica          {'Bruto':>10}  {'Isotônico':>12}  {'Δ':>10}")
    print(f"  Brier        {metrics['raw']['Brier']:>10.5f}  {metrics['isotonic']['Brier']:>12.5f}  "
          f"{(metrics['isotonic']['Brier']-metrics['raw']['Brier'])/metrics['raw']['Brier']*100:>+9.2f}%")
    print(f"  LogLoss      {metrics['raw']['LogLoss']:>10.5f}  {metrics['isotonic']['LogLoss']:>12.5f}")
    print(f"  ROC-AUC      {metrics['raw']['ROC_AUC']:>10.4f}  {metrics['isotonic']['ROC_AUC']:>12.4f}")
    print(f"  PR-AUC       {metrics['raw']['PR_AUC']:>10.4f}  {metrics['isotonic']['PR_AUC']:>12.4f}")
    print(f"\nBaseline Brier (predizer base_rate constante): {metrics['baseline_brier']:.5f}")

    # ── Figura: calibração comparada ──────────────────────────────────────
    print("\n== Gerando figuras ==")
    prob_pred_raw, prob_obs_raw = calibration_curve(y_test, p_raw, n_bins=15, strategy="quantile")
    prob_pred_iso, prob_obs_iso = calibration_curve(y_test, p_iso, n_bins=15, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibração perfeita")
    ax1.plot(prob_pred_raw, prob_obs_raw, "o-", color="#9CA3AF", linewidth=2, markersize=8,
             label=f"LightGBM bruto (Brier={metrics['raw']['Brier']:.4f})")
    ax1.plot(prob_pred_iso, prob_obs_iso, "o-", color="#10B981", linewidth=2, markersize=8,
             label=f"LightGBM + Isotonic (Brier={metrics['isotonic']['Brier']:.4f})")
    ax1.set_xlabel("Probabilidade prevista")
    ax1.set_ylabel("Frequência observada")
    ax1.set_title("Curva de Calibração — Antes vs Depois da Calibração Isotônica\n(Teste Jun/2025, target 60min)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Histograma das probabilidades — ANTES e DEPOIS
    ax2.hist(p_raw, bins=80, alpha=0.5, label="Bruto", color="#9CA3AF", edgecolor="white")
    ax2.hist(p_iso, bins=80, alpha=0.5, label="Isotônico", color="#10B981", edgecolor="white")
    ax2.set_yscale("log")
    ax2.set_xlabel("Probabilidade prevista")
    ax2.set_ylabel("Contagem (log)")
    ax2.set_title("Distribuição das Probabilidades — Antes vs Depois")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "isotonic_calibration.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'isotonic_calibration.png'}")

    # Mapeamento da função isotônica
    fig, ax = plt.subplots(figsize=(8, 7))
    grid = np.linspace(0.001, 0.999, 200)
    mapped = iso.predict(grid)
    ax.plot(grid, grid, "k--", alpha=0.5, label="y = x (identidade)")
    ax.plot(grid, mapped, color="#10B981", linewidth=2.5, label="Função isotônica ajustada")
    ax.set_xlabel("Probabilidade bruta (LightGBM)")
    ax.set_ylabel("Probabilidade calibrada (Isotonic)")
    ax.set_title("Função de Calibração — Mapeamento Isotônico\n(ajustado no val Mai/2025)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "isotonic_mapping.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'isotonic_mapping.png'}")

    # ── Persistência ──────────────────────────────────────────────────────
    with open(GOLD_DIR / "isotonic_calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)

    payload = {
        "metrics": metrics,
        "improvement_brier_pct": float((metrics["raw"]["Brier"] - metrics["isotonic"]["Brier"]) / metrics["raw"]["Brier"] * 100),
        "test_month": list(TEST_MONTHS),
        "val_month_for_fit": list(VAL_MONTHS),
    }
    with open(GOLD_DIR / "isotonic_calibration.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Calibrador salvo em {GOLD_DIR / 'isotonic_calibrator.pkl'}")
    print(f"✓ Métricas em {GOLD_DIR / 'isotonic_calibration.json'}")
    print(f"\n🎯 Melhoria de Brier: {payload['improvement_brier_pct']:+.2f}%")
    print("══════════════ SPRINT 6 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
