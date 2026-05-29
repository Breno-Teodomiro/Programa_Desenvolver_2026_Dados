"""Sprint 5 — Modelo LightGBM especializado para escavadeira LeTourneau L 1850.

Motivação: o modelo geral (Sprint 1-3) tem Recall=0.14 e PR-AUC=0.01 em
escavadeiras — falha quase completa por dominância de padrões 793-D no
fingerprint. Aqui treinamos um modelo dedicado SOMENTE com dados de
escavadeira para validar se um especialista pode capturar o padrão
operacional específico desses equipamentos.

Comparação esperada:
- Modelo geral em escavadeira: F1=0.015, Recall=0.14
- Modelo escavadeira: F1=?, Recall=? (esperamos melhoria significativa)

Salva: outputs/gold/escavadeira_model.json + modelo + figuras.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
TARGET = "is_dont_go_next_60m"
FROTA_FILTRO = "LeTourneau L 1850"

TRAIN_MONTHS = {"jan", "feb", "mar", "abr"}
VAL_MONTHS = {"may"}
TEST_MONTHS = {"jun"}


def load_escavadeira_split(gold_files, feature_cols, month_set, neg_sample_ratio=None):
    files = [f for f in gold_files if any(m in f.stem for m in month_set)]
    cols = feature_cols + [TARGET, "Tag_Frota"]
    df = pl.read_parquet([str(f) for f in files], columns=cols)
    df = df.filter(pl.col("Tag_Frota") == FROTA_FILTRO)
    if neg_sample_ratio is not None:
        _t = pl.col(TARGET).cast(pl.Int8)
        pos = df.filter(_t == 1)
        neg_all = df.filter(_t == 0)
        n_neg_sample = min(len(pos) * neg_sample_ratio, len(neg_all))
        if n_neg_sample > 0:
            neg = neg_all.sample(n=n_neg_sample, seed=42)
            df = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=42)
            print(f"  undersampled {neg_sample_ratio}:1 → pos={len(pos):,}, neg={len(neg):,}")
            del pos, neg, neg_all
    X = df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_pandas()
    y = df[TARGET].cast(pl.Int8).to_pandas()
    return X, y


def main():
    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)
    print(f"Features: {len(feature_cols)}")

    print(f"\n== Treino (Jan-Abr) — filtrado {FROTA_FILTRO} ==")
    X_tr, y_tr = load_escavadeira_split(gold_files, feature_cols, TRAIN_MONTHS, neg_sample_ratio=10)
    print(f"  {len(X_tr):,} linhas | pos={y_tr.sum():,} ({y_tr.mean():.3%})")

    print(f"\n== Validação (Mai) ==")
    X_val, y_val = load_escavadeira_split(gold_files, feature_cols, VAL_MONTHS)
    print(f"  {len(X_val):,} linhas | pos={y_val.sum():,} ({y_val.mean():.4%})")

    print(f"\n== Teste (Jun) ==")
    X_te, y_te = load_escavadeira_split(gold_files, feature_cols, TEST_MONTHS)
    print(f"  {len(X_te):,} linhas | pos={y_te.sum():,} ({y_te.mean():.5%})")

    # ── Treino LightGBM dedicado ───────────────────────────────────────────
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = n_neg / max(n_pos, 1)
    print(f"\nscale_pos_weight = {spw:.2f}")

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "scale_pos_weight": spw,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("\n== Treinando LightGBM escavadeira ==")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    del X_tr, y_tr

    # Threshold ótimo no val
    yp_val = model.predict_proba(X_val)[:, 1]
    thrs = np.linspace(0.05, 0.99, 95)
    f1s_val = [f1_score(y_val, (yp_val >= t).astype(int), zero_division=0) for t in thrs]
    best_thr = float(thrs[int(np.argmax(f1s_val))])
    print(f"\nThreshold ótimo (val Mai): {best_thr:.3f} (F1={max(f1s_val):.4f})")

    # ── Avaliação no teste ────────────────────────────────────────────────
    yp_te = model.predict_proba(X_te)[:, 1]
    yp_pred = (yp_te >= best_thr).astype(int)
    metrics = {
        "threshold": best_thr,
        "n_test": int(len(y_te)),
        "n_positives": int(y_te.sum()),
        "F1": f1_score(y_te, yp_pred, zero_division=0),
        "Precision": precision_score(y_te, yp_pred, zero_division=0),
        "Recall": recall_score(y_te, yp_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_te, yp_te) if y_te.sum() > 0 else None,
        "PR_AUC": average_precision_score(y_te, yp_te) if y_te.sum() > 0 else None,
        "confusion_matrix": confusion_matrix(y_te, yp_pred).tolist(),
        "best_iteration": int(model.best_iteration_),
    }

    print(f"\n══ Métricas no Teste (Jun) — {FROTA_FILTRO} ══")
    print(f"  F1        : {metrics['F1']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  ROC-AUC   : {metrics['ROC_AUC']:.4f}" if metrics['ROC_AUC'] else "  ROC-AUC   : —")
    print(f"  PR-AUC    : {metrics['PR_AUC']:.4f}" if metrics['PR_AUC'] else "  PR-AUC    : —")

    # Comparação com modelo geral (Sprint 2 fleet segmentation)
    try:
        fleet = json.loads((GOLD_DIR / "fleet_segmentation.json").read_text())
        geral_esc = next(f for f in fleet["by_fleet"] if f["Frota"] == FROTA_FILTRO)
    except (FileNotFoundError, StopIteration):
        geral_esc = {"F1": 0.015, "Recall": 0.14, "PR_AUC": 0.01, "Precision": 0.008}

    print(f"\n══ Comparação: Modelo Geral vs Modelo Escavadeira ══")
    comp_rows = [
        ["Modelo", "F1", "Precision", "Recall", "PR-AUC"],
        ["Geral (Sprints 1-3)",
         f"{geral_esc.get('F1', 0):.4f}",
         f"{geral_esc.get('Precision', 0):.4f}",
         f"{geral_esc.get('Recall', 0):.4f}",
         f"{geral_esc.get('PR_AUC', 0):.4f}"],
        ["Especializado (Sprint 5)",
         f"{metrics['F1']:.4f}",
         f"{metrics['Precision']:.4f}",
         f"{metrics['Recall']:.4f}",
         f"{metrics['PR_AUC']:.4f}"],
    ]
    for row in comp_rows:
        print(f"  {row[0]:<28} {row[1]:>8} {row[2]:>10} {row[3]:>8} {row[4]:>9}")

    # ── Figuras ───────────────────────────────────────────────────────────
    print("\n== Gerando figuras ==")
    if y_te.sum() > 0:
        # ROC + PR side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        fpr, tpr, _ = roc_curve(y_te, yp_te)
        ax1.plot(fpr, tpr, color="#3B82F6", linewidth=2, label=f"Especializado (AUC={metrics['ROC_AUC']:.4f})")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.set_title(f"ROC — Modelo Especializado para {FROTA_FILTRO}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        prec, rec, _ = precision_recall_curve(y_te, yp_te)
        ax2.plot(rec, prec, color="#10B981", linewidth=2, label=f"AP={metrics['PR_AUC']:.4f}")
        base = y_te.mean()
        ax2.axhline(base, color="k", linestyle="--", alpha=0.4, label=f"Baseline ({base:.5f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall — Modelo Especializado Escavadeira")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "escavadeira_roc_pr.png", dpi=150)
        plt.close()
        print(f"  ✓ {FIGURES_DIR / 'escavadeira_roc_pr.png'}")

    # Bar comparison
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics_list = ["F1", "Precision", "Recall", "PR-AUC"]
    geral_vals = [geral_esc.get("F1", 0), geral_esc.get("Precision", 0),
                  geral_esc.get("Recall", 0), geral_esc.get("PR_AUC", 0)]
    esp_vals = [metrics["F1"], metrics["Precision"], metrics["Recall"], metrics["PR_AUC"] or 0]
    x = np.arange(len(metrics_list))
    w = 0.4
    ax.bar(x - w/2, geral_vals, w, label="Modelo geral (Sprints 1-3)", color="#9CA3AF")
    ax.bar(x + w/2, esp_vals, w, label="Modelo especializado escavadeira", color="#10B981")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.set_ylabel("Métrica")
    ax.set_title(f"Modelo Geral vs Modelo Especializado — Teste em {FROTA_FILTRO} (Jun/2025)")
    ax.legend()
    for i, (g, e) in enumerate(zip(geral_vals, esp_vals)):
        ax.text(i - w/2, g + 0.01, f"{g:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, e + 0.01, f"{e:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(max(geral_vals), max(esp_vals)) * 1.2)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "escavadeira_comparison.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'escavadeira_comparison.png'}")

    # ── Persistência ───────────────────────────────────────────────────────
    with open(GOLD_DIR / "lgbm_escavadeira.pkl", "wb") as f:
        pickle.dump(model, f)
    payload = {
        "frota": FROTA_FILTRO,
        "metrics_test": metrics,
        "comparison": {
            "geral": {k: geral_esc.get(k) for k in ["F1", "Precision", "Recall", "PR_AUC"]},
            "especializado": {k: metrics[k] for k in ["F1", "Precision", "Recall", "PR_AUC"]},
        },
        "params": params,
        "best_iteration": int(model.best_iteration_),
        "n_features": len(feature_cols),
    }
    with open(GOLD_DIR / "escavadeira_model.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Modelo salvo em {GOLD_DIR / 'lgbm_escavadeira.pkl'}")
    print(f"✓ Resultados em {GOLD_DIR / 'escavadeira_model.json'}")
    print("══════════════ SPRINT 5 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
