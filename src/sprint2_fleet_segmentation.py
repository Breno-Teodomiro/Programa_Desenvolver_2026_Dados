"""Sprint 2 — Análise segmentada por frota e equipamento.

Substituiu a análise de operador (dados ausentes no parquet). Atende ao
diferencial "padrões operacionais" do PRD respondendo:
1. O modelo tem desempenho uniforme entre frotas (793-D 2S/3S/4S/5S, escavadeiras)?
2. Existem equipamentos onde o modelo falha sistematicamente?
3. Há features de fingerprint específicas por frota?

Salva: outputs/gold/fleet_segmentation.json + figuras em outputs/figures/.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from models import load_model, _load_split, TEST_MONTHS, TARGET_COL
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"


def main():
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", []))

    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)

    print("== Carregando teste (Jun) ==")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    for c in lgbm_features:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test_lgbm = X_test[lgbm_features].fillna(0)
    y_prob = lgbm.predict_proba(X_test_lgbm)[:, 1]

    # Threshold custo-ótimo (calculado no script de erros)
    try:
        with open(GOLD_DIR / "error_cost_analysis.json") as f:
            err = json.load(f)
        thr = err.get("best_cost_threshold_val", 0.5)
    except FileNotFoundError:
        thr = 0.5
    print(f"Usando threshold custo-ótimo = {thr:.3f}")
    y_pred = (y_prob >= thr).astype(int)

    # Metadados (Tag_Frota, TAG) para segmentação
    test_files = [f for f in gold_files if any(m in f.stem for m in TEST_MONTHS)]
    meta = pl.read_parquet(
        [str(f) for f in test_files],
        columns=["TAG", "Tag_Frota", "Tipo"],
    ).to_pandas()
    meta["y_true"] = y_test.values
    meta["y_pred"] = y_pred
    meta["y_prob"] = y_prob

    # ── 1. Performance por frota ──────────────────────────────────────────
    print("\n== Performance segmentada por frota ==")
    rows_by_fleet = []
    for fleet, g in meta.groupby("Tag_Frota"):
        n = len(g)
        n_pos = int(g["y_true"].sum())
        if n_pos == 0:
            continue
        f1 = f1_score(g["y_true"], g["y_pred"], zero_division=0)
        prec = precision_score(g["y_true"], g["y_pred"], zero_division=0)
        rec = recall_score(g["y_true"], g["y_pred"], zero_division=0)
        try:
            roc = roc_auc_score(g["y_true"], g["y_prob"])
        except ValueError:
            roc = None
        pr = average_precision_score(g["y_true"], g["y_prob"]) if n_pos > 0 else None
        rate_pct = 100.0 * n_pos / n
        rows_by_fleet.append({
            "Frota": fleet,
            "N_eventos": n,
            "N_positivos": n_pos,
            "Taxa_DG_pct": rate_pct,
            "F1": f1, "Precision": prec, "Recall": rec,
            "ROC_AUC": roc, "PR_AUC": pr,
        })
    by_fleet = pd.DataFrame(rows_by_fleet).sort_values("N_positivos", ascending=False)
    print(by_fleet.to_string(index=False))

    # ── 2. Top 10 equipamentos com mais Don't Go e desempenho do modelo ────
    print("\n== Top 10 equipamentos (Jun) por volume de DG real ==")
    rows_by_tag = []
    for tag, g in meta.groupby("TAG"):
        n_pos = int(g["y_true"].sum())
        if n_pos == 0:
            continue
        n = len(g)
        f1 = f1_score(g["y_true"], g["y_pred"], zero_division=0)
        prec = precision_score(g["y_true"], g["y_pred"], zero_division=0)
        rec = recall_score(g["y_true"], g["y_pred"], zero_division=0)
        try:
            roc = roc_auc_score(g["y_true"], g["y_prob"])
        except ValueError:
            roc = None
        rows_by_tag.append({
            "TAG": tag,
            "Frota": g["Tag_Frota"].iloc[0],
            "N_eventos": n,
            "N_positivos": n_pos,
            "Taxa_DG_pct": 100.0 * n_pos / n,
            "F1": f1, "Precision": prec, "Recall": rec, "ROC_AUC": roc,
        })
    by_tag = pd.DataFrame(rows_by_tag).sort_values("N_positivos", ascending=False)
    top10 = by_tag.head(10)
    print(top10.to_string(index=False))

    # ── 3. Equipamentos onde o modelo MAIS falha (entre os com positivos) ──
    worst_recall = by_tag[by_tag["N_positivos"] >= 10].nsmallest(5, "Recall")
    print("\n== 5 equipamentos com pior recall (modelo falha mais) ==")
    print(worst_recall.to_string(index=False))

    # ── 4. Figuras ─────────────────────────────────────────────────────────
    print("\n== Gerando figuras ==")

    # F1 por frota
    fig, ax = plt.subplots(figsize=(11, 5))
    bf = by_fleet.copy()
    colors = ["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#06B6D4"][:len(bf)]
    ax.bar(bf["Frota"], bf["F1"], color=colors)
    ax.set_ylabel("F1-Score")
    ax.set_title(f"Desempenho do Modelo por Frota — Teste Jun/2025 (threshold custo-ótimo = {thr:.3f})")
    for i, v in enumerate(bf["F1"]):
        ax.text(i, v + 0.01, f"{v:.3f}\n(n={bf['N_positivos'].iloc[i]:,})",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(bf["F1"].max() * 1.25, 0.05))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "f1_by_fleet.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'f1_by_fleet.png'}")

    # F1 vs volume de DG por TAG (scatter)
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(by_tag["N_positivos"], by_tag["F1"],
                    s=by_tag["N_eventos"] / 200, alpha=0.7,
                    c=by_tag["Recall"], cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(sc, label="Recall")
    # Anotar top 5 equipamentos
    for _, row in by_tag.head(5).iterrows():
        ax.annotate(row["TAG"], (row["N_positivos"], row["F1"]),
                    fontsize=8, ha="left", xytext=(5, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Volume de eventos pré-DG no teste (escala log)")
    ax.set_ylabel("F1-Score do modelo")
    ax.set_title("Desempenho do Modelo por Equipamento — Jun/2025\n"
                 "Tamanho ∝ volume total de telemetria | Cor = Recall")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "f1_by_equipment_scatter.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'f1_by_equipment_scatter.png'}")

    # ── 5. Persistência ────────────────────────────────────────────────────
    payload = {
        "threshold_used": thr,
        "by_fleet": by_fleet.to_dict(orient="records"),
        "by_tag_top10": top10.to_dict(orient="records"),
        "worst_recall_equipments": worst_recall.to_dict(orient="records"),
        "n_fleets": len(by_fleet),
        "n_equipments_with_positives": len(by_tag),
    }
    out = GOLD_DIR / "fleet_segmentation.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Salvo em {out}")
    print("══════════════ SPRINT 2 (parte 2/2) COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
