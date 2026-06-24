"""Sprint 2 — Análise de erros (FP/FN) + análise de custo (CM 5.2).

Para o modelo LightGBM no conjunto de teste (Jun/2025):
1. Identifica casos FP e FN representativos e descreve por que o modelo errou.
2. Constrói matriz de custo operacional FN vs FP e justifica o threshold ótimo
   por custo (não apenas F1).

Salva: outputs/gold/error_cost_analysis.json + figuras em outputs/figures/.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from models import load_model, _load_split, TEST_MONTHS, VAL_MONTHS, TARGET_COL
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Parâmetros de custo (ordem de grandeza — referência operacional) ────────
# Estimativas conservadoras baseadas em literatura de mineração pesada.
# Custos em BRL aproximados:
COST_FN = 50_000   # Don't Go não previsto: parada não planejada de equipamento
                   # ~4h indisponibilidade × R$ 12.500/h (custo médio caminhão 793-D)
COST_FP = 800      # Alarme falso: inspeção preventiva desnecessária
                   # ~1h inspeção técnica + parada operacional curta
# Razão FN/FP = 62.5 → threshold ótimo por custo deve favorecer recall


# ── Custo por EPISÓDIO (não por linha de telemetria) ─────────────────────────
# Um único Don't Go gera dezenas/centenas de linhas de evento na janela pré-DG.
# Precificar cada linha como uma parada independente de R$50k superestima o custo
# em ordens de grandeza. Aqui deduplicamos: 1 episódio de DG = 1 parada física.

def _count_dg_episodes(df_pos: pd.DataFrame) -> pd.DataFrame:
    """Agrupa linhas pré-DG pelo Don't Go futuro a que apontam (1 grupo = 1 parada).

    O instante do DG ≈ Data_Evento + minutes_to_next_dg; linhas que apontam para o
    mesmo DG compartilham essa chave. Episódio é "capturado" se o modelo dispara
    (y_pred=1) em ao menos uma linha da janela.
    """
    d = df_pos.copy()
    d["Data_Evento"] = pd.to_datetime(d["Data_Evento"])
    d["dg_time"] = d["Data_Evento"] + pd.to_timedelta(
        d["minutes_to_next_dg"].fillna(0).clip(lower=0), unit="m")
    d["dg_key"] = d["dg_time"].dt.floor("min")
    return d.groupby(["TAG", "dg_key"])["y_pred"].max().reset_index()


def _count_fp_episodes(df_fp: pd.DataFrame, gap_min: int = 60) -> int:
    """Conta rajadas de falsos alarmes por equipamento (gap > gap_min inicia nova).

    Cada rajada equivale a uma inspeção preventiva desnecessária despachada.
    """
    if len(df_fp) == 0:
        return 0
    d = df_fp.copy()
    d["Data_Evento"] = pd.to_datetime(d["Data_Evento"])
    n = 0
    for _tag, g in d.sort_values("Data_Evento").groupby("TAG"):
        t = g["Data_Evento"].values
        if len(t) == 0:
            continue
        gaps = np.diff(t).astype("timedelta64[m]").astype(float)
        n += 1 + int((gaps > gap_min).sum())
    return n


def compute_episode_cost(meta: pd.DataFrame, threshold: float,
                         cost_fn: int, cost_fp: int, gap_min: int = 60) -> dict:
    yp = (meta["y_prob"].values >= threshold).astype(int)
    m = meta.assign(y_pred=yp)
    ep = _count_dg_episodes(m[m["y_true"] == 1])
    n_ep = len(ep)
    missed = int((ep["y_pred"] == 0).sum())
    fp_ep = _count_fp_episodes(m[(m["y_true"] == 0) & (m["y_pred"] == 1)], gap_min)
    cost = missed * cost_fn + fp_ep * cost_fp
    return {"threshold": float(threshold), "n_dg_episodes": int(n_ep),
            "caught_episodes": int(n_ep - missed), "missed_episodes": int(missed),
            "fp_episodes": int(fp_ep), "cost_BRL": int(cost)}


def main():
    print("== Carregando modelo e dados ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", []))
    print(f"Modelo: {len(lgbm_features)} features")

    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))

    # Features do gold atual (60); precisamos das 54 do modelo
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)

    # Validação para escolha de threshold ótimo por custo
    print("\n== Validação (Mai) ==")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS)
    for c in lgbm_features:
        if c not in X_val.columns:
            X_val[c] = 0.0
    X_val_lgbm = X_val[lgbm_features].fillna(0)
    y_prob_val = lgbm.predict_proba(X_val_lgbm)[:, 1]
    print(f"  {len(X_val):,} linhas | {y_val.sum():,} positivos")

    # Teste
    print("\n== Teste (Jun) ==")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    for c in lgbm_features:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test_lgbm = X_test[lgbm_features].fillna(0)
    y_prob_test = lgbm.predict_proba(X_test_lgbm)[:, 1]
    print(f"  {len(X_test):,} linhas | {y_test.sum():,} positivos")

    # ── 1. Curva de custo total vs threshold ──────────────────────────────
    print("\n== Análise de custo vs threshold (val) ==")
    thresholds = np.linspace(0.05, 0.99, 95)
    costs, fns, fps, f1s = [], [], [], []
    for t in thresholds:
        y_pred = (y_prob_val >= t).astype(int)
        fn = int(((y_val == 1) & (y_pred == 0)).sum())
        fp = int(((y_val == 0) & (y_pred == 1)).sum())
        cost = fn * COST_FN + fp * COST_FP
        costs.append(cost)
        fns.append(fn)
        fps.append(fp)
        f1s.append(f1_score(y_val, y_pred, zero_division=0))

    best_cost_idx = int(np.argmin(costs))
    best_cost_thr = float(thresholds[best_cost_idx])
    best_f1_idx = int(np.argmax(f1s))
    best_f1_thr = float(thresholds[best_f1_idx])

    print(f"Threshold ótimo por F1 (val): {best_f1_thr:.3f} (F1={f1s[best_f1_idx]:.4f}, custo=R${costs[best_f1_idx]:,.0f})")
    print(f"Threshold ótimo por custo (val): {best_cost_thr:.3f} (custo=R${costs[best_cost_idx]:,.0f}, F1={f1s[best_cost_idx]:.4f})")
    print(f"Δ custo F1-ótimo vs custo-ótimo: R${costs[best_f1_idx] - costs[best_cost_idx]:,.0f}")

    # Figura: custo total vs threshold
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, np.array(costs) / 1e6, color="#EF4444", linewidth=2, label="Custo total (R$ M)")
    ax1.axvline(best_cost_thr, color="#EF4444", linestyle="--", alpha=0.6, label=f"Mín. custo @ {best_cost_thr:.2f}")
    ax1.axvline(best_f1_thr, color="#3B82F6", linestyle=":", alpha=0.6, label=f"Máx. F1 @ {best_f1_thr:.2f}")
    ax1.set_xlabel("Threshold de decisão")
    ax1.set_ylabel("Custo total estimado (R$ Milhões)", color="#EF4444")
    ax1.tick_params(axis="y", labelcolor="#EF4444")
    ax2 = ax1.twinx()
    ax2.plot(thresholds, f1s, color="#3B82F6", linewidth=2, alpha=0.7, label="F1-Score")
    ax2.set_ylabel("F1-Score", color="#3B82F6")
    ax2.tick_params(axis="y", labelcolor="#3B82F6")
    plt.title(f"Custo Total vs F1-Score por Threshold (validação Mai/2025)\n"
              f"Custos: FN=R${COST_FN:,.0f}, FP=R${COST_FP:,.0f} (razão {COST_FN/COST_FP:.0f}×)")
    fig.tight_layout()
    plt.savefig(FIGURES_DIR / "cost_vs_threshold.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'cost_vs_threshold.png'}")

    # ── 2. Confusion Matrix nos dois thresholds (teste) ───────────────────
    print("\n== Métricas no teste (Jun) por threshold ==")
    test_results = {}
    for label, thr in [("f1_optimo", best_f1_thr), ("custo_otimo", best_cost_thr)]:
        y_pred = (y_prob_test >= thr).astype(int)
        tp = int(((y_test == 1) & (y_pred == 1)).sum())
        tn = int(((y_test == 0) & (y_pred == 0)).sum())
        fp = int(((y_test == 0) & (y_pred == 1)).sum())
        fn = int(((y_test == 1) & (y_pred == 0)).sum())
        cost = fn * COST_FN + fp * COST_FP
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        test_results[label] = {
            "threshold": thr, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "F1": f1, "precision": prec, "recall": rec,
            "cost_BRL": cost,
        }
        print(f"  [{label}] thr={thr:.3f}  TP={tp}  FP={fp}  FN={fn}  "
              f"F1={f1:.4f}  custo=R${cost:,.0f}")

    # test_results sempre contém "f1_optimo" e "custo_otimo" (preenchidos no loop acima)
    delta_cost = test_results["f1_optimo"]["cost_BRL"] - test_results["custo_otimo"]["cost_BRL"]
    print(f"  Δ custo no teste: R${delta_cost:,.0f} (custo-ótimo economiza esse valor)")

    # ── 3. Análise de casos representativos FP e FN ────────────────────────
    print("\n== Análise de casos FP/FN ==")
    # Usa o threshold custo-ótimo para classificar
    thr = best_cost_thr
    y_pred = (y_prob_test >= thr).astype(int)

    # Carrega metadados (TAG, Data_Evento, Tag_Frota) para o teste
    test_files = [f for f in gold_files if any(m in f.stem for m in TEST_MONTHS)]
    meta = pl.read_parquet(
        [str(f) for f in test_files],
        columns=["Data_Evento", "TAG", "Tag_Frota", "Id_Alarme", "Id_Criticidade",
                 "n_criticos_30m", "n_criticos_60m", "aceleracao_criticos", "minutes_to_next_dg"],
    ).to_pandas()
    meta["y_true"] = y_test.values
    meta["y_pred"] = y_pred
    meta["y_prob"] = y_prob_test

    # ── Custo por EPISÓDIO (deduplicado) — métrica honesta de impacto ─────────
    print("\n== Custo por EPISÓDIO de Don't Go (não por linha de evento) ==")
    ep_f1 = compute_episode_cost(meta, best_f1_thr, COST_FN, COST_FP)
    ep_cost = compute_episode_cost(meta, best_cost_thr, COST_FN, COST_FP)
    episode_delta = ep_f1["cost_BRL"] - ep_cost["cost_BRL"]
    print(f"  Episódios de DG no teste: {ep_f1['n_dg_episodes']:,} "
          f"(vs {int(meta['y_true'].sum()):,} linhas pré-DG)")
    print(f"  [F1-ótimo   thr={best_f1_thr:.2f}] perdidos={ep_f1['missed_episodes']}  "
          f"FP_ep={ep_f1['fp_episodes']}  custo=R${ep_f1['cost_BRL']:,}")
    print(f"  [custo-ótimo thr={best_cost_thr:.2f}] perdidos={ep_cost['missed_episodes']}  "
          f"FP_ep={ep_cost['fp_episodes']}  custo=R${ep_cost['cost_BRL']:,}")
    print(f"  Economia por episódio (custo-ótimo vs F1-ótimo): R${episode_delta:,}")

    fp_mask = (meta["y_true"] == 0) & (meta["y_pred"] == 1)
    fn_mask = (meta["y_true"] == 1) & (meta["y_pred"] == 0)
    tp_mask = (meta["y_true"] == 1) & (meta["y_pred"] == 1)

    print(f"  TP: {tp_mask.sum():,}  FP: {fp_mask.sum():,}  FN: {fn_mask.sum():,}")

    # 3 FPs representativos: maior y_prob (modelo mais "convencido")
    fp_top = meta[fp_mask].nlargest(3, "y_prob")
    # 3 FNs representativos: menor y_prob entre os reais (modelo mais "errado")
    fn_top = meta[fn_mask].nsmallest(3, "y_prob")

    cases = {"FP": [], "FN": []}
    for _, row in fp_top.iterrows():
        cases["FP"].append({
            "Data_Evento": str(row["Data_Evento"]),
            "TAG": row["TAG"],
            "Tag_Frota": row["Tag_Frota"],
            "y_prob": float(row["y_prob"]),
            "n_criticos_30m": float(row["n_criticos_30m"]) if pd.notna(row["n_criticos_30m"]) else None,
            "n_criticos_60m": float(row["n_criticos_60m"]) if pd.notna(row["n_criticos_60m"]) else None,
            "aceleracao_criticos": float(row["aceleracao_criticos"]) if pd.notna(row["aceleracao_criticos"]) else None,
            "minutes_to_next_dg": float(row["minutes_to_next_dg"]) if pd.notna(row["minutes_to_next_dg"]) else None,
        })
    for _, row in fn_top.iterrows():
        cases["FN"].append({
            "Data_Evento": str(row["Data_Evento"]),
            "TAG": row["TAG"],
            "Tag_Frota": row["Tag_Frota"],
            "y_prob": float(row["y_prob"]),
            "n_criticos_30m": float(row["n_criticos_30m"]) if pd.notna(row["n_criticos_30m"]) else None,
            "n_criticos_60m": float(row["n_criticos_60m"]) if pd.notna(row["n_criticos_60m"]) else None,
            "aceleracao_criticos": float(row["aceleracao_criticos"]) if pd.notna(row["aceleracao_criticos"]) else None,
            "minutes_to_next_dg": float(row["minutes_to_next_dg"]) if pd.notna(row["minutes_to_next_dg"]) else None,
        })

    print("\n--- Top 3 FPs (modelo previu DG, mas não houve) ---")
    for i, c in enumerate(cases["FP"], 1):
        print(f"  FP-{i}: {c['TAG']} ({c['Tag_Frota']}) @ {c['Data_Evento'][:19]}  "
              f"prob={c['y_prob']:.3f}  n_crit_30m={c['n_criticos_30m']}  acel={c['aceleracao_criticos']}")

    print("\n--- Top 3 FNs (DG real não previsto) ---")
    for i, c in enumerate(cases["FN"], 1):
        print(f"  FN-{i}: {c['TAG']} ({c['Tag_Frota']}) @ {c['Data_Evento'][:19]}  "
              f"prob={c['y_prob']:.3f}  n_crit_30m={c['n_criticos_30m']}  acel={c['aceleracao_criticos']}  "
              f"min_to_dg={c['minutes_to_next_dg']}")

    # ── 4. Figura: distribuição de probabilidades por classe ───────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_prob_test[y_test == 0], bins=80, alpha=0.6, label="Negativos (não-DG)", color="#10B981", density=True)
    ax.hist(y_prob_test[y_test == 1], bins=80, alpha=0.6, label="Positivos (pré-DG)", color="#EF4444", density=True)
    ax.axvline(thr, color="black", linestyle="--", linewidth=2, label=f"Threshold custo-ótimo = {thr:.3f}")
    ax.axvline(best_f1_thr, color="gray", linestyle=":", linewidth=2, label=f"Threshold F1-ótimo = {best_f1_thr:.3f}")
    ax.set_xlabel("Probabilidade prevista pelo LightGBM")
    ax.set_ylabel("Densidade (normalizada por classe)")
    ax.set_title("Distribuição das Probabilidades por Classe — Teste Jun/2025\n"
                 "Sobreposição = região de FP/FN; threshold separa as decisões")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prob_distribution_by_class.png", dpi=150)
    plt.close()
    print(f"\n  ✓ {FIGURES_DIR / 'prob_distribution_by_class.png'}")

    # ── 5. Persistência ─────────────────────────────────────────────────────
    payload = {
        "cost_assumptions": {
            "COST_FN_BRL": COST_FN,
            "COST_FP_BRL": COST_FP,
            "ratio_FN_FP": COST_FN / COST_FP,
            "rationale": "FN = parada não planejada ~4h × R$12.500/h. FP = inspeção preventiva ~1h.",
        },
        "best_f1_threshold_val": best_f1_thr,
        "best_cost_threshold_val": best_cost_thr,
        "test_results_by_threshold": test_results,
        "delta_cost_test_BRL": delta_cost,
        "delta_cost_test_note": "Valor por LINHA de telemetria — superestima (cada "
                                "episódio de DG gera muitas linhas). Use episode_level_cost.",
        "episode_level_cost": {
            "note": "Custo deduplicado: 1 episódio de Don't Go = 1 parada física. "
                    "Métrica honesta de impacto operacional.",
            "gap_min_fp_cluster": 60,
            "f1_optimo": ep_f1,
            "custo_otimo": ep_cost,
            "delta_cost_episode_BRL": int(episode_delta),
        },
        "cases_fp": cases["FP"],
        "cases_fn": cases["FN"],
        "summary_test": {
            "n_test": int(len(y_test)),
            "n_positives": int(y_test.sum()),
            "tp_total": int(tp_mask.sum()),
            "fp_total": int(fp_mask.sum()),
            "fn_total": int(fn_mask.sum()),
        },
    }
    out_json = GOLD_DIR / "error_cost_analysis.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Resultados salvos em {out_json}")
    print("\n══════════════ SPRINT 2 (parte 1/2) COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
