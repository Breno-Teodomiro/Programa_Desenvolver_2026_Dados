"""Sprint 9 — Política de threshold custo-ótimo POR FROTA.

Síntese de três sprints anteriores:
  - Sprint 2: análise de custo (FN=R$50k, FP=R$800) → threshold global custo-ótimo
  - Sprint 2/7: segmentação por frota mostrou desempenho heterogêneo (PR-AUC de
    0.87 em 793-D 4S até 0.01 em LeTourneau)
  - Sprint 6: calibração isotônica (probabilidades confiáveis por classe)

Hipótese: um único threshold global é subótimo porque as frotas têm base rates
e separabilidades muito diferentes. Calibrar UM threshold custo-ótimo por frota
(na validação Mai) e aplicá-lo no teste (Jun) deve reduzir o custo operacional
total vs. o threshold único do Sprint 2.

Entrega operacional: uma TABELA DE DECISÃO (frota → threshold) deployável e a
economia adicional quantificada vs. política de threshold único.

Salva: outputs/gold/fleet_threshold_policy.json + figuras em outputs/figures/.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from models import load_model, _load_split, VAL_MONTHS, TEST_MONTHS, TARGET_COL
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Mesmas premissas de custo do Sprint 2 (consistência inter-sprint)
COST_FN = 50_000   # Don't Go não previsto: parada não planejada ~4h × R$12.500/h
COST_FP = 800      # Alarme falso: inspeção preventiva desnecessária

# Grade de busca de threshold (mesma resolução do Sprint 2)
THRESHOLDS = np.linspace(0.05, 0.99, 95)

# Threshold global custo-ótimo herdado do Sprint 2 (baseline a superar)
GLOBAL_COST_THRESHOLD = 0.51


def _load_fleet_labels(gold_files: list[Path], month_set: set[str]) -> np.ndarray:
    """Lê Tag_Frota na MESMA ordem de linhas que _load_split (sem undersampling
    → ordem do parquet é preservada deterministicamente)."""
    files = [f for f in gold_files if any(m in f.stem for m in month_set)]
    return (
        pl.read_parquet([str(f) for f in files], columns=["Tag_Frota"])["Tag_Frota"]
        .to_numpy()
    )


def _confusion_cost(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    cost = fn * COST_FN + fp * COST_FP
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "F1": f1,
            "precision": prec, "recall": rec, "cost_BRL": cost}


def _best_cost_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Threshold que minimiza o custo total na grade. Retorna (thr, custo)."""
    best_thr, best_cost = GLOBAL_COST_THRESHOLD, np.inf
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        cost = fn * COST_FN + fp * COST_FP
        if cost < best_cost:
            best_cost, best_thr = cost, float(t)
    return best_thr, float(best_cost)


def main():
    print("== Carregando modelo e dados ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", []))
    print(f"Modelo: {len(lgbm_features)} features")

    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)

    # ── Validação (Mai) — calibra os thresholds ───────────────────────────
    print("\n== Validação (Mai) — calibração de thresholds ==")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS)
    for c in lgbm_features:
        if c not in X_val.columns:
            X_val[c] = 0.0
    y_prob_val = lgbm.predict_proba(X_val[lgbm_features].fillna(0))[:, 1]
    fleet_val = _load_fleet_labels(gold_files, VAL_MONTHS)
    y_val = y_val.to_numpy()
    assert len(fleet_val) == len(y_val), "desalinhamento val frota/target"
    print(f"  {len(y_val):,} linhas | {int(y_val.sum()):,} positivos")

    # ── Teste (Jun) — avalia a política ────────────────────────────────────
    print("\n== Teste (Jun) — avaliação da política ==")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    for c in lgbm_features:
        if c not in X_test.columns:
            X_test[c] = 0.0
    y_prob_test = lgbm.predict_proba(X_test[lgbm_features].fillna(0))[:, 1]
    fleet_test = _load_fleet_labels(gold_files, TEST_MONTHS)
    y_test = y_test.to_numpy()
    assert len(fleet_test) == len(y_test), "desalinhamento test frota/target"
    print(f"  {len(y_test):,} linhas | {int(y_test.sum()):,} positivos")

    fleets = sorted(set(fleet_val) | set(fleet_test))

    # ── 1. Calibra threshold custo-ótimo por frota (na validação) ──────────
    print("\n== Threshold custo-ótimo por frota (calibrado em Mai) ==")
    policy = {}            # frota → threshold
    val_diag = {}          # diagnóstico por frota na validação
    for fl in fleets:
        m = fleet_val == fl
        n_pos = int(y_val[m].sum())
        if n_pos == 0 or m.sum() == 0:
            # Sem positivos para calibrar → usa global (decisão conservadora)
            policy[fl] = GLOBAL_COST_THRESHOLD
            val_diag[fl] = {"n_val": int(m.sum()), "n_pos_val": n_pos,
                            "threshold": GLOBAL_COST_THRESHOLD, "fallback_global": True}
            print(f"  {fl:<20} sem positivos em Mai → fallback global {GLOBAL_COST_THRESHOLD}")
            continue
        thr, cost = _best_cost_threshold(y_val[m], y_prob_val[m])
        policy[fl] = thr
        val_diag[fl] = {"n_val": int(m.sum()), "n_pos_val": n_pos,
                        "threshold": thr, "val_cost_BRL": cost, "fallback_global": False}
        print(f"  {fl:<20} thr*={thr:.3f}  (n_pos_val={n_pos:,})")

    # ── 2. Aplica política POR FROTA no teste ──────────────────────────────
    print("\n== Avaliação no teste (Jun) — política por frota ==")
    y_pred_policy = np.zeros_like(y_test)
    for fl in fleets:
        m = fleet_test == fl
        y_pred_policy[m] = (y_prob_test[m] >= policy[fl]).astype(int)

    per_fleet = []
    for fl in fleets:
        m = fleet_test == fl
        if m.sum() == 0:
            continue
        cm = _confusion_cost(y_test[m], y_pred_policy[m])
        per_fleet.append({
            "Frota": fl, "threshold": policy[fl],
            "N_eventos": int(m.sum()), "N_positivos": int(y_test[m].sum()),
            **cm,
        })
        print(f"  {fl:<20} thr={policy[fl]:.2f}  TP={cm['TP']:<6} FP={cm['FP']:<7} "
              f"FN={cm['FN']:<5} F1={cm['F1']:.3f} Rec={cm['recall']:.2f} "
              f"custo=R${cm['cost_BRL']:,.0f}")

    policy_total = _confusion_cost(y_test, y_pred_policy)

    # ── 3. Baseline: threshold GLOBAL único (Sprint 2) no teste ────────────
    print("\n== Baseline: threshold global único (Sprint 2) ==")
    y_pred_global = (y_prob_test >= GLOBAL_COST_THRESHOLD).astype(int)
    global_total = _confusion_cost(y_test, y_pred_global)
    global_per_fleet = []
    for fl in fleets:
        m = fleet_test == fl
        if m.sum() == 0:
            continue
        cm = _confusion_cost(y_test[m], y_pred_global[m])
        global_per_fleet.append({"Frota": fl, **cm})

    # ── 4. Economia adicional da política por frota ────────────────────────
    delta_cost = global_total["cost_BRL"] - policy_total["cost_BRL"]
    pct = 100 * delta_cost / global_total["cost_BRL"] if global_total["cost_BRL"] else 0.0
    print("\n" + "═" * 60)
    print(f"  Custo global único  (thr={GLOBAL_COST_THRESHOLD}): R${global_total['cost_BRL']:,.0f}  "
          f"(F1={global_total['F1']:.3f}, FN={global_total['FN']}, FP={global_total['FP']})")
    print(f"  Custo política/frota             : R${policy_total['cost_BRL']:,.0f}  "
          f"(F1={policy_total['F1']:.3f}, FN={policy_total['FN']}, FP={policy_total['FP']})")
    print(f"  ► Economia adicional             : R${delta_cost:,.0f}  ({pct:+.1f}%)")
    print("═" * 60)

    # ── 5. Figuras ─────────────────────────────────────────────────────────
    # Fig A: thresholds por frota
    df_pf = pd.DataFrame(per_fleet).sort_values("threshold")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(df_pf)))
    bars = ax.barh(df_pf["Frota"], df_pf["threshold"], color=colors)
    ax.axvline(GLOBAL_COST_THRESHOLD, color="#EF4444", linestyle="--", linewidth=2,
               label=f"Threshold global único = {GLOBAL_COST_THRESHOLD}")
    for b, t in zip(bars, df_pf["threshold"]):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{t:.2f}", va="center", fontsize=10)
    ax.set_xlabel("Threshold custo-ótimo (calibrado em Mai/2025)")
    ax.set_title("Política de Threshold por Frota vs. Threshold Global Único\n"
                 "Cada frota tem base rate e separabilidade distintas")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fleet_threshold_policy.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'fleet_threshold_policy.png'}")

    # Fig B: custo por frota — global vs política
    g_map = {d["Frota"]: d["cost_BRL"] for d in global_per_fleet}
    p_map = {d["Frota"]: d["cost_BRL"] for d in per_fleet}
    order = sorted(p_map, key=lambda f: p_map[f], reverse=True)
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, [g_map[f] / 1e6 for f in order], w,
           label="Threshold global único", color="#EF4444", alpha=0.85)
    ax.bar(x + w / 2, [p_map[f] / 1e6 for f in order], w,
           label="Política por frota", color="#10B981", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Custo operacional estimado (R$ Milhões)")
    ax.set_title(f"Custo por Frota — Teste Jun/2025\n"
                 f"Economia total da política: R${delta_cost/1e6:,.1f}M ({pct:+.1f}%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fleet_threshold_cost_comparison.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'fleet_threshold_cost_comparison.png'}")

    # ── 6. Persistência ─────────────────────────────────────────────────────
    payload = {
        "cost_assumptions": {"COST_FN_BRL": COST_FN, "COST_FP_BRL": COST_FP,
                             "ratio_FN_FP": COST_FN / COST_FP},
        "global_threshold_baseline": GLOBAL_COST_THRESHOLD,
        "decision_table": {fl: policy[fl] for fl in fleets},
        "val_calibration": val_diag,
        "test_per_fleet_policy": per_fleet,
        "test_per_fleet_global": global_per_fleet,
        "test_total_policy": policy_total,
        "test_total_global": global_total,
        "additional_savings_BRL": delta_cost,
        "additional_savings_pct": pct,
    }
    out_json = GOLD_DIR / "fleet_threshold_policy.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Resultados salvos em {out_json}")
    print("\n══════════════ SPRINT 9 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
