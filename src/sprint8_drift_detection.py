"""Sprint 8 — Detecção de drift mensal (Page-Hinkley + KS-test).

Motivação: Sprint 5 mostrou que treinar modelo separado para escavadeira
falhou por distribution shift severo (variação 600× na taxa DG entre meses).
Aqui implementamos detectores online que dispariam alerta de drift e
disparariam retreino automático em produção.

Sinais monitorados (mensais):
1. Taxa DG observada (ground truth — KPI primário)
2. Média de probabilidade prevista pelo modelo (KPI score — proxy quando label
   não está disponível imediatamente)
3. Distância KS entre distribuições de score de meses consecutivos

Detector: Page-Hinkley test — robusto, simples, padrão industrial.
Para cada série x_t, mantém cumulative sum m_t = sum(x_i - mean - delta).
Alarme quando MAX_t(m_t) - MIN_t(m_t) > lambda.

Salva: outputs/gold/drift_detection.json + figuras em outputs/figures/.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
from models import load_model
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"

MONTH_ORDER = ["jan", "feb", "mar", "abr", "may", "jun"]
MONTH_LABEL = {"jan": "Jan", "feb": "Fev", "mar": "Mar", "abr": "Abr", "may": "Mai", "jun": "Jun"}


def page_hinkley(series: np.ndarray, delta: float = 0.005, lam: float = 0.05) -> dict:
    """Page-Hinkley change detection.

    Args:
        series: sinal monitorado.
        delta: bias mínimo aceito (não dispara para variações pequenas).
        lam: threshold de alarme (escalar; em unidade do sinal).
    Returns:
        dict com índices de detecção e estatísticas cumulativas.
    """
    n = len(series)
    if n < 2:
        return {"detected_at": [], "stat": np.zeros(n).tolist(), "min_stat": np.zeros(n).tolist()}
    mean_run = np.zeros(n)
    m = np.zeros(n)
    mt = np.zeros(n)
    min_m = np.zeros(n)
    mean_run[0] = series[0]
    detected = []
    for t in range(1, n):
        mean_run[t] = mean_run[t-1] + (series[t] - mean_run[t-1]) / (t + 1)
        m[t] = m[t-1] + (series[t] - mean_run[t] - delta)
        min_m[t] = min(min_m[t-1], m[t])
        mt[t] = m[t] - min_m[t]
        if mt[t] > lam:
            detected.append(t)
    return {
        "detected_at": detected,
        "stat": mt.tolist(),
        "min_stat": min_m.tolist(),
        "mean_run": mean_run.tolist(),
    }


def main():
    print("== Carregando modelo principal ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_features = list(getattr(lgbm, "feature_name_", []))

    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)

    # ── Computa estatísticas mensais por frota ─────────────────────────────
    print("\n== Coletando estatísticas mensais ==")
    fleets = ["793-D 2S", "793-D 3S", "793-D 4S", "793-D 5S", "LeTourneau L 1850"]
    monthly_records = []
    score_samples_by_fleet_month = {}  # para KS test

    for f in gold_files:
        mes = f.stem.replace("gold_", "")
        df = pl.read_parquet(
            f, columns=["Tag_Frota", "is_dont_go_next_60m"] + feature_cols
        )
        for fleet in fleets:
            sub = df.filter(pl.col("Tag_Frota") == fleet).to_pandas()
            n = len(sub)
            if n == 0:
                continue
            y = sub["is_dont_go_next_60m"].astype(int).values
            # Adiciona features faltantes
            for c in lgbm_features:
                if c not in sub.columns:
                    sub[c] = 0.0
            X = sub[lgbm_features].fillna(0).astype("float32")
            # Score amostral (até 100K rows por mês×frota para eficiência KS)
            sample_idx = np.random.RandomState(42).choice(
                len(X), size=min(len(X), 100_000), replace=False)
            p = lgbm.predict_proba(X.iloc[sample_idx])[:, 1]
            score_samples_by_fleet_month[(fleet, mes)] = p
            monthly_records.append({
                "mes": mes,
                "mes_label": MONTH_LABEL[mes],
                "frota": fleet,
                "n_eventos": n,
                "n_positivos": int(y.sum()),
                "taxa_dg_pct": 100 * y.mean(),
                "score_mean": float(p.mean()),
                "score_p95": float(np.quantile(p, 0.95)),
            })
            print(f"  {mes:>3} | {fleet:<22} n={n:>9,} | DG={100*y.mean():.4f}% | "
                  f"scorē={p.mean():.4f}")

    df_monthly = pd.DataFrame(monthly_records)

    # ── Page-Hinkley por frota sobre taxa_dg_pct e score_mean ────────────
    print("\n== Aplicando Page-Hinkley por frota ==")
    drift_results = {}
    for fleet in fleets:
        sub = df_monthly[df_monthly["frota"] == fleet].copy()
        sub["mes_idx"] = sub["mes"].map({m: i for i, m in enumerate(MONTH_ORDER)})
        sub = sub.sort_values("mes_idx")
        if len(sub) < 2:
            continue
        # Page-Hinkley sobre taxa observada (em pp/100, então valores pequenos)
        ph_obs = page_hinkley(sub["taxa_dg_pct"].values / 100.0, delta=0.001, lam=0.02)
        # Page-Hinkley sobre score médio
        ph_score = page_hinkley(sub["score_mean"].values, delta=0.001, lam=0.02)
        # KS test entre meses consecutivos
        ks_pvals = [None]
        ks_stats = [None]
        for i in range(1, len(sub)):
            mes_prev = sub.iloc[i-1]["mes"]
            mes_cur = sub.iloc[i]["mes"]
            s_prev = score_samples_by_fleet_month.get((fleet, mes_prev))
            s_cur = score_samples_by_fleet_month.get((fleet, mes_cur))
            if s_prev is None or s_cur is None:
                ks_pvals.append(None); ks_stats.append(None)
                continue
            stat, pval = ks_2samp(s_prev, s_cur)
            ks_pvals.append(float(pval))
            ks_stats.append(float(stat))

        drift_results[fleet] = {
            "meses": sub["mes_label"].tolist(),
            "taxa_dg_pct": sub["taxa_dg_pct"].tolist(),
            "score_mean": sub["score_mean"].tolist(),
            "ph_obs_stat": ph_obs["stat"],
            "ph_obs_detected_at": ph_obs["detected_at"],
            "ph_score_stat": ph_score["stat"],
            "ph_score_detected_at": ph_score["detected_at"],
            "ks_pvalues": ks_pvals,
            "ks_stats": ks_stats,
        }
        print(f"\n  {fleet}:")
        if ph_obs["detected_at"]:
            print(f"    Drift detectado em (taxa DG): meses {[sub['mes_label'].iloc[i] for i in ph_obs['detected_at']]}")
        else:
            print(f"    Sem drift detectado em taxa DG")
        if ph_score["detected_at"]:
            print(f"    Drift detectado em (score): meses {[sub['mes_label'].iloc[i] for i in ph_score['detected_at']]}")

    # ── Figuras ────────────────────────────────────────────────────────────
    print("\n== Gerando figuras ==")
    fig, axes = plt.subplots(len(fleets), 2, figsize=(14, 3 * len(fleets)), sharex=True)
    if len(fleets) == 1:
        axes = [axes]
    for i, fleet in enumerate(fleets):
        if fleet not in drift_results:
            continue
        r = drift_results[fleet]
        meses = r["meses"]
        x = np.arange(len(meses))

        ax_l = axes[i][0]
        ax_l.plot(x, r["taxa_dg_pct"], "o-", color="#3B82F6", linewidth=2, label="Taxa DG observada (%)")
        for det in r["ph_obs_detected_at"]:
            ax_l.axvline(det, color="#EF4444", linestyle="--", alpha=0.6)
        ax_l.set_title(f"{fleet} — Taxa DG mensal + Page-Hinkley")
        ax_l.set_xticks(x); ax_l.set_xticklabels(meses)
        ax_l.set_ylabel("Taxa DG (%)")
        ax_l.legend(loc="upper left", fontsize=9)
        ax_l.grid(alpha=0.3)

        ax_r = axes[i][1]
        ax_r.plot(x, r["score_mean"], "s-", color="#10B981", linewidth=2, label="Score médio LightGBM")
        for det in r["ph_score_detected_at"]:
            ax_r.axvline(det, color="#EF4444", linestyle="--", alpha=0.6)
        ax_r.set_title(f"{fleet} — Score Médio + Page-Hinkley")
        ax_r.set_xticks(x); ax_r.set_xticklabels(meses)
        ax_r.set_ylabel("Score médio")
        ax_r.legend(loc="upper left", fontsize=9)
        ax_r.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "drift_detection_by_fleet.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'drift_detection_by_fleet.png'}")

    # KS heatmap (frotas × pares meses)
    fig, ax = plt.subplots(figsize=(11, 5))
    ks_matrix = []
    rows_lbl = []
    cols_lbl = [f"{MONTH_ORDER[i].upper()}→{MONTH_ORDER[i+1].upper()}" for i in range(len(MONTH_ORDER) - 1)]
    for fleet in fleets:
        if fleet not in drift_results:
            continue
        rows_lbl.append(fleet)
        ks_vals = drift_results[fleet]["ks_stats"][1:]  # primeiro é None
        # Pad to len(cols_lbl)
        while len(ks_vals) < len(cols_lbl):
            ks_vals.append(np.nan)
        ks_matrix.append(ks_vals)
    ks_arr = np.array(ks_matrix, dtype=float)
    im = ax.imshow(ks_arr, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(cols_lbl))); ax.set_xticklabels(cols_lbl)
    ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl)
    ax.set_title("Distância Kolmogorov-Smirnov entre Distribuições de Score — Meses Consecutivos\n"
                 "(KS=0 distribuições iguais; KS=1 totalmente disjuntas)")
    for i in range(len(rows_lbl)):
        for j in range(len(cols_lbl)):
            v = ks_arr[i][j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.4 else "black", fontsize=9)
    plt.colorbar(im, label="KS statistic")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "drift_ks_heatmap.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'drift_ks_heatmap.png'}")

    # ── Persistência ───────────────────────────────────────────────────────
    payload = {
        "method": "Page-Hinkley + KS-2sample",
        "parameters": {"PH_delta": 0.001, "PH_lambda": 0.02},
        "monthly_stats": df_monthly.to_dict(orient="records"),
        "drift_by_fleet": drift_results,
        "summary": {
            fleet: {
                "drift_detected_taxa_dg": [drift_results[fleet]["meses"][i] for i in drift_results[fleet]["ph_obs_detected_at"]] if fleet in drift_results else [],
                "drift_detected_score": [drift_results[fleet]["meses"][i] for i in drift_results[fleet]["ph_score_detected_at"]] if fleet in drift_results else [],
            }
            for fleet in fleets if fleet in drift_results
        },
    }
    with open(GOLD_DIR / "drift_detection.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Resultados salvos em {GOLD_DIR / 'drift_detection.json'}")
    print("══════════════ SPRINT 8 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
