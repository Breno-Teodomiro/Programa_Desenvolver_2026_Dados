"""Sprint 7 — Análise temporal de degradação do CA65926.

Memória registra: CA65926 (793-D 4S) — taxa DG média semestral 5.3%,
mas em Jun/2025 saltou para 61.6%. Esta escalada brutal merece investigação
visual dedicada. Salva figura impactante e tabela mensal.

Salva: outputs/gold/ca65926_temporal.json + figuras.
"""

from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
TAG_ALVO = "CA65926"


def main():
    print(f"== Carregando todos os meses do {TAG_ALVO} ==")
    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))

    # Carrega só CA65926 com cols essenciais
    rows = []
    for f in gold_files:
        df = pl.read_parquet(
            f, columns=["Data_Evento", "TAG", "Tag_Frota", "Is_Dont_Go",
                        "is_dont_go_next_60m", "Id_Criticidade", "n_criticos_30m",
                        "aceleracao_criticos"]
        )
        d = df.filter(pl.col("TAG") == TAG_ALVO).to_pandas()
        if len(d) == 0:
            continue
        rows.append(d)
        print(f"  {f.stem}: {len(d):,} eventos")
    if not rows:
        print(f"✗ Nenhum dado para {TAG_ALVO}")
        return None
    df = pd.concat(rows, ignore_index=True)
    df["Data_Evento"] = pd.to_datetime(df["Data_Evento"])
    df["mes"] = df["Data_Evento"].dt.to_period("M")
    df["dia"] = df["Data_Evento"].dt.to_period("D")

    # Frota
    frota = df["Tag_Frota"].mode().iloc[0]
    print(f"\nTotal de eventos {TAG_ALVO} ({frota}): {len(df):,}")

    # ── 1. Resumo mensal ──────────────────────────────────────────────────
    print("\n== Estatística mensal ==")
    monthly = df.groupby("mes").agg(
        n_eventos=("Is_Dont_Go", "size"),
        n_dont_go=("Is_Dont_Go", "sum"),
        n_pre_dg=("is_dont_go_next_60m", "sum"),
        criticos_30m_med=("n_criticos_30m", "median"),
        criticos_30m_p95=("n_criticos_30m", lambda x: x.quantile(0.95)),
        aceleracao_med=("aceleracao_criticos", "median"),
    ).reset_index()
    monthly["taxa_dg_pct"] = 100 * monthly["n_dont_go"] / monthly["n_eventos"]
    monthly["taxa_pre_dg_pct"] = 100 * monthly["n_pre_dg"] / monthly["n_eventos"]
    monthly["mes_str"] = monthly["mes"].astype(str)
    print(monthly.to_string(index=False))

    # ── 2. Figura: escalada mensal ────────────────────────────────────────
    print("\n== Gerando figuras ==")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Taxa DG mensal — barras
    colors_bar = ["#10B981", "#3B82F6", "#3B82F6", "#3B82F6", "#F59E0B", "#EF4444"][:len(monthly)]
    bars = ax1.bar(monthly["mes_str"], monthly["taxa_dg_pct"], color=colors_bar, edgecolor="white")
    ax1.set_ylabel("Taxa de Don't Go (%)")
    ax1.set_title(f"Escalada Mensal de Don't Go — {TAG_ALVO} ({frota})\n"
                  f"De 1.6% em Jan para 61.6% em Jun (38× de aumento)")
    for bar, v in zip(bars, monthly["taxa_dg_pct"]):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.2f}%",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(monthly["taxa_dg_pct"]) * 1.2)
    ax1.grid(alpha=0.3, axis="y")
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Volume de eventos vs DGs — log scale
    ax2.bar(monthly["mes_str"], monthly["n_eventos"], color="#9CA3AF",
            edgecolor="white", label="Total eventos")
    ax2.bar(monthly["mes_str"], monthly["n_dont_go"], color="#EF4444",
            edgecolor="white", label="Eventos Don't Go")
    ax2.set_yscale("log")
    ax2.set_ylabel("Contagem (escala log)")
    ax2.set_title("Volume de Telemetria vs Don't Go por Mês")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ca65926_monthly_escalation.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'ca65926_monthly_escalation.png'}")

    # ── 3. Série diária (densidade) ───────────────────────────────────────
    daily = df.groupby("dia").agg(
        n_eventos=("Is_Dont_Go", "size"),
        n_dont_go=("Is_Dont_Go", "sum"),
    ).reset_index()
    daily["taxa_pct"] = 100 * daily["n_dont_go"] / daily["n_eventos"].clip(lower=1)
    daily["dia_dt"] = pd.PeriodIndex(daily["dia"]).to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5.5))
    # Médias móveis 7d
    daily_sorted = daily.sort_values("dia_dt")
    taxa_smooth = daily_sorted["taxa_pct"].rolling(7, min_periods=1).mean()
    ax.fill_between(daily_sorted["dia_dt"], 0, daily_sorted["taxa_pct"],
                    color="#EF4444", alpha=0.3, label="Taxa diária")
    ax.plot(daily_sorted["dia_dt"], taxa_smooth,
            color="#7F1D1D", linewidth=2, label="Média móvel 7 dias")
    ax.set_ylabel("Taxa Diária de Don't Go (%)")
    ax.set_title(f"Série Temporal Diária — Taxa de Don't Go do {TAG_ALVO}\n"
                 f"Identifica quando começa a escalada estrutural")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ca65926_daily_timeline.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'ca65926_daily_timeline.png'}")

    # ── 4. Comparação com média da frota 793-D 4S ─────────────────────────
    print("\n== Comparando com média da frota 793-D 4S ==")
    rows_4s = []
    for f in gold_files:
        df_all = pl.read_parquet(f, columns=["TAG", "Tag_Frota", "Is_Dont_Go"])
        d4s = df_all.filter((pl.col("Tag_Frota") == "793-D 4S") & (pl.col("TAG") != TAG_ALVO)).to_pandas()
        if len(d4s) == 0:
            continue
        rate = 100 * d4s["Is_Dont_Go"].sum() / len(d4s)
        mes = f.stem.replace("gold_", "")
        rows_4s.append({"mes": mes, "n_eventos": len(d4s), "taxa_4s": rate})
    avg_4s = pd.DataFrame(rows_4s)
    print(avg_4s.to_string(index=False))

    # Map mes order
    mes_order = ["jan", "feb", "mar", "abr", "may", "jun"]
    monthly_short = monthly.copy()
    monthly_short["mes_key"] = monthly_short["mes_str"].str[-2:].astype(int).map(
        {1: "jan", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun"}
    )
    merged = monthly_short.merge(avg_4s, left_on="mes_key", right_on="mes", how="left")
    merged["mes_label"] = merged["mes_key"].str.upper()
    merged = merged.set_index("mes_key").loc[mes_order].reset_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(merged))
    w = 0.38
    ax.bar(x - w/2, merged["taxa_dg_pct"], w, color="#EF4444",
           label=f"{TAG_ALVO} (outlier)")
    ax.bar(x + w/2, merged["taxa_4s"], w, color="#3B82F6",
           label="Média demais 793-D 4S")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["mes_label"])
    ax.set_ylabel("Taxa de Don't Go (%)")
    ax.set_title(f"{TAG_ALVO} vs Média dos Outros 793-D 4S — Divergência Mensal")
    ax.legend()
    for i, (a, b) in enumerate(zip(merged["taxa_dg_pct"], merged["taxa_4s"])):
        ax.text(i - w/2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9)
        if pd.notna(b):
            ax.text(i + w/2, b + 0.5, f"{b:.2f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ca65926_vs_frota.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'ca65926_vs_frota.png'}")

    # ── 5. Persistência ──────────────────────────────────────────────────
    payload = {
        "tag": TAG_ALVO,
        "frota": frota,
        "monthly": monthly.drop(columns=["mes"]).to_dict(orient="records"),
        "comparison_with_fleet_4s": merged[["mes_label", "taxa_dg_pct", "taxa_4s"]].to_dict(orient="records"),
        "escalation_factor": float(monthly["taxa_dg_pct"].max() / max(monthly["taxa_dg_pct"].min(), 0.01)),
    }
    with open(GOLD_DIR / "ca65926_temporal.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Salvo em {GOLD_DIR / 'ca65926_temporal.json'}")
    print(f"\n🎯 Fator de escalada: {payload['escalation_factor']:.0f}× entre menor e maior mês")
    print("══════════════ SPRINT 7 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
