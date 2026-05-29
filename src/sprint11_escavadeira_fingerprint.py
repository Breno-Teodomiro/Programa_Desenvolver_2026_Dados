"""Sprint 11 — Fingerprint específico da escavadeira LeTourneau L 1850.

Motivação: Sprint 2 mostrou que o modelo geral é cego à escavadeira (Recall=0,14,
PR-AUC=0,01) e Sprint 5 mostrou que um modelo dedicado também falha (distribution
shift). A hipótese remanescente (trabalho futuro #1): o problema é de FEATURE
ENGINEERING — o fingerprint global usa os top-30 alarmes por FREQUÊNCIA, mas a
escavadeira responde por 91% de todos os eventos, então esses alarmes são, em
grande parte, ruído rotineiro da própria escavadeira, não sinais de Don't Go.

Aqui trocamos o critério de FREQUÊNCIA por LIFT PREDITIVO: para cada alarme,
mede-se quanto sua presença num evento eleva a probabilidade de um Don't Go nos
próximos 60min (is_dont_go_next_60m), separadamente para escavadeira e caminhões.

Perguntas respondidas:
  1. Existe um fingerprint de alto lift para a escavadeira? Quão forte?
  2. Ele coincide com o top-30 global (frequência) usado pelo modelo atual?
  3. A assinatura de falha da escavadeira difere da dos caminhões?
  4. Que cobertura dos DG da escavadeira o top-30 global captura vs. um top-N
     por lift específico?

Salva: outputs/gold/escavadeira_fingerprint.json + figuras.
"""

from pathlib import Path
import json
import sys
import duckdb
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SILVER_GLOB = str(ROOT / "outputs" / "silver" / "silver_*.parquet")
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ESCAVADEIRA = "LeTourneau L 1850"
MIN_SUPPORT = 300   # alarme precisa ocorrer ao menos N vezes p/ lift ser confiável
TARGET = "is_dont_go_next_60m"


def _per_alarm_lift(where: str) -> "duckdb.DuckDBPyRelation":
    """Estatística por alarme: suporte, taxa pré-DG e lift sobre a base do grupo."""
    return duckdb.query(f"""
        WITH base AS (
            SELECT AVG(CAST({TARGET} AS DOUBLE)) AS br
            FROM read_parquet('{SILVER_GLOB}') WHERE {where}
        )
        SELECT
            s.Id_Alarme,
            ANY_VALUE(s.Alarme) AS Alarme,
            COUNT(*) AS suporte,
            SUM(CAST(s.{TARGET} AS INT)) AS n_pre_dg,
            AVG(CAST(s.{TARGET} AS DOUBLE)) AS taxa_pre_dg,
            AVG(CAST(s.{TARGET} AS DOUBLE)) / (SELECT br FROM base) AS lift
        FROM read_parquet('{SILVER_GLOB}') s
        WHERE {where}
        GROUP BY s.Id_Alarme
        HAVING COUNT(*) >= {MIN_SUPPORT}
        ORDER BY lift DESC
    """)


def main():
    print("== Base rates por grupo ==")
    br = duckdb.query(f"""
        SELECT
          CASE WHEN Tag_Frota = '{ESCAVADEIRA}' THEN 'escavadeira' ELSE 'caminhao' END AS grupo,
          COUNT(*) AS n,
          SUM(CAST({TARGET} AS INT)) AS pre_dg,
          AVG(CAST({TARGET} AS DOUBLE)) AS base_rate
        FROM read_parquet('{SILVER_GLOB}')
        GROUP BY grupo
    """).df()
    print(br.to_string(index=False))
    br_esc = float(br.loc[br.grupo == "escavadeira", "base_rate"].iloc[0])
    br_cam = float(br.loc[br.grupo == "caminhao", "base_rate"].iloc[0])

    # ── 1. Fingerprint por lift — escavadeira ──────────────────────────────
    print("\n== Fingerprint por LIFT — Escavadeira ==")
    esc = _per_alarm_lift(f"Tag_Frota = '{ESCAVADEIRA}'").df()
    print(f"  {len(esc)} alarmes com suporte >= {MIN_SUPPORT}")
    print(esc.head(15)[["Id_Alarme", "Alarme", "suporte", "taxa_pre_dg", "lift"]].to_string(index=False))

    # ── 2. Fingerprint por lift — caminhões (contraste) ────────────────────
    print("\n== Fingerprint por LIFT — Caminhões (contraste) ==")
    cam = _per_alarm_lift(f"Tag_Frota <> '{ESCAVADEIRA}'").df()
    print(cam.head(10)[["Id_Alarme", "Alarme", "suporte", "taxa_pre_dg", "lift"]].to_string(index=False))

    # ── 3. Top-30 global por FREQUÊNCIA (critério do modelo atual) ─────────
    print("\n== Top-30 global por FREQUÊNCIA (fingerprint do modelo atual) ==")
    glob_freq = duckdb.query(f"""
        SELECT Id_Alarme, COUNT(*) AS cnt
        FROM read_parquet('{SILVER_GLOB}')
        GROUP BY Id_Alarme ORDER BY cnt DESC LIMIT 30
    """).df()
    global_top30 = set(int(x) for x in glob_freq.Id_Alarme.tolist())

    esc_top_lift = [int(x) for x in esc.head(30).Id_Alarme.tolist()]
    overlap = sorted(set(esc_top_lift) & global_top30)
    print(f"  Top-30 escavadeira por lift ∩ top-30 global por frequência: "
          f"{len(overlap)} alarmes em comum")
    print(f"  → {len(esc_top_lift) - len(overlap)} dos alarmes de alto lift da escavadeira "
          f"estão FORA do fingerprint atual")

    # ── 4. Cobertura dos DG da escavadeira ─────────────────────────────────
    # Que fração dos eventos pré-DG da escavadeira tem seu alarme em cada conjunto?
    def _coverage(alarm_ids: list[int]) -> float:
        if not alarm_ids:
            return 0.0
        ids = ",".join(str(a) for a in alarm_ids)
        r = duckdb.query(f"""
            SELECT
              AVG(CASE WHEN Id_Alarme IN ({ids}) THEN 1.0 ELSE 0.0 END) AS coverage
            FROM read_parquet('{SILVER_GLOB}')
            WHERE Tag_Frota = '{ESCAVADEIRA}' AND {TARGET} = TRUE
        """).df()
        return float(r["coverage"].iloc[0])

    cov_global = _coverage(sorted(global_top30))
    cov_esc_lift = _coverage(esc_top_lift)
    print("\n== Cobertura dos eventos pré-DG da escavadeira ==")
    print(f"  Top-30 global (frequência): {cov_global:.1%}")
    print(f"  Top-30 escavadeira (lift):  {cov_esc_lift:.1%}")

    # ── 5. Diagnóstico do teto de lift ─────────────────────────────────────
    max_lift = float(esc.lift.max())
    max_rate = float(esc.taxa_pre_dg.max())
    # quantos alarmes têm lift "forte" (>= 5x)
    n_strong = int((esc.lift >= 5).sum())
    print(f"\n== Diagnóstico ==")
    print(f"  Lift máximo na escavadeira: {max_lift:.1f}× (taxa pré-DG {max_rate:.1%} vs base {br_esc:.3%})")
    print(f"  Alarmes com lift >= 5×: {n_strong}")
    print(f"  Lift máximo nos caminhões: {float(cam.lift.max()):.1f}×")

    # ── Figuras ────────────────────────────────────────────────────────────
    # Fig A: top-15 lift escavadeira
    top = esc.head(15).iloc[::-1]
    labels = [f"{int(a)}" for a in top.Id_Alarme]
    fig, ax = plt.subplots(figsize=(10, 7))
    inside = [aid in global_top30 for aid in top.Id_Alarme]
    colors = ["#10B981" if i else "#EF4444" for i in inside]
    ax.barh(range(len(top)), top.lift, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Lift preditivo (taxa pré-DG / base {br_esc:.3%})")
    ax.set_ylabel("Id_Alarme")
    ax.set_title("Fingerprint da Escavadeira por LIFT Preditivo — Top 15\n"
                 "Verde = já no top-30 global (frequência) | Vermelho = ausente do modelo atual")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6, label="Lift = 1 (sem sinal)")
    ax.axvline(5.0, color="black", linestyle=":", alpha=0.5, label="Lift = 5×")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "escavadeira_fingerprint_lift.png", dpi=150)
    plt.close()
    print(f"\n  ✓ {FIGURES_DIR / 'escavadeira_fingerprint_lift.png'}")

    # Fig B: escavadeira vs caminhão — lift máximo e cobertura
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    a1.bar(["Caminhões", "Escavadeira"], [float(cam.lift.max()), max_lift],
           color=["#8B5CF6", "#EF4444"])
    a1.set_ylabel("Lift máximo de alarme")
    a1.set_title("Força do melhor sinal de alarme\n(quanto o melhor alarme eleva a prob. de DG)")
    for i, v in enumerate([float(cam.lift.max()), max_lift]):
        a1.text(i, v, f"{v:.1f}×", ha="center", va="bottom", fontweight="bold")
    a2.bar(["Top-30 global\n(frequência)", "Top-30 escavadeira\n(lift)"],
           [cov_global * 100, cov_esc_lift * 100], color=["#9CA3AF", "#10B981"])
    a2.set_ylabel("Cobertura dos eventos pré-DG da escavadeira (%)")
    a2.set_title("Cobertura do fingerprint sobre DG da escavadeira")
    for i, v in enumerate([cov_global * 100, cov_esc_lift * 100]):
        a2.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "escavadeira_vs_caminhao_fingerprint.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'escavadeira_vs_caminhao_fingerprint.png'}")

    # ── Persistência ─────────────────────────────────────────────────────────
    def _rows(df, n):
        out = []
        for _, r in df.head(n).iterrows():
            out.append({
                "Id_Alarme": int(r.Id_Alarme),
                "Alarme": str(r.Alarme),
                "suporte": int(r.suporte),
                "n_pre_dg": int(r.n_pre_dg),
                "taxa_pre_dg": float(r.taxa_pre_dg),
                "lift": float(r.lift),
                "no_top30_global": bool(int(r.Id_Alarme) in global_top30),
            })
        return out

    payload = {
        "base_rate_escavadeira": br_esc,
        "base_rate_caminhao": br_cam,
        "min_support": MIN_SUPPORT,
        "escavadeira_top_alarms_by_lift": _rows(esc, 30),
        "caminhao_top_alarms_by_lift": _rows(cam, 15),
        "global_top30_freq_ids": sorted(global_top30),
        "overlap_count_esc_lift_vs_global_freq": len(overlap),
        "coverage_pre_dg_escavadeira": {
            "global_top30_freq": cov_global,
            "escavadeira_top30_lift": cov_esc_lift,
        },
        "diagnostics": {
            "max_lift_escavadeira": max_lift,
            "max_rate_escavadeira": max_rate,
            "n_alarms_lift_ge_5_escavadeira": n_strong,
            "max_lift_caminhao": float(cam.lift.max()),
        },
    }
    out_json = GOLD_DIR / "escavadeira_fingerprint.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ Resultados salvos em {out_json}")
    print("\n══════════════ SPRINT 11 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
