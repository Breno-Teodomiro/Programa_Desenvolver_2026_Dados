"""Visualizações interativas — funções reutilizáveis e dashboards HTML standalone.

As funções `dash_*` geram HTMLs completos em outputs/dashboards/ sem servidor.
As funções `plot_*` retornam figuras Plotly para uso em notebooks.
"""

from pathlib import Path
import pickle

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import shap

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
DASH_DIR = ROOT / "outputs" / "dashboards"
MODEL_PATH = GOLD_DIR / "lgbm_dontgo.pkl"

VALE_GREEN = "#00A650"
VALE_DARK = "#003D1F"
MONTHS_ORDER = ["jan", "feb", "mar", "abr", "may", "jun"]
MONTH_LABELS = {
    "jan": "Jan", "feb": "Fev", "mar": "Mar",
    "abr": "Abr", "may": "Mai", "jun": "Jun",
}

_PLOTLY_CONFIG = {"displayModeBar": True, "responsive": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_html(fig: go.Figure, name: str, title: str) -> Path:
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    path = DASH_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn",
                   config=_PLOTLY_CONFIG, full_html=True)
    print(f"  ✔ {title} → {path.name}")
    return path


def _save_png(fig: go.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.html"
    fig.write_html(str(path))
    return path


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARDS HTML STANDALONE
# ══════════════════════════════════════════════════════════════════════════════

def dash_fleet_risk() -> Path:
    """Dashboard 1: Ranking de risco Don't Go por equipamento."""
    df = duckdb.execute(f"""
        SELECT
            TAG,
            MAX(Tag_Frota) AS frota,
            MAX(Tipo)      AS tipo,
            COUNT(*)       AS total_eventos,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS total_dg,
            ROUND(AVG(CAST(Is_Dont_Go AS FLOAT)) * 100, 4) AS taxa_dg_pct,
            SUM(CAST(is_dont_go_next_60m AS INTEGER)) AS pre_dg_60m
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY TAG
        ORDER BY taxa_dg_pct DESC
    """).df()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Taxa de Don't Go por Equipamento (%)",
                        "Volume de Alarmes vs. Eventos Don't Go"),
        column_widths=[0.55, 0.45],
    )

    colors = [
        "#D32F2F" if v > 1 else "#F57C00" if v > 0.05 else VALE_GREEN
        for v in df["taxa_dg_pct"]
    ]
    fig.add_trace(go.Bar(
        x=df["taxa_dg_pct"], y=df["TAG"],
        orientation="h",
        marker_color=colors,
        text=df["taxa_dg_pct"].apply(lambda v: f"{v:.3f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Taxa DG: %{x:.4f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["total_eventos"], y=df["total_dg"],
        mode="markers+text",
        text=df["TAG"],
        textposition="top center",
        marker=dict(
            size=df["taxa_dg_pct"] * 300 + 8,
            color=df["taxa_dg_pct"],
            colorscale=[[0, VALE_GREEN], [0.5, "#FFA000"], [1, "#D32F2F"]],
            showscale=True,
            colorbar=dict(title="Taxa DG (%)", x=1.02),
        ),
        hovertemplate="<b>%{text}</b><br>Eventos: %{x:,}<br>DG: %{y:,}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Ranking de Risco Don't Go — Frota Completa (Jan–Jun 2025)</b>",
            font=dict(size=18, color=VALE_DARK),
        ),
        height=500, showlegend=False,
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="Taxa DG (%)", row=1, col=1)
    fig.update_yaxes(title_text="Equipamento", row=1, col=1, autorange="reversed")
    fig.update_xaxes(title_text="Total de Eventos", row=1, col=2)
    fig.update_yaxes(title_text="Eventos Don't Go", row=1, col=2)

    return _save_html(fig, "01_fleet_risk", "Ranking de risco por equipamento")


def dash_temporal_patterns() -> Path:
    """Dashboard 2: Padrões temporais de Don't Go (mensal, hora×dia, top TAGs)."""
    monthly = duckdb.execute(f"""
        SELECT mes, TAG,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS dg_count,
            COUNT(*) AS total
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY mes, TAG ORDER BY mes
    """).df()

    hourly = duckdb.execute(f"""
        SELECT hora_dia, dia_semana,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS dg_count
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY hora_dia, dia_semana ORDER BY dia_semana, hora_dia
    """).df()

    crit_data = duckdb.execute(f"""
        SELECT mes,
            SUM(CASE WHEN Id_Criticidade = 1 AND Is_Dont_Go = 1 THEN 1 ELSE 0 END) AS critico_dg,
            SUM(CASE WHEN Id_Criticidade != 1 AND Is_Dont_Go = 1 THEN 1 ELSE 0 END) AS outros_dg
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY mes ORDER BY mes
    """).df()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Eventos Don't Go por Mês (frota completa)",
            "Concentração por Hora e Dia da Semana",
            "Taxa DG (%) mensal — Top 8 equipamentos",
            "Don't Go: Críticos vs. Outros por Mês",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    # Tendência mensal total
    monthly_total = monthly.groupby("mes")["dg_count"].sum().reset_index()
    month_names = [MONTH_LABELS.get(MONTHS_ORDER[m - 1], str(m))
                   for m in monthly_total["mes"]]
    fig.add_trace(go.Scatter(
        x=month_names, y=monthly_total["dg_count"],
        mode="lines+markers",
        line=dict(color=VALE_GREEN, width=3),
        marker=dict(size=10, color=VALE_GREEN),
        fill="tozeroy", fillcolor="rgba(0,166,80,0.12)",
        hovertemplate="<b>%{x}</b>: %{y:,} eventos<extra></extra>",
    ), row=1, col=1)

    # Heatmap hora × dia
    days_map = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
    hourly["dia_nome"] = hourly["dia_semana"].map(days_map)
    day_order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    pivot = (
        hourly.pivot(index="dia_nome", columns="hora_dia", values="dg_count")
        .fillna(0)
        .reindex([d for d in day_order if d in hourly["dia_nome"].unique()])
    )
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}h" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="DG", x=1.02, len=0.45, y=0.78),
        hovertemplate="<b>%{y} %{x}</b>: %{z:.0f} DG<extra></extra>",
    ), row=1, col=2)

    # Top 8 equipamentos — taxa mensal
    top8 = (
        monthly.groupby("TAG")["dg_count"].sum()
        .sort_values(ascending=False).head(8).index.tolist()
    )
    monthly_top = monthly[monthly["TAG"].isin(top8)].copy()
    monthly_top["taxa"] = monthly_top["dg_count"] / monthly_top["total"] * 100
    monthly_top["mes_nome"] = monthly_top["mes"].apply(
        lambda m: MONTH_LABELS.get(MONTHS_ORDER[m - 1], str(m))
    )
    palette = px.colors.qualitative.Set2
    for i, tag in enumerate(top8):
        sub = monthly_top[monthly_top["TAG"] == tag].sort_values("mes")
        fig.add_trace(go.Scatter(
            x=sub["mes_nome"], y=sub["taxa"],
            mode="lines+markers", name=tag,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{tag}</b> %{{x}}: %{{y:.3f}}%<extra></extra>",
        ), row=2, col=1)

    # Criticidade mensal
    crit_data["mes_nome"] = crit_data["mes"].apply(
        lambda m: MONTH_LABELS.get(MONTHS_ORDER[m - 1], str(m))
    )
    fig.add_trace(go.Bar(
        x=crit_data["mes_nome"], y=crit_data["critico_dg"],
        name="Crítico", marker_color="#D32F2F",
        hovertemplate="<b>%{x}</b> Críticos: %{y:,}<extra></extra>",
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=crit_data["mes_nome"], y=crit_data["outros_dg"],
        name="Outros", marker_color="#90CAF9",
        hovertemplate="<b>%{x}</b> Outros: %{y:,}<extra></extra>",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Padrões Temporais de Don't Go — Jan a Jun 2025</b>",
            font=dict(size=18, color=VALE_DARK),
        ),
        height=700, barmode="stack",
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.08),
    )

    return _save_html(fig, "02_temporal_patterns", "Padrões temporais")


def dash_alarm_fingerprint() -> Path:
    """Dashboard 3: Alarm Fingerprint — DNA dos alarmes que precedem Don't Go."""
    sample = duckdb.execute(f"""
        SELECT * FROM read_parquet('{GOLD_DIR}/gold_jan.parquet') LIMIT 1
    """).df()
    fp_cols_list = [c for c in sample.columns if c.startswith("fp_alarm_")]

    rows = []
    for col in fp_cols_list:
        alarm_id = col.replace("fp_alarm_", "")
        result = duckdb.execute(f"""
            SELECT
                '{alarm_id}' AS alarm_id,
                SUM(CAST({col} AS INTEGER)) AS n_ativo,
                SUM(CASE WHEN {col} = 1 AND is_dont_go_next_60m = true THEN 1 ELSE 0 END) AS n_dg_quando_ativo,
                ROUND(
                    100.0 * SUM(CASE WHEN {col} = 1 AND is_dont_go_next_60m = true THEN 1 ELSE 0 END)
                    / NULLIF(SUM(CAST({col} AS INTEGER)), 0), 3
                ) AS taxa_dg_pct
            FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        """).df()
        rows.append(result.iloc[0])

    fp_df = pd.DataFrame(rows).sort_values("taxa_dg_pct", ascending=False)

    alarm_names = duckdb.execute(f"""
        SELECT CAST(Id_Alarme AS VARCHAR) AS alarm_id, MAX(Alarme) AS nome
        FROM read_parquet('{GOLD_DIR}/gold_jan.parquet')
        GROUP BY Id_Alarme
    """).df()
    alarm_names["alarm_id"] = alarm_names["alarm_id"].astype(str)
    fp_df = fp_df.merge(alarm_names, on="alarm_id", how="left")
    fp_df["label"] = fp_df.apply(
        lambda r: (str(r["nome"])[:35] + "…") if pd.notna(r.get("nome")) and len(str(r.get("nome", ""))) > 35
        else str(r.get("nome", f"Alarme {r['alarm_id'][:8]}")),
        axis=1,
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Taxa DG (%) quando alarme ativo na janela 4h",
            "Frequência de ativação do fingerprint",
        ),
        column_widths=[0.5, 0.5],
    )

    top15 = fp_df.head(15).sort_values("taxa_dg_pct")
    fig.add_trace(go.Bar(
        x=top15["taxa_dg_pct"], y=top15["label"],
        orientation="h",
        marker=dict(color=top15["taxa_dg_pct"],
                    colorscale=[[0, "#FFF9C4"], [1, "#D32F2F"]]),
        text=top15["taxa_dg_pct"].apply(lambda v: f"{v:.2f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Taxa DG: %{x:.3f}%<extra></extra>",
    ), row=1, col=1)

    top15_freq = fp_df.sort_values("n_ativo", ascending=False).head(15).sort_values("n_ativo")
    fig.add_trace(go.Bar(
        x=top15_freq["n_ativo"], y=top15_freq["label"],
        orientation="h",
        marker=dict(color=top15_freq["n_ativo"],
                    colorscale=[[0, "#E8F5E9"], [1, VALE_GREEN]]),
        text=top15_freq["n_ativo"].apply(lambda v: f"{int(v):,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Ativações: %{x:,}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Alarm Fingerprint — DNA dos Alarmes que Precedem Don't Go</b>",
            font=dict(size=18, color=VALE_DARK),
        ),
        height=520, showlegend=False,
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="Taxa DG quando alarme ativo (%)", row=1, col=1)
    fig.update_xaxes(title_text="Nº de ativações (todos os meses)", row=1, col=2)

    return _save_html(fig, "03_alarm_fingerprint", "Alarm Fingerprint")


def dash_risk_timeline(tag: str = "CA65926", mes: str = "jun") -> Path:
    """Dashboard 4: Timeline do score de risco com marcação de eventos reais."""
    model = _load_model()
    feature_cols = model.feature_name_

    gold_file = str(GOLD_DIR / f"gold_{mes}.parquet")
    df = duckdb.execute(f"""
        SELECT * FROM read_parquet('{gold_file}')
        WHERE TAG = '{tag}'
        ORDER BY Data_Evento
    """).df()

    if df.empty:
        print(f"  [AVISO] Sem dados para {tag} em {mes}")
        return DASH_DIR / f"04_risk_timeline_{tag}_{mes}.html"

    available = [c for c in feature_cols if c in df.columns]
    proba = model.predict_proba(df[available].fillna(0))[:, 1]
    dates = pd.to_datetime(df["Data_Evento"])
    dg_mask = df["Is_Dont_Go"].astype(bool).values
    pre_dg_mask = df["is_dont_go_next_60m"].fillna(False).astype(bool).values

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"Score de Risco Don't Go — {tag} ({MONTH_LABELS.get(mes, mes)}/2025)",
            "Alarmes Críticos acumulados (janela 60 min)",
            "Aceleração de alarmes críticos",
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
    )

    fig.add_trace(go.Scatter(
        x=dates, y=proba * 100,
        mode="lines", name="Score de Risco",
        line=dict(color=VALE_GREEN, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,166,80,0.08)",
        hovertemplate="%{x|%d/%m %H:%M}: %{y:.1f}%<extra>Score</extra>",
    ), row=1, col=1)

    if pre_dg_mask.any():
        fig.add_trace(go.Scatter(
            x=dates[pre_dg_mask], y=proba[pre_dg_mask] * 100,
            mode="markers", name="Janela pré-DG (60m)",
            marker=dict(color="#FFA000", size=5, opacity=0.7),
            hovertemplate="%{x|%d/%m %H:%M}: pré-DG<extra></extra>",
        ), row=1, col=1)

    if dg_mask.any():
        fig.add_trace(go.Scatter(
            x=dates[dg_mask], y=proba[dg_mask] * 100,
            mode="markers", name="Don't Go real",
            marker=dict(color="#D32F2F", size=9, symbol="x",
                        line=dict(width=2, color="#D32F2F")),
            hovertemplate="%{x|%d/%m %H:%M}: <b>DON'T GO</b><extra></extra>",
        ), row=1, col=1)

    fig.add_hline(y=50, line_dash="dash", line_color="#D32F2F",
                  line_width=1.5, row=1, col=1,
                  annotation_text="Limiar 50%", annotation_position="top right")

    if "n_criticos_60m" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["n_criticos_60m"],
            mode="lines", name="Críticos 60m",
            line=dict(color="#EF9A9A", width=1.5),
            fill="tozeroy", fillcolor="rgba(211,47,47,0.07)",
            hovertemplate="%{x|%d/%m %H:%M}: %{y} críticos<extra></extra>",
        ), row=2, col=1)

    if "aceleracao_criticos" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["aceleracao_criticos"].clip(upper=20),
            mode="lines", name="Aceleração críticos",
            line=dict(color="#7B1FA2", width=1.5),
            hovertemplate="%{x|%d/%m %H:%M}: acc=%{y:.1f}<extra></extra>",
        ), row=3, col=1)

    fig.update_layout(
        height=680,
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Risco (%)", row=1, col=1)
    fig.update_yaxes(title_text="N críticos/60m", row=2, col=1)
    fig.update_yaxes(title_text="Aceleração", row=3, col=1)
    fig.update_xaxes(title_text="Data/Hora", row=3, col=1)

    return _save_html(fig, f"04_risk_timeline_{tag}_{mes}",
                      f"Timeline de risco {tag}")


def dash_shap_global(mes: str = "jun", n_samples: int = 500) -> Path:
    """Dashboard 5: SHAP global — importância explicável das features."""
    model = _load_model()
    feature_cols = model.feature_name_

    gold_file = str(GOLD_DIR / f"gold_{mes}.parquet")
    df = duckdb.execute(f"""
        SELECT * FROM read_parquet('{gold_file}')
        ORDER BY RANDOM()
        LIMIT {n_samples}
    """).df()

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    mean_abs = np.abs(sv).mean(axis=0)
    mean_signed = sv.mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": available,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=True).tail(20)

    def categorize(feat: str) -> str:
        if feat.startswith("fp_alarm_"):      return "Alarm Fingerprint"
        if feat.startswith("n_") or feat.startswith("aceleracao"): return "Frequência"
        if feat == "min_desde_ultimo_dg":     return "Recência"
        if feat in ("hora_dia", "dia_semana", "mes", "posicao_turno", "is_turno_noturno"):
            return "Contexto Temporal"
        if feat in ("frota_encoded", "is_em_operacao", "is_em_manutencao", "sem_apontamento"):
            return "Contexto Operacional"
        return "Outros"

    cat_colors = {
        "Alarm Fingerprint":   "#D32F2F",
        "Frequência":          "#F57C00",
        "Recência":            "#7B1FA2",
        "Contexto Temporal":   "#1976D2",
        "Contexto Operacional": VALE_GREEN,
        "Outros":              "#9E9E9E",
    }
    shap_df["categoria"] = shap_df["feature"].apply(categorize)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Importância Global (|SHAP| médio) — Top 20",
            "Direção do Impacto (SHAP médio com sinal)",
        ),
        column_widths=[0.5, 0.5],
    )

    fig.add_trace(go.Bar(
        x=shap_df["mean_abs_shap"], y=shap_df["feature"],
        orientation="h",
        marker_color=[cat_colors[c] for c in shap_df["categoria"]],
        text=shap_df["mean_abs_shap"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
        customdata=shap_df["categoria"],
        hovertemplate="<b>%{y}</b><br>|SHAP|: %{x:.5f}<br>%{customdata}<extra></extra>",
    ), row=1, col=1)

    shap_signed = shap_df.sort_values("mean_shap")
    fig.add_trace(go.Bar(
        x=shap_signed["mean_shap"], y=shap_signed["feature"],
        orientation="h",
        marker_color=["#D32F2F" if v > 0 else VALE_GREEN for v in shap_signed["mean_shap"]],
        text=shap_signed["mean_shap"].apply(lambda v: f"{v:+.4f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.5f}<extra></extra>",
    ), row=1, col=2)

    legend_html = " | ".join([
        f'<span style="color:{c}">■</span> {cat}'
        for cat, c in cat_colors.items()
        if cat in shap_df["categoria"].values
    ])

    fig.update_layout(
        title=dict(
            text=(f"<b>Explicabilidade SHAP — Modelo LightGBM Don't Go</b><br>"
                  f"<sup>{n_samples} amostras aleatórias de {MONTH_LABELS.get(mes, mes)}/2025</sup>"),
            font=dict(size=17, color=VALE_DARK),
        ),
        height=560, showlegend=False,
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        annotations=[dict(
            x=0.5, y=-0.12, xref="paper", yref="paper",
            text=legend_html, showarrow=False,
            font=dict(size=11), xanchor="center",
        )],
    )
    fig.update_xaxes(title_text="|SHAP| médio", row=1, col=1)
    fig.update_xaxes(title_text="SHAP médio (+ aumenta risco)", row=1, col=2)

    return _save_html(fig, "05_shap_global", "SHAP global")


def dash_critical_equipment(tag: str = "CA65926") -> Path:
    """Dashboard 6: Análise detalhada do equipamento mais crítico."""
    monthly_cmp = duckdb.execute(f"""
        SELECT mes,
            SUM(CASE WHEN TAG = '{tag}' THEN CAST(Is_Dont_Go AS INTEGER) ELSE 0 END) AS dg_tag,
            ROUND(100.0 * SUM(CASE WHEN TAG = '{tag}' THEN CAST(Is_Dont_Go AS INTEGER) ELSE 0 END)
                / NULLIF(SUM(CASE WHEN TAG = '{tag}' THEN 1 ELSE 0 END), 0), 3) AS taxa_tag,
            ROUND(100.0 * SUM(CASE WHEN TAG != '{tag}' THEN CAST(Is_Dont_Go AS INTEGER) ELSE 0 END)
                / NULLIF(SUM(CASE WHEN TAG != '{tag}' THEN 1 ELSE 0 END), 0), 3) AS taxa_outros
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY mes ORDER BY mes
    """).df()
    monthly_cmp["mes_nome"] = monthly_cmp["mes"].apply(
        lambda m: MONTH_LABELS.get(MONTHS_ORDER[m - 1], str(m))
    )

    top_alarms = duckdb.execute(f"""
        SELECT Id_Alarme, MAX(Alarme) AS alarme,
            COUNT(*) AS freq,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS n_dg,
            ROUND(100.0 * SUM(CAST(Is_Dont_Go AS INTEGER)) / COUNT(*), 2) AS taxa_dg_pct
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        WHERE TAG = '{tag}'
        GROUP BY Id_Alarme
        ORDER BY n_dg DESC LIMIT 15
    """).df()
    top_alarms["label"] = top_alarms["alarme"].apply(
        lambda n: (str(n)[:30] + "…") if len(str(n)) > 30 else str(n)
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Taxa DG (%): {tag} vs. restante da frota",
            f"Volume mensal de Don't Go — {tag}",
            f"Top 15 alarmes por nº de Don't Go",
            f"Taxa DG (%) por alarme",
        ),
        vertical_spacing=0.16, horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(
        x=monthly_cmp["mes_nome"], y=monthly_cmp["taxa_tag"],
        mode="lines+markers", name=tag,
        line=dict(color="#D32F2F", width=3), marker=dict(size=9),
        hovertemplate=f"<b>{tag}</b> %{{x}}: %{{y:.3f}}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=monthly_cmp["mes_nome"], y=monthly_cmp["taxa_outros"],
        mode="lines+markers", name="Restante da frota",
        line=dict(color=VALE_GREEN, width=2, dash="dot"), marker=dict(size=7),
        hovertemplate="<b>Outros</b> %{x}: %{y:.3f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=monthly_cmp["mes_nome"], y=monthly_cmp["dg_tag"],
        marker_color="#EF9A9A",
        text=monthly_cmp["dg_tag"], textposition="outside",
        showlegend=False,
        hovertemplate="%{x}: %{y} DG<extra></extra>",
    ), row=1, col=2)

    top_sorted = top_alarms.sort_values("n_dg")
    fig.add_trace(go.Bar(
        x=top_sorted["n_dg"], y=top_sorted["label"],
        orientation="h",
        marker=dict(color=top_sorted["n_dg"],
                    colorscale=[[0, "#FFCCBC"], [1, "#BF360C"]]),
        text=top_sorted["n_dg"], textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>Don't Go: %{x:,}<extra></extra>",
    ), row=2, col=1)

    top_taxa = top_alarms.sort_values("taxa_dg_pct")
    fig.add_trace(go.Bar(
        x=top_taxa["taxa_dg_pct"], y=top_taxa["label"],
        orientation="h",
        marker=dict(color=top_taxa["taxa_dg_pct"],
                    colorscale=[[0, "#FFF9C4"], [1, "#D32F2F"]]),
        text=top_taxa["taxa_dg_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>Taxa DG: %{x:.2f}%<extra></extra>",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>Análise Aprofundada — Equipamento Crítico: {tag}</b>",
            font=dict(size=18, color=VALE_DARK),
        ),
        height=720,
        plot_bgcolor="#FAFAFA", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.04),
    )
    fig.update_yaxes(title_text="Taxa DG (%)", row=1, col=1)
    fig.update_xaxes(title_text="Nº Don't Go", row=2, col=1)
    fig.update_xaxes(title_text="Taxa DG (%)", row=2, col=2)

    return _save_html(fig, f"06_critical_{tag}", f"Análise {tag}")


def generate_all_dashboards() -> list[Path]:
    """Gera todos os 6 dashboards HTML standalone."""
    print("\n=== Gerando dashboards HTML standalone ===\n")
    paths = []
    steps = [
        ("[1/6] Ranking de risco por equipamento...",   dash_fleet_risk),
        ("[2/6] Padrões temporais...",                   dash_temporal_patterns),
        ("[3/6] Alarm Fingerprint...",                   dash_alarm_fingerprint),
        ("[4/6] Timeline de risco — CA65926 (Jun)...",  lambda: dash_risk_timeline("CA65926", "jun")),
        ("[5/6] SHAP global...",                         lambda: dash_shap_global("jun", 500)),
        ("[6/6] Análise equipamento crítico CA65926...", lambda: dash_critical_equipment("CA65926")),
    ]
    for msg, fn in steps:
        print(msg)
        paths.append(fn())

    print(f"\n=== {len(paths)} dashboards gerados em outputs/dashboards/ ===")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES REUTILIZÁVEIS PARA NOTEBOOKS (plot_*)
# ══════════════════════════════════════════════════════════════════════════════

def plot_alarm_timeline(
    df: pl.DataFrame,
    equipment_tag: str,
    save: bool = True,
) -> go.Figure:
    """Timeline interativa de alarmes de um equipamento com marcação de Don't Go."""
    sub = df.filter(pl.col("TAG") == equipment_tag).sort("Data_Evento")

    fig = go.Figure()
    non_dg = sub.filter(pl.col("Is_Dont_Go") == 0)
    if len(non_dg) > 0:
        fig.add_trace(go.Scatter(
            x=non_dg["Data_Evento"].to_list(),
            y=non_dg["Id_Criticidade"].to_list(),
            mode="markers", name="Alarme",
            marker=dict(color=non_dg["Id_Criticidade"].to_list(),
                        colorscale="RdYlGn_r", size=5, opacity=0.6),
            text=non_dg["Alarme"].to_list(),
            hovertemplate="%{text}<br>%{x}<extra></extra>",
        ))

    dg = sub.filter(pl.col("Is_Dont_Go") == 1)
    if len(dg) > 0:
        fig.add_trace(go.Scatter(
            x=dg["Data_Evento"].to_list(),
            y=dg["Id_Criticidade"].to_list(),
            mode="markers", name="Don't Go",
            marker=dict(color="red", size=10, symbol="x"),
            hovertemplate="%{x}<extra>Don't Go</extra>",
        ))

    fig.update_layout(
        title=f"Timeline de Alarmes — {equipment_tag}",
        xaxis_title="Timestamp", yaxis_title="Criticidade",
        yaxis=dict(tickvals=[1, 2, 3, 4],
                   ticktext=["Crítico", "Não-Crítico", "Info", "Outro"]),
        height=450,
    )
    if save:
        _save_png(fig, f"timeline_{equipment_tag}")
    return fig


def plot_alarm_heatmap(
    df: pl.DataFrame,
    top_n_alarms: int = 30,
    top_n_tags: int = 20,
    save: bool = True,
) -> go.Figure:
    """Heatmap equipamento × tipo de alarme × frequência (log scale)."""
    top_alarms = (
        df.group_by("Id_Alarme").agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True).head(top_n_alarms)["Id_Alarme"].to_list()
    )
    top_tags = (
        df.group_by("TAG").agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True).head(top_n_tags)["TAG"].to_list()
    )

    heat = (
        df.filter(pl.col("Id_Alarme").is_in(top_alarms) & pl.col("TAG").is_in(top_tags))
        .group_by(["TAG", "Id_Alarme"]).agg(pl.len().alias("cnt"))
        .pivot(on="Id_Alarme", index="TAG", values="cnt", aggregate_function="sum")
        .fill_null(0)
    )

    tags = heat["TAG"].to_list()
    alarm_cols = [c for c in heat.columns if c != "TAG"]
    z = heat.select(alarm_cols).to_numpy()

    fig = go.Figure(go.Heatmap(
        x=[str(a) for a in alarm_cols], y=tags,
        z=np.log1p(z), colorscale="YlOrRd",
        hovertemplate="TAG: %{y}<br>Alarme: %{x}<br>Count: %{customdata}<extra></extra>",
        customdata=z,
    ))
    fig.update_layout(
        title=f"Frequência de Alarmes por Equipamento (top {top_n_alarms})",
        xaxis_title="Id_Alarme", yaxis_title="Equipamento (TAG)", height=600,
    )
    if save:
        _save_png(fig, "heatmap_alarm_equipment")
    return fig


def plot_feature_importance_shap(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
) -> go.Figure:
    """Bar chart de importância de features por mean(|SHAP|)."""
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_sample.columns,
    ).nlargest(top_n).sort_values()

    fig = go.Figure(go.Bar(
        x=mean_abs.values, y=mean_abs.index.tolist(),
        orientation="h", marker_color="steelblue",
    ))
    fig.update_layout(
        title=f"Top {top_n} Features — Importância SHAP",
        xaxis_title="Mean |SHAP value|", yaxis_title="Feature",
        height=max(400, top_n * 25),
    )
    if save:
        _save_png(fig, "shap_importance")
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    save: bool = True,
) -> go.Figure:
    """Matriz de confusão interativa."""
    from sklearn.metrics import confusion_matrix as cm_fn
    if labels is None:
        labels = ["Sem DG (0)", "Com DG (1)"]
    cm = cm_fn(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    text = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
    ))
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito", yaxis_title="Real",
        height=400, width=450,
    )
    if save:
        _save_png(fig, "confusion_matrix")
    return fig


def plot_precision_recall_curve(
    model,
    X_test: pd.DataFrame,
    y_test,
    save: bool = True,
) -> go.Figure:
    """Curva Precision-Recall com marcação do threshold ótimo."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode="lines",
        name=f"PR Curve (AUC={pr_auc:.3f})",
        line=dict(color="darkorange", width=2),
    ))
    fig.update_layout(
        title="Curva Precision-Recall",
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]),
        height=450,
    )
    if save:
        _save_png(fig, "precision_recall_curve")
    return fig


if __name__ == "__main__":
    generate_all_dashboards()
