"""Don't Go Predictor — Vale Programa Desenvolver 2026.

App Streamlit para demonstração do modelo preditivo de eventos Don't Go
em equipamentos de mineração pesada.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import shap
import streamlit as st
import pickle

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
GOLD_DIR = ROOT / "outputs" / "gold"
MODEL_PATH = GOLD_DIR / "lgbm_dontgo.pkl"
FIGURES_DIR = ROOT / "outputs" / "figures"

VALE_GREEN = "#00A650"
MONTHS_ORDER = ["jan", "feb", "mar", "abr", "may", "jun"]
MONTH_LABELS = {
    "jan": "Janeiro", "feb": "Fevereiro", "mar": "Março",
    "abr": "Abril", "may": "Maio", "jun": "Junho",
}

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Don't Go Predictor — Vale",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem; font-weight: 700; color: #00A650;
        border-bottom: 3px solid #00A650; padding-bottom: .4rem; margin-bottom: 1rem;
    }
    .kpi-label { font-size: .8rem; color: #666; margin-bottom: 0; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #262730; margin-top: 0; }
    .risk-alto  { color: #D32F2F; font-weight: 700; }
    .risk-medio { color: #F57C00; font-weight: 700; }
    .risk-baixo { color: #00A650; font-weight: 700; }
    .info-box {
        background: #E8F5E9; border-left: 4px solid #00A650;
        padding: .8rem 1rem; border-radius: 4px; margin: .5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Carregamento de dados (cacheados) ─────────────────────────────────────────

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_fleet_overview() -> pl.DataFrame:
    return duckdb.execute(f"""
        SELECT
            TAG,
            MAX(Tag_Frota)  AS frota,
            MAX(Tipo)       AS tipo,
            COUNT(*)        AS total_eventos,
            SUM(CAST(Is_Dont_Go AS INTEGER))       AS total_dg,
            ROUND(AVG(CAST(Is_Dont_Go AS FLOAT)) * 100, 4) AS taxa_dg_pct,
            SUM(CAST(is_dont_go_next_60m AS INTEGER)) AS eventos_pre_dg
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY TAG
        ORDER BY taxa_dg_pct DESC
    """).pl()


@st.cache_data(ttl=3600)
def load_monthly_trend() -> pl.DataFrame:
    return duckdb.execute(f"""
        SELECT
            mes, TAG,
            COUNT(*) AS eventos,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS dg_count
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY mes, TAG
        ORDER BY mes, TAG
    """).pl()


@st.cache_data(ttl=3600)
def load_hourly_heatmap() -> pd.DataFrame:
    return duckdb.execute(f"""
        SELECT
            hora_dia, dia_semana,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS dg_count
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        GROUP BY hora_dia, dia_semana
        ORDER BY dia_semana, hora_dia
    """).df()


@st.cache_data(ttl=3600)
def load_equipment_month(tag: str, mes: str) -> pd.DataFrame:
    gold_file = str(GOLD_DIR / f"gold_{mes}.parquet")
    return duckdb.execute(f"""
        SELECT * FROM read_parquet('{gold_file}')
        WHERE TAG = $1
        ORDER BY Data_Evento
    """, [tag]).df()


@st.cache_data(ttl=3600)
def load_alarm_top(tag: str) -> pd.DataFrame:
    return duckdb.execute(f"""
        SELECT
            Id_Alarme, MAX(Alarme) AS alarme_nome,
            COUNT(*) AS freq,
            SUM(CAST(Is_Dont_Go AS INTEGER)) AS n_dg
        FROM read_parquet('{GOLD_DIR}/gold_*.parquet')
        WHERE TAG = $1
        GROUP BY Id_Alarme
        ORDER BY freq DESC
        LIMIT 15
    """, [tag]).df()


# ── Funções auxiliares ────────────────────────────────────────────────────────

def risk_label_color(score: float) -> tuple[str, str]:
    if score >= 0.5:
        return "Alto", "#D32F2F"
    if score >= 0.2:
        return "Médio", "#F57C00"
    return "Baixo", VALE_GREEN


def predict_risk(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].fillna(0)
    return model.predict_proba(X)[:, 1]


def gauge_figure(score: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        title={"text": f"Nível de Risco: <b>{label}</b>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 20],  "color": "#E8F5E9"},
                {"range": [20, 50], "color": "#FFF3E0"},
                {"range": [50, 100],"color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "#D32F2F", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

model = load_model()
feature_cols: list[str] = model.feature_name_

with st.sidebar:
    st.markdown("## 🚛 Don't Go Predictor")
    st.markdown("**Programa Desenvolver 2026 — Vale**")
    st.caption("Modelo preditivo de eventos Don't Go em equipamentos de mineração.")
    st.divider()

    fleet_df = load_fleet_overview()
    all_tags = fleet_df["TAG"].to_list()

    default_idx = all_tags.index("CA65926") if "CA65926" in all_tags else 0
    selected_tag = st.selectbox("Equipamento (TAG)", all_tags, index=default_idx)
    selected_mes = st.selectbox(
        "Mês de análise",
        MONTHS_ORDER,
        index=4,
        format_func=lambda m: MONTH_LABELS[m],
    )

    st.divider()
    st.markdown("### Sobre o modelo")
    st.markdown(f"- **Algoritmo:** LightGBM\n- **Features:** {len(feature_cols)}\n"
                f"- **Target:** Don't Go em 60 min\n- **Treino:** Jan–Mai 2025\n"
                f"- **Teste:** Jun 2025 (temporal)")


# ── Abas principais ───────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📊 Visão da Frota",
    "🔍 Monitor por Equipamento",
    "🧠 Insights do Modelo",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Visão da Frota
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="main-header">Visão Geral da Frota — Jan a Jun 2025</div>',
                unsafe_allow_html=True)

    fleet_pd = fleet_df.to_pandas()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Equipamentos monitorados", len(fleet_pd))
    k2.metric("Total eventos Don't Go", f"{int(fleet_pd['total_dg'].sum()):,}")
    k3.metric("Taxa média de Don't Go", f"{fleet_pd['taxa_dg_pct'].mean():.3f}%")
    worst = fleet_pd.iloc[0]
    k4.metric("Equipamento crítico", worst["TAG"],
              delta=f"{worst['taxa_dg_pct']:.2f}% DG rate", delta_color="inverse")

    st.divider()

    col_bar, col_trend = st.columns(2)

    with col_bar:
        fig_bar = px.bar(
            fleet_pd.sort_values("taxa_dg_pct"),
            x="taxa_dg_pct", y="TAG",
            orientation="h",
            color="taxa_dg_pct",
            color_continuous_scale=["#C8E6C9", "#FFF176", "#EF9A9A"],
            labels={"taxa_dg_pct": "Taxa DG (%)", "TAG": "Equipamento"},
            title="Taxa de Don't Go por Equipamento (%)",
        )
        fig_bar.update_layout(height=420, showlegend=False,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_trend:
        monthly_df = load_monthly_trend()
        monthly_agg = (
            monthly_df
            .group_by("mes")
            .agg([pl.col("dg_count").sum().alias("total_dg")])
            .sort("mes")
            .to_pandas()
        )
        month_names = [MONTH_LABELS.get(MONTHS_ORDER[m - 1], str(m))
                       for m in monthly_agg["mes"]]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=month_names, y=monthly_agg["total_dg"],
            mode="lines+markers",
            line=dict(color=VALE_GREEN, width=3),
            marker=dict(size=9, color=VALE_GREEN),
            name="Don't Go",
            fill="tozeroy",
            fillcolor="rgba(0,166,80,0.1)",
        ))
        fig_trend.update_layout(
            title="Eventos Don't Go por Mês (toda a frota)",
            yaxis_title="Nº de eventos DG",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Heatmap hora × dia da semana
    st.markdown("#### Concentração de Don't Go — Hora × Dia da Semana")
    heat_df = load_hourly_heatmap()
    days_map = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
    heat_df["dia_nome"] = heat_df["dia_semana"].map(days_map)
    pivot = heat_df.pivot(index="dia_nome", columns="hora_dia", values="dg_count").fillna(0)
    day_order = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}h" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        colorbar_title="Eventos DG",
    ))
    fig_heat.update_layout(
        title="Alarmes Don't Go por Hora e Dia da Semana",
        xaxis_title="Hora do Dia",
        yaxis_title="Dia da Semana",
        height=320,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Tabela de frotas
    st.markdown("#### Resumo por Equipamento")
    display_fleet = fleet_pd.rename(columns={
        "TAG": "TAG", "frota": "Frota", "tipo": "Tipo",
        "total_eventos": "Total Eventos", "total_dg": "Eventos DG",
        "taxa_dg_pct": "Taxa DG (%)", "eventos_pre_dg": "Eventos pré-DG (60m)",
    })
    st.dataframe(display_fleet, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Monitor por Equipamento
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f'<div class="main-header">Monitor: {selected_tag} — {MONTH_LABELS[selected_mes]}</div>',
        unsafe_allow_html=True,
    )

    # Info do equipamento
    equip_row = fleet_df.filter(pl.col("TAG") == selected_tag)
    if len(equip_row) > 0:
        r = equip_row.row(0, named=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frota", r["frota"] or "—")
        c2.metric("Tipo", r["tipo"] or "—")
        c3.metric("Total eventos (6 meses)", f"{r['total_eventos']:,}")
        c4.metric("Taxa Don't Go", f"{r['taxa_dg_pct']:.3f}%")

    st.divider()

    with st.spinner(f"Carregando dados de {selected_tag}..."):
        df_eq = load_equipment_month(selected_tag, selected_mes)

    if df_eq.empty:
        st.warning(f"Nenhum dado encontrado para {selected_tag} em {MONTH_LABELS[selected_mes]}.")
        st.stop()

    # Verificar features disponíveis
    available_feats = [c for c in feature_cols if c in df_eq.columns]

    # Risco atual (últimos 100 eventos)
    tail_100 = df_eq.tail(100)
    proba_100 = predict_risk(model, tail_100, available_feats)
    risk_score = float(proba_100.max())
    risk_lbl, risk_color = risk_label_color(risk_score)

    col_gauge, col_alarm_bar = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(gauge_figure(risk_score, risk_lbl, risk_color),
                        use_container_width=True)
        st.caption(f"Score máximo nos últimos 100 eventos registrados em {MONTH_LABELS[selected_mes]}.")

    with col_alarm_bar:
        st.markdown(f"#### Alarmes por Dia — {MONTH_LABELS[selected_mes]}")
        df_eq["Data_Evento"] = pd.to_datetime(df_eq["Data_Evento"])
        df_daily = df_eq.groupby(df_eq["Data_Evento"].dt.date).agg(
            n_alarmes=("Id_Eventos_Telemetria", "count"),
            n_criticos=("Id_Criticidade", lambda x: (x == 1).sum()),
            n_dg=("Is_Dont_Go", "sum"),
        ).reset_index()
        df_daily.columns = ["data", "n_alarmes", "n_criticos", "n_dg"]

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=df_daily["data"], y=df_daily["n_alarmes"],
            name="Total alarmes", marker_color="#90CAF9", opacity=0.7,
        ))
        fig_daily.add_trace(go.Bar(
            x=df_daily["data"], y=df_daily["n_criticos"],
            name="Alarmes críticos", marker_color="#EF9A9A",
        ))
        fig_daily.add_trace(go.Scatter(
            x=df_daily["data"], y=df_daily["n_dg"],
            name="Don't Go", mode="markers",
            marker=dict(color="#D32F2F", size=10, symbol="x"),
            yaxis="y2",
        ))
        fig_daily.update_layout(
            barmode="overlay", height=260,
            yaxis=dict(title="Nº de alarmes"),
            yaxis2=dict(title="Don't Go", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    # Linha do tempo do score de risco
    st.markdown("#### Evolução do Score de Risco ao longo do mês")
    n_timeline = min(400, len(df_eq))
    df_tail = df_eq.tail(n_timeline)
    proba_timeline = predict_risk(model, df_tail, available_feats)
    dates_timeline = pd.to_datetime(df_tail["Data_Evento"].values)
    dg_mask = df_tail["Is_Dont_Go"].astype(bool).values

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=dates_timeline, y=proba_timeline * 100,
        mode="lines", name="Score de Risco",
        line=dict(color=VALE_GREEN, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,166,80,0.08)",
    ))
    if dg_mask.any():
        fig_risk.add_trace(go.Scatter(
            x=dates_timeline[dg_mask], y=proba_timeline[dg_mask] * 100,
            mode="markers", name="Don't Go real",
            marker=dict(color="#D32F2F", size=7, symbol="circle"),
        ))
    fig_risk.add_hline(y=50, line_dash="dash", line_color="#D32F2F",
                       annotation_text="Limiar 50%", annotation_position="top right")
    fig_risk.update_layout(
        yaxis_title="Score de Risco (%)", xaxis_title="Timestamp",
        height=300, legend=dict(orientation="h", y=1.1),
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    col_alarms, col_top = st.columns(2)

    with col_alarms:
        st.markdown("#### Últimos 20 eventos registrados")
        show_cols = ["Data_Evento", "Alarme", "Id_Criticidade", "Classe", "Is_Dont_Go"]
        show_cols = [c for c in show_cols if c in df_eq.columns]
        recent = df_eq[show_cols].tail(20).sort_values("Data_Evento", ascending=False)
        st.dataframe(recent, use_container_width=True, hide_index=True)

    with col_top:
        st.markdown("#### Top 15 alarmes mais frequentes (6 meses)")
        alarm_top = load_alarm_top(selected_tag)
        fig_atop = px.bar(
            alarm_top.sort_values("freq"),
            x="freq", y="alarme_nome",
            orientation="h",
            color="n_dg",
            color_continuous_scale=["#C8E6C9", "#D32F2F"],
            labels={"freq": "Frequência", "alarme_nome": "Alarme", "n_dg": "Nº DG"},
            title="Frequência × Ocorrências de Don't Go",
        )
        fig_atop.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_atop, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Insights do Modelo
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="main-header">Insights do Modelo Preditivo (LightGBM)</div>',
                unsafe_allow_html=True)

    col_imp, col_figs = st.columns(2)

    with col_imp:
        st.markdown("#### Importância das Features (gain)")
        imp_df = pd.DataFrame({
            "feature": model.feature_name_,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(20)

        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            color="importance",
            color_continuous_scale=["#C8E6C9", "#00A650"],
            title="Top 20 Features — LightGBM Importance",
        )
        fig_imp.update_layout(height=500, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_figs:
        figures_available = {f.stem: f for f in sorted(FIGURES_DIR.glob("*.png"))}
        fig_labels = {
            "pr_curve":        "Precision-Recall Curve",
            "roc_curve":       "ROC Curve",
            "confusion_matrix":"Matriz de Confusão",
            "shap_importance": "SHAP — Importância Global",
            "lgbm_importance": "LightGBM — Feature Importance",
            "timeline_CA65926":"Timeline — CA65926 (equipamento crítico)",
        }
        if figures_available:
            selected_fig_key = st.selectbox(
                "Selecionar gráfico salvo",
                list(figures_available.keys()),
                format_func=lambda k: fig_labels.get(k, k),
            )
            st.image(str(figures_available[selected_fig_key]), use_container_width=True)
        else:
            st.info("Nenhuma figura encontrada em outputs/figures/")

    # SHAP para equipamento selecionado
    st.divider()
    st.markdown(f"#### Explicação SHAP — {selected_tag} em {MONTH_LABELS[selected_mes]}")

    with st.spinner("Calculando SHAP values (últimos 200 eventos)..."):
        df_shap_src = load_equipment_month(selected_tag, selected_mes)

    if df_shap_src.empty:
        st.info(f"Sem dados de {selected_tag} em {MONTH_LABELS[selected_mes]} para SHAP.")
    else:
        available_feats_shap = [c for c in feature_cols if c in df_shap_src.columns]
        X_shap = df_shap_src[available_feats_shap].tail(200).fillna(0)

        with st.spinner("Processando SHAP..."):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_shap)

        # LightGBM binary → shap_values retorna array (n, f) ou lista de 2
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals

        mean_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": available_feats_shap,
            "mean_shap": mean_shap,
        }).sort_values("mean_shap", ascending=True).tail(15)

        col_shap, col_shap_info = st.columns([2, 1])

        with col_shap:
            fig_shap = px.bar(
                shap_df, x="mean_shap", y="feature", orientation="h",
                color="mean_shap",
                color_continuous_scale=["#E8F5E9", "#D32F2F"],
                title=f"SHAP — Impacto médio sobre o risco Don't Go ({selected_tag})",
                labels={"mean_shap": "|SHAP|", "feature": "Feature"},
            )
            fig_shap.update_layout(height=480, showlegend=False,
                                    coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_shap_info:
            st.markdown("""
<div class="info-box">
<b>Como interpretar o SHAP:</b><br>
Cada barra mostra quanto aquela feature contribui,
em média, para aumentar ou reduzir a predição de
risco Don't Go neste equipamento.<br><br>
Features com alto <code>|SHAP|</code> são as mais
determinantes para as decisões do modelo neste
contexto específico.
</div>
""", unsafe_allow_html=True)

            # Top 3 features para este equipamento
            top3 = shap_df.tail(3)[::-1].reset_index(drop=True)
            st.markdown("**Top 3 features de risco neste equipamento:**")
            for i, row in top3.iterrows():
                st.markdown(f"**{i+1}.** `{row['feature']}` — SHAP médio: `{row['mean_shap']:.4f}`")

    # Resumo metodológico
    st.divider()
    st.markdown("#### Metodologia do Pipeline")
    st.markdown("""
| Etapa | Descrição |
|---|---|
| **Bronze** | Ingestão dos parquets de telemetria (37M registros, Jan–Jun 2025) |
| **Silver** | Join com apontamentos + flag `is_dont_go_next_60m` (look-ahead temporal) |
| **Gold** | Feature engineering: frequências de alarmes, fingerprint, recência, contexto de turno |
| **Modelo** | LightGBM com `scale_pos_weight=40` para o desbalanceamento severo (0,45% positivos) |
| **Validação** | Temporal: treino Jan–Abr, validação Mai (threshold), teste Jun (sem data leakage) |
| **Explicabilidade** | SHAP TreeExplainer sobre as 53 features do modelo |

**Destaques:** desbalanceamento severo (0.054% DG), pipeline medallion completo,
alarm fingerprint dos top-30 alarmes como features binárias, aceleração de alarmes
críticos como signal preditivo principal.
""")
