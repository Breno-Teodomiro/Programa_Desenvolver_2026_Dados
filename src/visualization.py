"""Funções de visualização interativa reutilizáveis (Plotly) para o projeto."""

from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"


def _save(fig: go.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.html"
    fig.write_html(str(path))
    return path


# ── EDA / Telemetria ──────────────────────────────────────────────────────────

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
            mode="markers",
            name="Alarme",
            marker=dict(
                color=non_dg["Id_Criticidade"].to_list(),
                colorscale="RdYlGn_r",
                size=5,
                opacity=0.6,
            ),
            text=non_dg["Alarme"].to_list(),
            hovertemplate="%{text}<br>%{x}<extra></extra>",
        ))

    dg = sub.filter(pl.col("Is_Dont_Go") == 1)
    if len(dg) > 0:
        fig.add_trace(go.Scatter(
            x=dg["Data_Evento"].to_list(),
            y=dg["Id_Criticidade"].to_list(),
            mode="markers",
            name="Don't Go",
            marker=dict(color="red", size=10, symbol="x"),
            hovertemplate="%{x}<extra>Don't Go</extra>",
        ))

    fig.update_layout(
        title=f"Timeline de Alarmes — {equipment_tag}",
        xaxis_title="Timestamp",
        yaxis_title="Criticidade (1=Crítico)",
        yaxis=dict(tickvals=[1, 2, 3, 4], ticktext=["Crítico", "Não-Crítico", "Info", "Outro"]),
        legend=dict(x=0, y=1),
        height=450,
    )

    if save:
        _save(fig, f"timeline_{equipment_tag}")
    return fig


def plot_alarm_heatmap(
    df: pl.DataFrame,
    top_n_alarms: int = 30,
    top_n_tags: int = 20,
    save: bool = True,
) -> go.Figure:
    """Heatmap equipamento × tipo de alarme × frequência (log scale)."""
    top_alarms = (
        df.group_by("Id_Alarme")
          .agg(pl.len().alias("cnt"))
          .sort("cnt", descending=True)
          .head(top_n_alarms)
          ["Id_Alarme"].to_list()
    )
    top_tags = (
        df.group_by("TAG")
          .agg(pl.len().alias("cnt"))
          .sort("cnt", descending=True)
          .head(top_n_tags)
          ["TAG"].to_list()
    )

    heat = (
        df.filter(pl.col("Id_Alarme").is_in(top_alarms) & pl.col("TAG").is_in(top_tags))
          .group_by(["TAG", "Id_Alarme"])
          .agg(pl.len().alias("cnt"))
          .pivot(on="Id_Alarme", index="TAG", values="cnt", aggregate_function="sum")
          .fill_null(0)
    )

    tags = heat["TAG"].to_list()
    alarm_cols = [c for c in heat.columns if c != "TAG"]
    z = heat.select(alarm_cols).to_numpy()

    fig = go.Figure(go.Heatmap(
        x=[str(a) for a in alarm_cols],
        y=tags,
        z=np.log1p(z),
        colorscale="YlOrRd",
        hovertemplate="TAG: %{y}<br>Alarme: %{x}<br>Count: %{customdata}<extra></extra>",
        customdata=z,
    ))
    fig.update_layout(
        title=f"Frequência de Alarmes por Equipamento (top {top_n_alarms} alarmes)",
        xaxis_title="Id_Alarme",
        yaxis_title="Equipamento (TAG)",
        height=600,
    )

    if save:
        _save(fig, "heatmap_alarm_equipment")
    return fig


def plot_dontgo_distribution(
    df: pl.DataFrame,
    save: bool = True,
) -> go.Figure:
    """Distribuição de eventos Don't Go por equipamento e por mês."""
    dg = df.filter(pl.col("Is_Dont_Go") == 1)

    by_tag = (
        dg.group_by("TAG")
           .agg(pl.len().alias("n_dg"))
           .sort("n_dg", descending=True)
           .head(30)
    )
    by_month = (
        dg.with_columns(pl.col("Data_Evento").dt.month().alias("mes"))
           .group_by(["mes", "Tipo"])
           .agg(pl.len().alias("n_dg"))
           .sort("mes")
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["DGs por Equipamento (top 30)", "DGs por Mês e Tipo"],
    )

    fig.add_trace(go.Bar(
        x=by_tag["TAG"].to_list(),
        y=by_tag["n_dg"].to_list(),
        name="Don't Go",
        marker_color="crimson",
    ), row=1, col=1)

    for tipo in by_month["Tipo"].unique().to_list():
        sub = by_month.filter(pl.col("Tipo") == tipo)
        fig.add_trace(go.Bar(
            x=sub["mes"].to_list(),
            y=sub["n_dg"].to_list(),
            name=tipo,
        ), row=1, col=2)

    fig.update_layout(
        title="Distribuição de Eventos Don't Go",
        barmode="stack",
        height=450,
        xaxis=dict(tickangle=45),
        xaxis2_title="Mês",
        yaxis2_title="Nº de Don't Gos",
    )

    if save:
        _save(fig, "dontgo_distribution")
    return fig


# ── Modelo ────────────────────────────────────────────────────────────────────

def plot_feature_importance_shap(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
) -> go.Figure:
    """Bar chart de importância de features por mean(|SHAP value|)."""
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
    ).nlargest(top_n).sort_values()

    fig = go.Figure(go.Bar(
        x=mean_abs.values,
        y=mean_abs.index.tolist(),
        orientation="h",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title=f"Top {top_n} Features — Importância SHAP (mean |SHAP|)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        height=max(400, top_n * 25),
    )

    if save:
        _save(fig, "shap_importance")
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 15,
    save: bool = True,
) -> go.Figure:
    """Beeswarm-style scatter dos SHAP values (top_n features)."""
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
    ).nlargest(top_n)
    top_features = mean_abs.index.tolist()

    fig = go.Figure()
    for feat in reversed(top_features):
        idx = X_sample.columns.get_loc(feat)
        sv = shap_values[:, idx]
        fv = X_sample[feat].values

        fig.add_trace(go.Scatter(
            x=sv,
            y=[feat] * len(sv),
            mode="markers",
            marker=dict(
                color=fv,
                colorscale="RdBu_r",
                size=3,
                opacity=0.5,
                colorbar=dict(title="Valor da feature", thickness=10) if feat == top_features[0] else None,
            ),
            name=feat,
            showlegend=False,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"SHAP Beeswarm — Top {top_n} Features",
        xaxis_title="SHAP value (impacto no log-odds)",
        height=max(500, top_n * 30),
    )

    if save:
        _save(fig, "shap_beeswarm")
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    save: bool = True,
) -> go.Figure:
    """Matriz de confusão interativa com valores absolutos e percentuais."""
    from sklearn.metrics import confusion_matrix as cm_fn
    if labels is None:
        labels = ["Sem DG (0)", "Com DG (1)"]

    cm = cm_fn(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    text = [[f"{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real",
        height=400,
        width=450,
    )

    if save:
        _save(fig, "confusion_matrix")
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
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"PR Curve (AUC={pr_auc:.3f})",
        line=dict(color="darkorange", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.mean()],
        y=[y_test.mean()],
        mode="markers",
        name="Baseline (random)",
        marker=dict(color="gray", size=8, symbol="diamond"),
    ))
    fig.update_layout(
        title="Curva Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        height=450,
    )

    if save:
        _save(fig, "precision_recall_curve")
    return fig


def plot_calibration_curve(
    model,
    X_test: pd.DataFrame,
    y_test,
    n_bins: int = 10,
    save: bool = True,
) -> go.Figure:
    """Curva de calibração: probabilidade predita vs. fração real de positivos."""
    from sklearn.calibration import calibration_curve
    y_prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Calibração perfeita",
        line=dict(dash="dash", color="gray"),
    ))
    fig.add_trace(go.Scatter(
        x=mean_pred, y=frac_pos, mode="lines+markers",
        name="LightGBM",
        marker=dict(size=8),
        line=dict(color="steelblue"),
    ))
    fig.update_layout(
        title="Curva de Calibração",
        xaxis_title="Probabilidade predita (média por bin)",
        yaxis_title="Fração de positivos real",
        height=400,
    )

    if save:
        _save(fig, "calibration_curve")
    return fig


def plot_score_distribution(
    model,
    X_test: pd.DataFrame,
    y_test,
    save: bool = True,
) -> go.Figure:
    """Histograma dos scores preditos separado por classe real."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_arr = np.array(y_test)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_prob[y_arr == 0],
        name="Sem DG (real)",
        opacity=0.6,
        nbinsx=50,
        marker_color="steelblue",
    ))
    fig.add_trace(go.Histogram(
        x=y_prob[y_arr == 1],
        name="Com DG (real)",
        opacity=0.6,
        nbinsx=50,
        marker_color="crimson",
    ))
    fig.update_layout(
        barmode="overlay",
        title="Distribuição de Scores por Classe Real",
        xaxis_title="Score (probabilidade predita de DG)",
        yaxis_title="Contagem",
        height=400,
    )

    if save:
        _save(fig, "score_distribution")
    return fig
