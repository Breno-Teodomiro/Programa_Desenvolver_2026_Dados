"""Bronze → Silver: joins, enriquecimento e criação do target preditivo."""

from pathlib import Path
import warnings
import polars as pl

from ingestion import load_telemetry, load_apontamentos, cast_telemetry_types

OUTPUT_SILVER = Path(__file__).parent.parent / "outputs" / "silver"


def flag_pre_dont_go_windows(
    lf: pl.LazyFrame,
    lead_times: list[int] = [60, 120, 240],
    dg_reference: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    """Para cada evento de telemetria, marca se um Don't Go ocorrerá nas próximas N minutos.

    Cria colunas: is_dont_go_next_60m, is_dont_go_next_120m, is_dont_go_next_240m
    e minutes_to_next_dg (minutos até o próximo evento DG, null se não houver no período).

    Args:
        lf: LazyFrame de telemetria com cast_telemetry_types já aplicado.
        lead_times: janelas de look-ahead em minutos.
        dg_reference: DataFrame com colunas (TAG, dg_time) de todos os eventos DG.
            Passar explicitamente ao processar por mês para capturar DGs de meses
            adjacentes. Se None, extrai do próprio lf.
    """
    if dg_reference is None:
        dg_reference = _extract_dg_reference(lf)

    events = (
        lf.select(["Id_Eventos_Telemetria", "TAG", "Data_Evento"])
          .sort(["TAG", "Data_Evento"])
          .collect()
    )

    # +1µs faz o join ser estritamente futuro (> e não >=).
    events_fwd = events.with_columns(
        (pl.col("Data_Evento") + pl.duration(microseconds=1)).alias("_t_fwd")
    )

    # join_asof forward: para cada evento, encontra o próximo DG do mesmo TAG.
    # O warning "Sortedness cannot be checked when by groups provided" é comportamento
    # esperado do Polars com by= em join_asof — resultado é correto, apenas suprimido.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        joined = events_fwd.join_asof(
            dg_reference,
            left_on="_t_fwd",
            right_on="dg_time",
            by="TAG",
            strategy="forward",
        )

    joined = (
        joined
        .with_columns(
            (pl.col("dg_time") - pl.col("Data_Evento"))
            .dt.total_minutes()
            .alias("minutes_to_next_dg")
        )
        .with_columns([
            (pl.col("minutes_to_next_dg") <= lt)
            .fill_null(False)
            .alias(f"is_dont_go_next_{lt}m")
            for lt in lead_times
        ])
        .drop(["_t_fwd", "dg_time", "TAG", "Data_Evento"])
    )

    return lf.join(joined.lazy(), on="Id_Eventos_Telemetria", how="left")


def join_telemetry_apontamentos(
    telemetry: pl.LazyFrame,
    apontamentos: pl.LazyFrame,
) -> pl.LazyFrame:
    """Para cada evento de telemetria, encontra o apontamento ativo no mesmo equipamento.

    Pendente — implementar na próxima sessão.
    Estratégia: join por TAG + Data_Evento dentro do intervalo [Inicio, Fim] do apontamento.
    """
    raise NotImplementedError("Implementar na Fase 2 — próxima sessão.")


def build_silver_dataset(months: list[str] | None = None) -> None:
    """Orquestra o pipeline Bronze → Silver e salva em outputs/silver/.

    Pendente — implementar na próxima sessão após join_telemetry_apontamentos().
    """
    raise NotImplementedError("Implementar na Fase 2 — próxima sessão.")


# ── Auxiliares ────────────────────────────────────────────────────────────────

def _extract_dg_reference(lf: pl.LazyFrame) -> pl.DataFrame:
    """Extrai e deduplica todos os timestamps de eventos Don't Go por TAG."""
    return (
        lf.filter(pl.col("Is_Dont_Go") == 1)
          .select(["TAG", pl.col("Data_Evento").alias("dg_time")])
          .unique()
          .sort(["TAG", "dg_time"])
          .collect()
    )
