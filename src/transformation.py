"""Bronze → Silver: joins, enriquecimento e criação do target preditivo."""

from pathlib import Path
import time
import warnings
import polars as pl

from ingestion import (
    MONTH_FILES,
    load_telemetry,
    load_apontamentos,
    cast_telemetry_types,
)

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

    Estratégia: join_asof backward em (TAG, Data_Evento → Inicio) para encontrar o
    último apontamento que começou antes ou no momento do evento. Em seguida, filtra
    eventos cujo Data_Evento ultrapassa o Fim do apontamento encontrado — esses eventos
    ficam sem apontamento correspondente (colunas de apontamento = null).

    Apontamentos com Inicio duplicado para a mesma TAG (340 casos nos dados) são
    resolvidos pelo sort estável: fica o de Id maior (mais recente no sistema de origem).

    Colunas adicionadas: apontamento_id, apontamento_inicio, apontamento_fim,
    apontamento_classe, frota, tipo_equipamento.
    """
    # Harmonizar datetime[ns] → datetime[us] para compatibilidade com telemetria
    apo = (
        apontamentos
        .with_columns([
            pl.col("Inicio").dt.cast_time_unit("us").alias("Inicio"),
            pl.col("Fim").dt.cast_time_unit("us").alias("Fim"),
        ])
        .rename({"Tag": "TAG"})
        .sort(["TAG", "Inicio", "Id"])  # Id desempata duplicatas de Inicio
        .select(["TAG", "Id", "Inicio", "Fim", "Classe", "Frota", "Tipo"])
    )

    tel = telemetry.sort(["TAG", "Data_Evento"])

    # join_asof backward: para cada evento, pega o apontamento cujo Inicio é <= Data_Evento
    # Warning "Sortedness cannot be checked when by groups provided" é comportamento
    # esperado do Polars com by= em join_asof — resultado é correto.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        joined = tel.join_asof(
            apo,
            left_on="Data_Evento",
            right_on="Inicio",
            by="TAG",
            strategy="backward",
        )

    # Descartar match quando o evento caiu após o Fim do apontamento
    return joined.with_columns([
        pl.when(pl.col("Data_Evento") > pl.col("Fim"))
          .then(None)
          .otherwise(pl.col("Id"))
          .alias("apontamento_id"),
        pl.when(pl.col("Data_Evento") > pl.col("Fim"))
          .then(None)
          .otherwise(pl.col("Inicio"))
          .alias("apontamento_inicio"),
        pl.when(pl.col("Data_Evento") > pl.col("Fim"))
          .then(None)
          .otherwise(pl.col("Fim"))
          .alias("apontamento_fim"),
        pl.when(pl.col("Data_Evento") > pl.col("Fim"))
          .then(None)
          .otherwise(pl.col("Classe_right"))
          .alias("apontamento_classe"),
        pl.when(pl.col("Data_Evento") > pl.col("Fim"))
          .then(None)
          .otherwise(pl.col("Frota"))
          .alias("frota"),
        pl.when(pl.col("Data_Evento") > pl.col("Fim"))
          .then(None)
          .otherwise(pl.col("Tipo_right"))
          .alias("tipo_equipamento"),
    ]).drop(["Id", "Inicio", "Fim", "Classe_right", "Frota", "Tipo_right"])


def build_silver_dataset(
    months: list[str] | None = None,
    dg_reference: pl.DataFrame | None = None,
    lead_times: list[int] = [60, 120, 240],
) -> pl.DataFrame:
    """Orquestra o pipeline Bronze → Silver e salva em outputs/silver/.

    Processa cada mês individualmente para controle de memória, então concatena
    e salva o resultado particionado por mês em outputs/silver/.

    Args:
        months: lista de meses a processar (ex: ['jan', 'feb']). None = todos.
        dg_reference: referência global de eventos DG para flag de janelas.
            Se None, extrai dos dados de todos os meses solicitados.
        lead_times: janelas em minutos para is_dont_go_next_{N}m.

    Returns:
        DataFrame Silver completo (carregado em memória após salvar em parquet).
    """
    OUTPUT_SILVER.mkdir(parents=True, exist_ok=True)

    if months is None:
        months = list(MONTH_FILES.keys())

    # dg_reference global: necessário para capturar DGs de meses adjacentes na flag
    if dg_reference is None:
        print("Extraindo referência global de eventos Don't Go...")
        lf_all = load_telemetry()
        lf_all = cast_telemetry_types(lf_all)
        dg_reference = _extract_dg_reference(lf_all)
        print(f"  → {len(dg_reference)} eventos DG únicos encontrados")

    lf_apo = load_apontamentos()
    parts = []

    for mes in months:
        t0 = time.perf_counter()
        print(f"[{mes}] Processando...", end=" ", flush=True)

        lf = load_telemetry([mes])
        lf = cast_telemetry_types(lf)

        # Join com apontamentos
        lf = join_telemetry_apontamentos(lf, lf_apo)

        # Flag de janelas pré-Don't Go
        lf = flag_pre_dont_go_windows(lf, lead_times=lead_times, dg_reference=dg_reference)

        df = lf.collect()

        out_path = OUTPUT_SILVER / f"silver_{mes}.parquet"
        df.write_parquet(str(out_path))

        dt = time.perf_counter() - t0
        n_dg = df.filter(pl.col("Is_Dont_Go") == 1).height
        print(f"{len(df):,} registros | {n_dg:,} DG | {dt:.1f}s → {out_path.name}")
        parts.append(df)

    result = pl.concat(parts)
    print(f"\nSilver total: {len(result):,} registros em {len(months)} meses")
    return result


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
