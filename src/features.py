"""Silver → Gold: feature engineering para o modelo preditivo de Don't Go."""

from pathlib import Path
import polars as pl

OUTPUT_GOLD = Path(__file__).parent.parent / "outputs" / "gold"
SILVER_DIR = Path(__file__).parent.parent / "outputs" / "silver"

ROLLING_WINDOWS = [30, 60, 240]  # minutos
TOP_N_FINGERPRINT = 30


# ── Feature groups ────────────────────────────────────────────────────────────

def compute_temporal_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Hora do dia, posição no turno, dia da semana, mês e flag turno noturno."""
    turno_dur = (
        (pl.col("Fim_Turno") - pl.col("Inicio_Turno"))
        .dt.total_minutes()
        .clip(lower_bound=1)
    )
    posicao = (
        (pl.col("Data_Evento") - pl.col("Inicio_Turno"))
        .dt.total_minutes()
        / turno_dur
    ).clip(lower_bound=0.0, upper_bound=1.0)

    is_noturno = (
        pl.col("Inicio_Turno").dt.hour().is_between(18, 23)
        | pl.col("Inicio_Turno").dt.hour().is_between(0, 5)
    ).cast(pl.Int8)

    return lf.with_columns([
        pl.col("Data_Evento").dt.hour().alias("hora_dia"),
        pl.col("Data_Evento").dt.weekday().alias("dia_semana"),
        pl.col("Data_Evento").dt.month().alias("mes"),
        posicao.alias("posicao_turno"),
        is_noturno.alias("is_turno_noturno"),
    ])


def compute_alarm_frequency_features(df: pl.DataFrame) -> pl.DataFrame:
    """Contagem e aceleração de alarmes por janela temporal anterior a cada evento.

    Janelas: 30min, 1h (60m), 4h (240m). Usa rolling_sum_by por TAG.
    closed='left' exclui o evento atual da janela (look-ahead free).
    """
    df = df.sort(["TAG", "Data_Evento"]).with_columns(
        pl.lit(1, dtype=pl.Int32).alias("_one")
    )

    freq_exprs = []
    for w in ROLLING_WINDOWS:
        freq_exprs.extend([
            pl.col("_one")
              .rolling_sum_by("Data_Evento", window_size=f"{w}m", closed="left")
              .over("TAG")
              .alias(f"n_alarmes_{w}m"),
            pl.col("Id_Criticidade").eq(1).cast(pl.Int32)
              .rolling_sum_by("Data_Evento", window_size=f"{w}m", closed="left")
              .over("TAG")
              .alias(f"n_criticos_{w}m"),
            pl.col("Id_Criticidade").eq(2).cast(pl.Int32)
              .rolling_sum_by("Data_Evento", window_size=f"{w}m", closed="left")
              .over("TAG")
              .alias(f"n_nao_criticos_{w}m"),
            pl.col("Is_Dont_Go").cast(pl.Int32)
              .rolling_sum_by("Data_Evento", window_size=f"{w}m", closed="left")
              .over("TAG")
              .alias(f"n_dg_{w}m"),
        ])

    df = df.with_columns(freq_exprs).drop("_one")

    # Aceleração: quantas vezes a taxa de críticos na última hora excede a média das 4h
    eps = 1e-3
    return df.with_columns(
        (pl.col("n_criticos_60m") / ((pl.col("n_criticos_240m") / 4.0) + eps))
        .alias("aceleracao_criticos")
    )


def compute_alarm_fingerprint(
    df: pl.DataFrame,
    top_alarm_ids: list[int] | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """Presença (0/1) dos top-N alarmes na janela de 4h anterior a cada evento.

    Produz colunas fp_alarm_{id} para os TOP_N_FINGERPRINT alarmes mais frequentes.
    Se top_alarm_ids for fornecido, reutiliza a lista (para consistência treino/teste).
    """
    if top_alarm_ids is None:
        top_alarm_ids = (
            df.group_by("Id_Alarme")
              .agg(pl.len().alias("cnt"))
              .sort("cnt", descending=True)
              .head(TOP_N_FINGERPRINT)
              ["Id_Alarme"]
              .to_list()
        )

    df = df.sort(["TAG", "Data_Evento"])

    fingerprint_exprs = [
        pl.col("Id_Alarme").eq(alarm_id).cast(pl.Int32)
          .rolling_sum_by("Data_Evento", window_size="240m", closed="left")
          .over("TAG")
          .gt(0).cast(pl.Int8)
          .alias(f"fp_alarm_{alarm_id}")
        for alarm_id in top_alarm_ids
    ]

    return df.with_columns(fingerprint_exprs), top_alarm_ids


def compute_equipment_history_features(df: pl.DataFrame) -> pl.DataFrame:
    """Label encoding de frota e flags de estado do apontamento."""
    frotas = sorted(df["Tag_Frota"].drop_nulls().unique().to_list())
    frota_map = {f: i for i, f in enumerate(frotas)}

    return df.with_columns([
        pl.col("Tag_Frota")
          .replace(frota_map, default=None)
          .cast(pl.Int16)
          .alias("frota_encoded"),
        pl.col("apontamento_classe")
          .eq("Operando").cast(pl.Int8).fill_null(0)
          .alias("is_em_operacao"),
        pl.col("apontamento_classe")
          .str.contains("anuten").cast(pl.Int8).fill_null(0)
          .alias("is_em_manutencao"),
        pl.col("apontamento_id").is_null().cast(pl.Int8)
          .alias("sem_apontamento"),
    ])


# ── Orquestrador ─────────────────────────────────────────────────────────────

def _compute_global_stats(
    silver_files: list[Path],
    top_alarm_ids: list[int] | None,
) -> tuple[list[int], dict[str, int]]:
    """Passa rápida (LazyFrame) para obter top_alarm_ids e frota_map globais."""
    import duckdb
    paths = [str(f) for f in silver_files]
    glob_expr = str(silver_files[0].parent / "silver_*.parquet")

    if top_alarm_ids is None:
        result = duckdb.execute(f"""
            SELECT Id_Alarme, COUNT(*) AS cnt
            FROM read_parquet('{glob_expr}')
            GROUP BY Id_Alarme
            ORDER BY cnt DESC
            LIMIT {TOP_N_FINGERPRINT}
        """).fetchall()
        top_alarm_ids = [row[0] for row in result]

    frotas = duckdb.execute(f"""
        SELECT DISTINCT Tag_Frota FROM read_parquet('{glob_expr}')
        WHERE Tag_Frota IS NOT NULL ORDER BY Tag_Frota
    """).fetchall()
    frota_map = {row[0]: i for i, row in enumerate(frotas)}

    return top_alarm_ids, frota_map


def build_feature_matrix(
    silver_months: list[str] | None = None,
    top_alarm_ids: list[int] | None = None,
    overlap_hours: int = 4,
    save: bool = True,
) -> tuple[pl.LazyFrame, list[int]]:
    """Carrega silver mês a mês, aplica features e salva outputs/gold/gold_{mes}.parquet.

    Processa um mês por vez para limitar uso de RAM. Inclui `overlap_hours` horas
    do mês anterior como contexto para rolling windows, removendo-as do output.

    Args:
        silver_months: sufixos dos meses a usar (ex: ['jan','feb']). None = todos.
        top_alarm_ids: IDs de alarme para fingerprint (reusar entre treino/teste).
        overlap_hours: horas de overlap do mês anterior para rolling windows corretas.
        save: se True, persiste em parquet por mês.

    Returns:
        (LazyFrame Gold apontando para os parquets salvos, lista de alarm_ids)
    """
    OUTPUT_GOLD.mkdir(parents=True, exist_ok=True)

    if silver_months is None:
        silver_files = sorted(SILVER_DIR.glob("silver_*.parquet"))
    else:
        silver_files = [SILVER_DIR / f"silver_{m}.parquet" for m in silver_months]

    missing = [f for f in silver_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Silver files ausentes: {[f.name for f in missing]}")

    print("Calculando estatísticas globais (DuckDB scan)...")
    top_alarm_ids, frota_map = _compute_global_stats(silver_files, top_alarm_ids)
    print(f"  → top {len(top_alarm_ids)} alarmes | {len(frota_map)} frotas")

    gold_files = []
    total_rows = 0

    for i, silver_file in enumerate(silver_files):
        mes = silver_file.stem.replace("silver_", "")
        print(f"\n[{mes}] Processando...", end=" ", flush=True)

        # Carrega mês atual
        df = pl.read_parquet(str(silver_file))

        # Adiciona overlap do mês anterior para rolling windows corretas
        if i > 0:
            prev_file = silver_files[i - 1]
            cutoff = df["Data_Evento"].min() - pl.duration(hours=overlap_hours)
            tail = (
                pl.scan_parquet(str(prev_file))
                .filter(pl.col("Data_Evento") >= cutoff)
                .collect()
            )
            df = pl.concat([tail, df]).sort(["TAG", "Data_Evento"])
            n_overlap = len(tail)
            del tail
        else:
            n_overlap = 0

        print(f"{len(df):,} registros (+{n_overlap} overlap)", end=" ", flush=True)

        df = compute_temporal_features(df.lazy()).collect()
        df = compute_alarm_frequency_features(df)
        df, _ = compute_alarm_fingerprint(df, top_alarm_ids=top_alarm_ids)

        # Aplica frota_map global (em vez de derivar por mês)
        df = df.with_columns([
            pl.col("Tag_Frota")
              .replace(frota_map, default=None)
              .cast(pl.Int16)
              .alias("frota_encoded"),
            pl.col("apontamento_classe")
              .eq("Operando").cast(pl.Int8).fill_null(0)
              .alias("is_em_operacao"),
            pl.col("apontamento_classe")
              .str.contains("anuten").cast(pl.Int8).fill_null(0)
              .alias("is_em_manutencao"),
            pl.col("apontamento_id").is_null().cast(pl.Int8)
              .alias("sem_apontamento"),
        ])

        # Remove linhas de overlap do output final
        if n_overlap > 0:
            df = df.slice(n_overlap)

        total_rows += len(df)
        print(f"→ {len(df):,} gold rows", end="")

        if save:
            out_path = OUTPUT_GOLD / f"gold_{mes}.parquet"
            df.write_parquet(str(out_path))
            gold_files.append(out_path)
            print(f" → {out_path.name}")
        else:
            gold_files.append(silver_file)  # placeholder
            print()

        del df  # libera RAM imediatamente

    print(f"\nGold total: {total_rows:,} registros em {len(silver_files)} meses")

    if save:
        lf = pl.scan_parquet([str(f) for f in gold_files])
    else:
        lf = pl.scan_parquet([str(f) for f in silver_files])  # fallback

    return lf, top_alarm_ids


def get_feature_columns(df: pl.DataFrame, include_fingerprint: bool = True) -> list[str]:
    """Lista de colunas de features para ML (exclui IDs, timestamps e targets)."""
    exclude_exact = {
        "Id_Eventos_Telemetria", "TAG", "Tag_Frota", "Tipo", "Localidade",
        "Alarme", "Criticidade", "Inicio_Turno", "Fim_Turno", "Valor",
        "Classe", "Nome_Operador_Anon", "Matricula_Operador_Hash",
        "apontamento_id", "apontamento_classe", "frota", "tipo_equipamento",
    }
    exclude_prefixes = (
        "Data_", "apontamento_inicio", "apontamento_fim",
        "is_dont_go_next_", "minutes_to_next_dg",
    )
    return [
        c for c in df.columns
        if c not in exclude_exact
        and not any(c.startswith(p) for p in exclude_prefixes)
        and (include_fingerprint or not c.startswith("fp_alarm_"))
    ]
