"""Silver → Gold: feature engineering para o modelo preditivo de Don't Go.

Processamento dia a dia para limitar uso de RAM:
cada chunk = 1 dia de dados + overlap_hours horas do dia anterior
(garante janelas rolantes de até 4h corretas sem carregar o mês inteiro).
"""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

OUTPUT_GOLD = Path(__file__).parent.parent / "outputs" / "gold"
SILVER_DIR = Path(__file__).parent.parent / "outputs" / "silver"

ROLLING_WINDOWS = [15, 30, 60, 240]  # minutos — 15m captura spikes imediatos pré-DG
TOP_N_FINGERPRINT = 30
OVERLAP_HOURS = 4  # deve ser >= maior janela (240m = 4h)


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

    Janelas: 30min, 1h (60m), 4h (240m). closed='left' exclui o evento atual.
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

    eps = 1e-3
    return df.with_columns([
        # aceleração de alarmes críticos (já existia)
        (pl.col("n_criticos_60m") / ((pl.col("n_criticos_240m") / 4.0) + eps))
        .alias("aceleracao_criticos"),
        # aceleração de DG: taxa 60m vs taxa média 240m — detecta clustering iminente
        (pl.col("n_dg_60m") / ((pl.col("n_dg_240m") / 4.0) + eps))
        .alias("aceleracao_dg"),
    ])


def compute_recency_features(df: pl.DataFrame) -> pl.DataFrame:
    """Tempo em minutos desde o último evento Don't Go por equipamento.

    Captura recência: se houve DG recente, o risco de novo DG é mais alto.
    Valor = minutos desde o último Is_Dont_Go=1 anterior ao evento atual.
    Nulo (sem DG anterior no histórico carregado) → preenchido com 9999.
    """
    df = df.sort(["TAG", "Data_Evento"])

    # timestamp do último DG anterior a cada evento (exclusive)
    last_dg_ts = (
        pl.when(pl.col("Is_Dont_Go").cast(pl.Int8) == 1)
        .then(pl.col("Data_Evento"))
        .otherwise(None)
        .shift(1)
        .forward_fill()
        .over("TAG")
    )

    min_desde_ultimo_dg = (
        (pl.col("Data_Evento") - last_dg_ts)
        .dt.total_minutes()
        .cast(pl.Float32)
        .fill_null(9999.0)
        .clip(lower_bound=0.0)
    )

    return df.with_columns(min_desde_ultimo_dg.alias("min_desde_ultimo_dg"))


def compute_alarm_fingerprint(
    df: pl.DataFrame,
    top_alarm_ids: list[int] | None = None,
) -> tuple[pl.DataFrame, list[int]]:
    """Presença (0/1) dos top-N alarmes na janela de 4h anterior a cada evento.

    Se top_alarm_ids for fornecido, reutiliza a lista (consistência treino/teste).
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


def _apply_frota_encoding(df: pl.DataFrame, frota_map: dict) -> pl.DataFrame:
    """Aplica label encoding de frota e flags de estado do apontamento."""
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


# ── Estatísticas globais ──────────────────────────────────────────────────────

def _compute_global_stats(
    silver_files: list[Path],
    top_alarm_ids: list[int] | None,
) -> tuple[list[int], dict[str, int]]:
    """Passa rápida (DuckDB) para obter top_alarm_ids e frota_map globais."""
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


# ── Orquestrador dia a dia ────────────────────────────────────────────────────

def build_feature_matrix(
    silver_months: list[str] | None = None,
    top_alarm_ids: list[int] | None = None,
    overlap_hours: int = OVERLAP_HOURS,
    save: bool = True,
) -> tuple[pl.LazyFrame, list[int]]:
    """Carrega silver DIA A DIA, aplica features e salva outputs/gold/gold_{mes}.parquet.

    Processa um dia por vez com `overlap_hours` horas do dia anterior como contexto
    para as janelas rolantes, depois descarta o overlap do output final.
    Uso de RAM ~proporcional a 1 dia de dados (~200K linhas) em vez do mês inteiro.

    Args:
        silver_months: sufixos dos meses (ex: ['jan','feb']). None = todos.
        top_alarm_ids: IDs de alarme para fingerprint (consistência treino/teste).
        overlap_hours: horas de contexto do dia anterior (deve ser >= 4).
        save: se True, persiste em parquet por mês em outputs/gold/.

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
    overlap_raw: pl.DataFrame | None = None  # últimas overlap_hours do dia anterior (raw silver)

    for silver_file in silver_files:
        mes = silver_file.stem.replace("silver_", "")
        gold_out = OUTPUT_GOLD / f"gold_{mes}.parquet"

        if gold_out.exists():
            print(f"\n[{mes}] gold já existe — pulando (carregando tail para overlap)")
            # Carrega tail para dar continuidade ao overlap no próximo mês
            silver_str = str(silver_file)
            tail_result = duckdb.execute(f"""
                SELECT * FROM read_parquet('{silver_str}')
                WHERE Data_Evento >= (
                    SELECT MAX(Data_Evento) - INTERVAL '{overlap_hours} hours'
                    FROM read_parquet('{silver_str}')
                )
                ORDER BY TAG, Data_Evento
            """).pl()
            overlap_raw = tail_result if len(tail_result) > 0 else None
            gold_files.append(gold_out)
            continue

        # Obtém dias únicos deste mês via DuckDB
        silver_str = str(silver_file)
        days_result = duckdb.execute(f"""
            SELECT DISTINCT CAST(Data_Evento AS DATE) AS d
            FROM read_parquet('{silver_str}')
            ORDER BY d
        """).fetchall()
        days = [row[0] for row in days_result]

        print(f"\n[{mes}] {len(days)} dias a processar...")
        daily_dfs: list[pl.DataFrame] = []

        for day_idx, day in enumerate(days):
            day_dt = datetime(day.year, day.month, day.day)
            next_day_dt = day_dt + timedelta(days=1)

            # Carrega dados do dia atual via DuckDB (push-down de predicado no parquet)
            df_day = duckdb.execute(f"""
                SELECT * FROM read_parquet('{silver_str}')
                WHERE Data_Evento >= TIMESTAMP '{day_dt}'
                  AND Data_Evento  < TIMESTAMP '{next_day_dt}'
                ORDER BY TAG, Data_Evento
            """).pl()

            if len(df_day) == 0:
                continue

            # Monta chunk: overlap do dia anterior + dia atual
            if overlap_raw is not None and len(overlap_raw) > 0:
                df_chunk = pl.concat([overlap_raw, df_day]).sort(["TAG", "Data_Evento"])
                n_overlap = len(overlap_raw)
            else:
                df_chunk = df_day.sort(["TAG", "Data_Evento"])
                n_overlap = 0

            # Feature engineering no chunk
            df_chunk = compute_temporal_features(df_chunk.lazy()).collect()
            df_chunk = compute_alarm_frequency_features(df_chunk)
            df_chunk = compute_recency_features(df_chunk)
            df_chunk, _ = compute_alarm_fingerprint(df_chunk, top_alarm_ids=top_alarm_ids)
            df_chunk = _apply_frota_encoding(df_chunk, frota_map)

            # Remove linhas de overlap do output filtrando por data (slice por posição
            # é incorreto pois o overlap está distribuído por TAG no frame ordenado)
            if n_overlap > 0:
                df_chunk = df_chunk.filter(
                    pl.col("Data_Evento") >= pl.lit(day_dt).cast(pl.Datetime("us"))
                )

            daily_dfs.append(df_chunk)

            # Atualiza overlap: últimas overlap_hours do dia atual (raw silver)
            cutoff_ts = next_day_dt - timedelta(hours=overlap_hours)
            overlap_raw = df_day.filter(
                pl.col("Data_Evento") >= pl.lit(cutoff_ts).cast(pl.Datetime("us"))
            )

            del df_chunk

            # Progresso a cada 5 dias
            if (day_idx + 1) % 5 == 0 or (day_idx + 1) == len(days):
                rows_so_far = sum(len(d) for d in daily_dfs)
                print(f"  dia {day_idx + 1}/{len(days)} — {rows_so_far:,} gold rows acumuladas")

        if not daily_dfs:
            print(f"  [AVISO] {mes}: nenhum dado processado")
            continue

        gold_month = pl.concat(daily_dfs)
        total_rows += len(gold_month)
        del daily_dfs

        if save:
            gold_month.write_parquet(str(gold_out))
            gold_files.append(gold_out)
            print(f"  → {len(gold_month):,} gold rows salvas em {gold_out.name}")

        del gold_month

    print(f"\nGold total: {total_rows:,} registros em {len(gold_files)} meses")

    lf = pl.scan_parquet([str(f) for f in gold_files])
    return lf, top_alarm_ids


def get_feature_columns(df: pl.DataFrame, include_fingerprint: bool = True) -> list[str]:
    """Lista de colunas de features para ML (exclui IDs, timestamps e targets)."""
    exclude_exact = {
        "Id_Eventos_Telemetria", "TAG", "Tag_Frota", "Tipo", "Localidade",
        "Alarme", "Criticidade", "Inicio_Turno", "Fim_Turno", "Valor",
        "Classe", "Nome_Operador_Anon", "Matricula_Operador_Hash",
        "apontamento_id", "apontamento_classe", "frota", "tipo_equipamento",
        "Id_Alarme", "Id_Criticidade",
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
