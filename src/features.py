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

def build_feature_matrix(
    silver_months: list[str] | None = None,
    top_alarm_ids: list[int] | None = None,
    save: bool = True,
) -> tuple[pl.DataFrame, list[int]]:
    """Carrega silver, aplica todas as features e salva outputs/gold/gold_features.parquet.

    Args:
        silver_months: sufixos dos meses a usar (ex: ['jan','feb']). None = todos.
        top_alarm_ids: IDs de alarme para fingerprint (reusar entre treino/teste).
        save: se True, persiste em parquet.

    Returns:
        (DataFrame Gold, lista de alarm_ids usados na fingerprint)
    """
    OUTPUT_GOLD.mkdir(parents=True, exist_ok=True)

    if silver_months is None:
        silver_files = sorted(SILVER_DIR.glob("silver_*.parquet"))
    else:
        silver_files = [SILVER_DIR / f"silver_{m}.parquet" for m in silver_months]

    missing = [f for f in silver_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Silver files ausentes: {[f.name for f in missing]}")

    print(f"Carregando {len(silver_files)} arquivo(s) silver...")
    df = pl.scan_parquet([str(f) for f in silver_files]).collect()
    print(f"  → {len(df):,} registros carregados")

    print("Features temporais...")
    df = compute_temporal_features(df.lazy()).collect()

    print("Features de frequência de alarmes (30m / 1h / 4h)...")
    df = compute_alarm_frequency_features(df)

    print(f"Alarm fingerprint (top {TOP_N_FINGERPRINT} alarmes, janela 4h)...")
    df, top_alarm_ids = compute_alarm_fingerprint(df, top_alarm_ids=top_alarm_ids)

    print("Features de contexto do equipamento...")
    df = compute_equipment_history_features(df)

    print(f"Gold dataset: {len(df):,} registros × {len(df.columns)} colunas")

    if save:
        out_path = OUTPUT_GOLD / "gold_features.parquet"
        df.write_parquet(str(out_path))
        print(f"Salvo em {out_path}")

    return df, top_alarm_ids


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
