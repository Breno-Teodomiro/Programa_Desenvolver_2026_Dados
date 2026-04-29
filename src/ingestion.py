"""Raw → Bronze: carregamento, tipagem e validação dos dados brutos."""

from pathlib import Path
import polars as pl

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "Base_Dados" / "datasets"
TELEMETRY_DIR = DATA_DIR / "telemetria"
APONTAMENTOS_PATH = DATA_DIR / "apontamentos" / "desenvolver_apontamentos.parquet"
ALARM_RULES_PATH = ROOT / "Base_Dados" / "Alarmes - Regra de Negocio.xlsx"

MONTH_FILES = {
    "jan": "telemetry_jan.parquet",
    "feb": "telemetry_feb.parquet",
    "mar": "telemetry_mar.parquet",
    "abr": "telemetry_abr.parquet",
    "may": "telemetry_may.parquet",
    "jun": "telemetry_jun.parquet",
}

# Schema real observado nos parquet (após inspeção dos dados)
TELEMETRY_SCHEMA = {
    "Id_Eventos_Telemetria": pl.Int64,
    "Data_Evento": pl.Datetime("us"),
    "Inicio_Turno": pl.String,      # datetime string — parsear com cast_telemetry_types()
    "Fim_Turno": pl.String,         # datetime string — parsear com cast_telemetry_types()
    "Dia": pl.Int64,
    "Localidade": pl.String,
    "TAG": pl.String,
    "Tag_Frota": pl.String,
    "Tipo": pl.String,
    "Nome_Operador_Anon": pl.String,
    "Matricula_Operador_Hash": pl.String,
    "Id_Alarme": pl.Int64,
    "Alarme": pl.String,
    "Id_Criticidade": pl.Int64,
    "Criticidade": pl.String,
    "Valor": pl.String,             # número BR (vírgula decimal) — parsear com cast_telemetry_types()
    "Classe": pl.String,
    "Is_Dont_Go": pl.Int8,
}

APONTAMENTOS_SCHEMA = {
    "Id": pl.Int64,
    "Inicio": pl.Datetime("ns"),
    "Fim": pl.Datetime("ns"),
    "Tag": pl.String,
    "Frota": pl.String,
    "Tipo": pl.String,
    "Classe": pl.String,
}


def load_telemetry(months: list[str] | None = None) -> pl.LazyFrame:
    """Carrega telemetria como LazyFrame. months=['jan','feb'] ou None para todos."""
    if months is None:
        return pl.scan_parquet(str(TELEMETRY_DIR / "*.parquet"))

    files = []
    for m in months:
        m = m.lower()
        if m not in MONTH_FILES:
            raise ValueError(f"Mês '{m}' inválido. Válidos: {list(MONTH_FILES)}")
        files.append(str(TELEMETRY_DIR / MONTH_FILES[m]))

    return pl.scan_parquet(files)


def cast_telemetry_types(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Converte colunas de tipos ambíguos para tipos analíticos corretos.

    - Valor: string com vírgula decimal → Float64
    - Inicio_Turno / Fim_Turno: string → Datetime
    - Id_Criticidade: Int64 → Int8 (economiza memória)
    """
    return lf.with_columns([
        pl.col("Valor")
          .str.replace(",", ".")
          .cast(pl.Float64, strict=False)
          .alias("Valor"),
        pl.col("Inicio_Turno")
          .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False)
          .alias("Inicio_Turno"),
        pl.col("Fim_Turno")
          .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False)
          .alias("Fim_Turno"),
        pl.col("Id_Criticidade").cast(pl.Int8),
    ])


def load_apontamentos() -> pl.LazyFrame:
    """Carrega apontamentos como LazyFrame."""
    return pl.scan_parquet(str(APONTAMENTOS_PATH))


def load_alarm_rules() -> pl.DataFrame:
    """Carrega as regras de negócio de alarmes do xlsx."""
    import pandas as pd
    df_pd = pd.read_excel(str(ALARM_RULES_PATH), engine="openpyxl")
    return pl.from_pandas(df_pd)


def validate_schema(lf: pl.LazyFrame, expected: dict) -> bool:
    """Valida que o schema do LazyFrame contém as colunas esperadas."""
    actual = lf.collect_schema()
    errors = []
    for col, dtype in expected.items():
        if col not in actual:
            errors.append(f"Coluna ausente: '{col}'")
        elif actual[col] != dtype:
            errors.append(f"'{col}': esperado {dtype}, encontrado {actual[col]}")
    if errors:
        for e in errors:
            print(f"[SCHEMA] {e}")
        return False
    return True


def get_telemetry_stats() -> pl.DataFrame:
    """Retorna contagem de registros, período e eventos Don't Go por mês."""
    import duckdb
    glob = str(TELEMETRY_DIR / "*.parquet")
    return duckdb.execute(f"""
        SELECT
            regexp_extract(filename, 'telemetry_(\\w+)\\.parquet', 1) AS mes,
            COUNT(*) AS registros,
            SUM(Is_Dont_Go) AS dont_go_eventos,
            ROUND(100.0 * SUM(Is_Dont_Go) / COUNT(*), 4) AS dont_go_pct,
            MIN(Data_Evento) AS data_inicio,
            MAX(Data_Evento) AS data_fim
        FROM read_parquet('{glob}', filename=true)
        GROUP BY mes
        ORDER BY data_inicio
    """).pl()
