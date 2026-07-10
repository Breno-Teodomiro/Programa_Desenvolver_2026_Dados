"""Inspeção inicial dos dados brutos (CM 2.1) e controle de alterações (CM 3.1).

Quantifica shape, nulos, duplicatas, inconsistências temporais e estatísticas
descritivas direto dos parquets originais via DuckDB, sem carregar os 37M
registros em memória. O resultado é serializado em
outputs/gold/data_inspection.json e consumido por generate_report.py.
"""

import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
TEL_GLOB = str(ROOT / "Base_Dados" / "datasets" / "telemetria" / "*.parquet")
APO_PATH = str(ROOT / "Base_Dados" / "datasets" / "apontamentos" / "desenvolver_apontamentos.parquet")
OUT_PATH = ROOT / "outputs" / "gold" / "data_inspection.json"


def _null_counts(source: str) -> dict[str, int]:
    cols = [r[0] for r in duckdb.execute(f"DESCRIBE SELECT * FROM '{source}'").fetchall()]
    expr = ", ".join(
        f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in cols
    )
    row = duckdb.execute(f"SELECT {expr} FROM '{source}'").fetchone()
    return dict(zip(cols, row))


def inspect_telemetria() -> dict:
    rows, n_tags, dt_min, dt_max = duckdb.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT TAG), MIN(Data_Evento), MAX(Data_Evento)
        FROM '{TEL_GLOB}'
    """).fetchone()
    n_cols = len(duckdb.execute(f"DESCRIBE SELECT * FROM '{TEL_GLOB}'").fetchall())
    nulls = _null_counts(TEL_GLOB)
    dup_ids = duckdb.execute(f"""
        SELECT COUNT(*) - COUNT(DISTINCT Id_Eventos_Telemetria) FROM '{TEL_GLOB}'
    """).fetchone()[0]
    # Valor chega como string BR; linhas com a string literal 'NULL' não são
    # conversíveis para número e viram nulo real no cast Bronze (strict=False).
    valor_null_str, = duckdb.execute(f"""
        SELECT SUM(CASE WHEN TRY_CAST(REPLACE(Valor, ',', '.') AS DOUBLE) IS NULL
                        AND Valor IS NOT NULL THEN 1 ELSE 0 END)
        FROM '{TEL_GLOB}'
    """).fetchone()
    valor_stats = duckdb.execute(f"""
        SELECT MIN(v), MAX(v), AVG(v), MEDIAN(v), STDDEV(v)
        FROM (SELECT TRY_CAST(REPLACE(Valor, ',', '.') AS DOUBLE) AS v FROM '{TEL_GLOB}')
        WHERE v IS NOT NULL
    """).fetchone()
    return {
        "rows": rows,
        "cols": n_cols,
        "n_tags": n_tags,
        "period": [str(dt_min), str(dt_max)],
        "null_cells_total": int(sum(nulls.values())),
        "dup_ids": int(dup_ids),
        "valor_null_str_rows": int(valor_null_str),
        "valor_stats": {
            "min": valor_stats[0], "max": valor_stats[1], "mean": valor_stats[2],
            "median": valor_stats[3], "std": valor_stats[4],
        },
    }


def inspect_apontamentos() -> dict:
    rows, n_tags, dt_min, dt_max = duckdb.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT Tag), MIN(Inicio), MAX(Fim) FROM '{APO_PATH}'
    """).fetchone()
    n_cols = len(duckdb.execute(f"DESCRIBE SELECT * FROM '{APO_PATH}'").fetchall())
    nulls = _null_counts(APO_PATH)
    inicio_gt_fim, dur_zero, dup_ids = duckdb.execute(f"""
        SELECT
            SUM(CASE WHEN Inicio > Fim THEN 1 ELSE 0 END),
            SUM(CASE WHEN Inicio = Fim THEN 1 ELSE 0 END),
            COUNT(*) - COUNT(DISTINCT Id)
        FROM '{APO_PATH}'
    """).fetchone()
    overlaps, = duckdb.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT CASE WHEN Inicio < LAG(Fim) OVER (PARTITION BY Tag ORDER BY Inicio, Id)
                        THEN 1 END AS x
            FROM '{APO_PATH}'
        ) WHERE x = 1
    """).fetchone()
    dur_stats = duckdb.execute(f"""
        SELECT MIN(d), MAX(d), AVG(d), MEDIAN(d), STDDEV(d)
        FROM (SELECT date_diff('second', Inicio, Fim) / 60.0 AS d FROM '{APO_PATH}')
    """).fetchone()
    return {
        "rows": rows,
        "cols": n_cols,
        "n_tags": n_tags,
        "period": [str(dt_min), str(dt_max)],
        "null_cells_total": int(sum(nulls.values())),
        "inicio_gt_fim": int(inicio_gt_fim),
        "dur_zero": int(dur_zero),
        "dup_ids": int(dup_ids),
        "overlapping_cycles": int(overlaps),
        "dur_stats_min": {
            "min": dur_stats[0], "max": dur_stats[1], "mean": dur_stats[2],
            "median": dur_stats[3], "std": dur_stats[4],
        },
    }


def main() -> None:
    payload = {
        "telemetria": inspect_telemetria(),
        "apontamentos": inspect_apontamentos(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✔ Inspeção salva em: {OUT_PATH}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
