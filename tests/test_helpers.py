"""Testes de funções puras de apoio do projeto Don't Go Predictor.

Cobrem utilitários determinísticos que NÃO dependem dos parquets de dados:
formatação BR de números, detector de drift Page-Hinkley, busca vetorizada de
threshold F1-ótimo, baselines de regra e seleção de colunas de features.

Executar: `uv run pytest -q`  (pythonpath=src configurado no pyproject.toml)
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from generate_report import _brl_num
from sprint8_drift_detection import page_hinkley
from sprint10_ensemble import _best_f1_threshold
from models import (
    evaluate_rule_baseline,
    find_best_rule_threshold,
    build_comparison_table,
)
from features import get_feature_columns


# ── _brl_num — formatação numérica BR (milhar '.', decimal ',') ──────────────

@pytest.mark.parametrize("valor,digits,sign,esperado", [
    (1234.5, 1, False, "1.234,5"),
    (0.6886, 4, False, "0,6886"),
    (343337200 / 1e6, 1, False, "343,3"),
    (1.2, 1, True, "+1,2"),
    (-3.0, 1, True, "-3,0"),
    (0.51, 2, False, "0,51"),
])
def test_brl_num_formats(valor, digits, sign, esperado):
    assert _brl_num(valor, digits, sign=sign) == esperado


def test_brl_num_none_returns_str():
    # valores não-numéricos caem no fallback str(v) sem levantar exceção
    assert _brl_num(None) == "None"


# ── page_hinkley — detector de drift ────────────────────────────────────────

def test_page_hinkley_serie_estavel_nao_detecta():
    serie = np.zeros(50, dtype=float)
    res = page_hinkley(serie)
    assert res["detected_at"] == []


def test_page_hinkley_detecta_salto():
    serie = np.concatenate([np.zeros(30), np.full(30, 10.0)])
    res = page_hinkley(serie)
    assert len(res["detected_at"]) > 0
    # a detecção só pode ocorrer após o salto (índice >= 30)
    assert min(res["detected_at"]) >= 30


def test_page_hinkley_serie_curta():
    res = page_hinkley(np.array([1.0]))
    assert res["detected_at"] == []


# ── _best_f1_threshold — busca vetorizada (sort + cumsum) ────────────────────

def test_best_f1_threshold_separavel():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    thr, f1 = _best_f1_threshold(y, p)
    assert f1 == pytest.approx(1.0)
    # no threshold ótimo a predição reproduz exatamente y
    assert np.array_equal((p >= thr).astype(int), y)


def test_best_f1_threshold_sem_positivos():
    y = np.zeros(10, dtype=int)
    p = np.random.default_rng(0).random(10)
    thr, f1 = _best_f1_threshold(y, p)
    assert (thr, f1) == (0.5, 0.0)


def test_best_f1_threshold_bate_forca_bruta():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    if y.sum() == 0:
        pytest.skip("sem positivos")
    thr, f1 = _best_f1_threshold(y, p)
    # força-bruta sobre todos os cortes candidatos
    from sklearn.metrics import f1_score
    melhor = max(f1_score(y, (p >= t).astype(int), zero_division=0) for t in np.unique(p))
    assert f1 == pytest.approx(melhor, abs=1e-9)


# ── baselines de regra de negócio (models.py) ───────────────────────────────

def test_evaluate_rule_baseline_separacao_perfeita():
    df = pd.DataFrame({"n_criticos_30m": [0, 0, 5, 5]})
    y = pd.Series([0, 0, 1, 1])
    m = evaluate_rule_baseline(df, y, rule_col="n_criticos_30m", threshold=3)
    assert m["f1"] == pytest.approx(1.0)
    assert m["model"] == "rule_nn_criticos_30m_ge_3"


def test_evaluate_rule_baseline_coluna_ausente():
    df = pd.DataFrame({"outra": [1, 2]})
    with pytest.raises(KeyError):
        evaluate_rule_baseline(df, pd.Series([0, 1]), rule_col="n_criticos_30m")


def test_find_best_rule_threshold_escolhe_n_otimo():
    df = pd.DataFrame({"n_criticos_30m": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])
    # apenas N=3 separa perfeitamente (>=3 → [0,0,1,1])
    assert find_best_rule_threshold(df, y, rule_col="n_criticos_30m") == 3


# ── build_comparison_table ───────────────────────────────────────────────────

def test_build_comparison_table_ordena_por_f1_desc():
    results = [
        {"model": "A", "f1": 0.30, "precision": 0.4, "recall": 0.5, "roc_auc": 0.8, "pr_auc": 0.2},
        {"model": "B", "f1": 0.70, "precision": 0.7, "recall": 0.6, "roc_auc": 0.9, "pr_auc": 0.6},
    ]
    tab = build_comparison_table(results)
    assert list(tab.columns) == ["Modelo", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
    assert tab.iloc[0]["Modelo"] == "B"  # maior F1 primeiro
    assert tab.iloc[-1]["Modelo"] == "A"


# ── get_feature_columns (features.py) ────────────────────────────────────────

def _df_schema_sintetico() -> pl.DataFrame:
    return pl.DataFrame({
        "TAG": ["x"],                      # excluída (id)
        "Id_Alarme": [1],                  # excluída (id)
        "Data_Evento": [None],             # excluída (prefixo Data_)
        "is_dont_go_next_60m": [0],        # excluída (target)
        "minutes_to_next_dg": [10],        # excluída (prefixo)
        "n_criticos_30m": [2],             # FEATURE
        "hora": [13],                      # FEATURE
        "fp_alarm_123": [1],               # FEATURE (fingerprint)
    })


def test_get_feature_columns_inclui_fingerprint():
    cols = get_feature_columns(_df_schema_sintetico(), include_fingerprint=True)
    assert cols == ["n_criticos_30m", "hora", "fp_alarm_123"]


def test_get_feature_columns_exclui_fingerprint():
    cols = get_feature_columns(_df_schema_sintetico(), include_fingerprint=False)
    assert cols == ["n_criticos_30m", "hora"]
    assert all(not c.startswith("fp_alarm_") for c in cols)
