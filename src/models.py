"""Gold → Outputs: treinamento, avaliação e explicabilidade do modelo preditivo."""

from pathlib import Path
import pickle
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from features import build_feature_matrix, get_feature_columns

GOLD_DIR = Path(__file__).parent.parent / "outputs" / "gold"
FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"
REPORTS_DIR = Path(__file__).parent.parent / "outputs" / "reports"

TARGET_COL = "is_dont_go_next_60m"

# Meses em ordem cronológica — split temporal estrito
TRAIN_MONTHS = {"jan", "feb", "mar", "abr"}
VAL_MONTHS = {"may"}
TEST_MONTHS = {"jun"}

# Parâmetros base LightGBM (ajustados pelo scale_pos_weight dos dados)
_LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.02,   # menor LR → mais iterações úteis antes de saturar
    "n_estimators": 600,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ── Carregamento com controle de memória ──────────────────────────────────────

def _load_split(
    gold_files: list[Path],
    feature_cols: list[str],
    month_set: set[str],
    target_col: str = TARGET_COL,
    neg_sample_ratio: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega um split em float32 com undersampling opcional dos negativos.

    Args:
        neg_sample_ratio: se não-None, amostra N negativos por positivo no split.
                          Use apenas no treino para caber em 8 GB de RAM.
    """
    needed = feature_cols + [target_col]
    files = [f for f in gold_files if any(m in f.stem for m in month_set)]
    df = pl.read_parquet([str(f) for f in files], columns=needed)

    if neg_sample_ratio is not None:
        # cast para Int8 para comparação segura (coluna pode ser Boolean ou Int8)
        _target = pl.col(target_col).cast(pl.Int8)
        pos = df.filter(_target == 1)
        neg_all = df.filter(_target == 0)
        n_neg_sample = min(len(pos) * neg_sample_ratio, len(neg_all))
        neg = neg_all.sample(n=n_neg_sample, seed=42)
        del neg_all
        df = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=42)
        print(f"  undersampled {neg_sample_ratio}:1 → {len(pos):,} pos + {len(neg):,} neg = {len(df):,} total")
        del pos, neg

    # cast Float32 no Polars antes do to_pandas() — evita intermediário float64 (~10 GB para 24M linhas)
    X = df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_pandas()
    y = df[target_col].cast(pl.Int8).to_pandas()
    del df
    return X, y


# ── Split temporal ────────────────────────────────────────────────────────────

def temporal_train_test_split(
    gold_files: list[Path],
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    neg_sample_ratio: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Carrega cada split diretamente dos parquets mensais em float32.

    Split temporal estrito: Jan-Abr = treino, Mai = validação, Jun = teste.
    O treino aplica undersampling dos negativos (neg_sample_ratio:1) para caber
    em RAM. Ratio real do gold é 1:77, então ratio=5 dá ~1.8M linhas no treino.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (pandas float32, para LightGBM)
    """
    print("Treino (Jan-Abr):")
    X_train, y_train = _load_split(gold_files, feature_cols, TRAIN_MONTHS, target_col, neg_sample_ratio)
    print(f"  {len(X_train):>9,} registros | positivos: {y_train.sum():,} ({y_train.mean():.2%})")

    print("Validação (Mai):")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS, target_col, None)
    print(f"  {len(X_val):>9,} registros | positivos: {y_val.sum():,} ({y_val.mean():.2%})")

    print("Teste (Jun):")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS, target_col, None)
    print(f"  {len(X_test):>9,} registros | positivos: {y_test.sum():,} ({y_test.mean():.2%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Treinamento ───────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 100,
    scale_pos_weight: float | None = None,
) -> lgb.LGBMClassifier:
    """Treina LightGBM com scale_pos_weight explícito ou calculado dos dados reais.

    Quando o treino já foi subamostrado (neg_sample_ratio), NÃO recalcular
    scale_pos_weight dos dados subamostrados — isso causa duplo rebalanceamento
    e faz o early stopping parar em poucas iterações (distribuição treino ≠ val).
    Passe scale_pos_weight=1.0 para treinos subamostrados.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if scale_pos_weight is None:
        spw = n_neg / max(n_pos, 1)
    else:
        spw = scale_pos_weight
    print(f"scale_pos_weight = {spw:.1f}  (neg={n_neg:,} / pos={n_pos:,})")

    params = {**_LGB_PARAMS, "scale_pos_weight": spw}
    model = lgb.LGBMClassifier(**params)

    # LR=0.02: AUC no val satura ~300 iterações (vs 4-19 com LR=0.05).
    # early_stopping_rounds=100 para ao redor de iteração 300+100=400.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"Modelo treinado com {model.best_iteration_} iterações")
    return model


# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Calcula métricas de classificação para o conjunto de teste.

    Returns:
        dict com f1, precision, recall, roc_auc, pr_auc, confusion_matrix.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
    }

    print("\n=== Métricas no Conjunto de Teste (Jun/2025) ===")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sem DG', 'Com DG'], zero_division=0)}")

    return metrics


def find_best_threshold(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
) -> float:
    """Otimiza o threshold de classificação usando o conjunto de validação.

    Busca o threshold que maximiza o F1-Score (ou Recall) no conjunto de validação.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)

    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    best_threshold = float(thresholds[best_idx])
    print(f"Melhor threshold ({metric}={scores[best_idx]:.4f}) no val: {best_threshold:.2f}")
    return best_threshold


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model: lgb.LGBMClassifier,
    X_sample: pd.DataFrame,
    max_samples: int = 10_000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Calcula SHAP values para uma amostra do dataset de teste.

    Args:
        model: modelo LightGBM treinado.
        X_sample: DataFrame de features (amostragem automática se > max_samples).
        max_samples: tamanho máximo da amostra para SHAP (explainer é O(n)).

    Returns:
        (shap_values array, X_sample DataFrame usado)
    """
    import shap

    if len(X_sample) > max_samples:
        X_sample = X_sample.sample(n=max_samples, random_state=42)
        print(f"SHAP: amostrado {max_samples:,} registros para eficiência")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # TreeExplainer para binário retorna lista [class0, class1] ou array 3D
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    mean_abs = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=X_sample.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)

    print("\nTop-10 features por importância SHAP:")
    for feat, val in mean_abs.head(10).items():
        print(f"  {feat:<35} {val:.4f}")

    return shap_vals, X_sample


# ── Persistência ──────────────────────────────────────────────────────────────

def save_model(model: lgb.LGBMClassifier, name: str = "lgbm_dontgo") -> Path:
    """Salva o modelo treinado em outputs/gold/{name}.pkl."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo salvo em {path}")
    return path


def load_model(name: str = "lgbm_dontgo") -> lgb.LGBMClassifier:
    """Carrega modelo de outputs/gold/{name}.pkl."""
    path = GOLD_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_pipeline(
    silver_months: list[str] | None = None,
    rebuild_gold: bool = False,
    neg_sample_ratio: int = 20,
) -> dict:
    """Orquestra todo o pipeline ML: Gold → treino → avaliação → SHAP → salvamento.

    Carrega os splits sequencialmente e libera memória entre etapas para operar
    em máquinas com 16 GB de RAM.

    Args:
        silver_months: meses a incluir. None = todos os 6.
        rebuild_gold: força re-geração do gold mesmo que o arquivo já exista.
        neg_sample_ratio: negativos por positivo no conjunto de treino (undersampling).
                          Ratio real do gold é 1:77; padrão 20 → ~8M linhas no treino.
                          scale_pos_weight residual = 77/neg_sample_ratio é aplicado.

    Returns:
        dict com model, metrics, shap_values, feature_cols, threshold, df (Polars).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gold dataset — verifica arquivos mensais (gold_{mes}.parquet)
    from features import SILVER_DIR
    if silver_months is None:
        expected_months = [f.stem.replace("silver_", "") for f in sorted(SILVER_DIR.glob("silver_*.parquet"))]
    else:
        expected_months = silver_months

    gold_files = [GOLD_DIR / f"gold_{m}.parquet" for m in expected_months]
    gold_ready = all(f.exists() for f in gold_files) and not rebuild_gold

    if not gold_ready:
        missing = [f.name for f in gold_files if not f.exists()]
        print(f"Gerando gold dataset ({len(missing)} meses ausentes: {missing})...")
        lf, top_alarm_ids = build_feature_matrix(silver_months=silver_months, save=True)
        del lf
    else:
        print(f"Gold existente ({len(gold_files)} meses).")

    # 2. Feature columns — deduzidas do schema sem carregar dados
    _schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(_schema)
    del _schema
    print(f"Features selecionadas: {len(feature_cols)}")

    # 3. Treino — dados completos (23M linhas) sem undersampling
    # Distribution shift entre treino e val é a causa do early stopping prematuro.
    # Float32 otimizado: pico ~7 GB (seguro com 10 GB disponíveis).
    print("\n--- Treino (Jan-Abr) ---")
    X_train, y_train = _load_split(gold_files, feature_cols, TRAIN_MONTHS, neg_sample_ratio=None)
    print(f"  {len(X_train):>9,} registros | pos={y_train.sum():,} ({y_train.mean():.2%})")

    # 4. Validação
    print("\n--- Validação (Mai) ---")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS)
    print(f"  {len(X_val):>9,} registros | pos={y_val.sum():,} ({y_val.mean():.2%})")

    # 5. Treino LightGBM — scale_pos_weight calculado do ratio real (~59)
    # Modelo converge em ~5 árvores; ROC-AUC = 0.99 reflete discriminação excelente.
    # Para melhorar F1 além de 0.68: necessário adicionar features no features.py.
    print("\n--- Treinamento LightGBM ---")
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    del X_train, y_train

    # 6. Threshold ótimo no val set
    best_threshold = find_best_threshold(model, X_val, y_val, metric="f1")
    del X_val, y_val

    # 7. Teste
    print("\n--- Teste (Jun) ---")
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    print(f"  {len(X_test):>9,} registros | pos={y_test.sum():,} ({y_test.mean():.2%})")

    # 8. Avaliação
    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)

    # 9. SHAP — enquanto X_test ainda está em memória
    print("\n--- SHAP Values ---")
    shap_vals, X_shap = compute_shap_values(model, X_test)
    del X_test, y_test

    # 10. Salvar modelo
    save_model(model)

    # 11. Carrega jun com colunas mínimas para visualizações do notebook 05
    # Não inclui feature_cols — o notebook as carrega separadamente com cast Float32
    _test_files = [f for f in gold_files if "jun" in f.stem]
    df = pl.read_parquet([str(f) for f in _test_files], columns=[TARGET_COL, "Data_Evento", "TAG"])

    return {
        "model": model,
        "metrics": metrics,
        "shap_values": shap_vals,
        "X_shap": X_shap,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "df": df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Baselines e modelos alternativos (CM 4.2-4.3 — comparação multi-família)
# ══════════════════════════════════════════════════════════════════════════════

def _metrics_from_predictions(y_true, y_pred, y_prob=None, name: str = "baseline") -> dict:
    """Compõe dict padronizado de métricas a partir de predições."""
    out = {
        "model": name,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": len(y_true),
        "n_positive": int(np.asarray(y_true).sum()),
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def evaluate_majority_baseline(y_test: pd.Series) -> dict:
    """Baseline trivial: sempre prever a classe majoritária (0 = Não-DontGo).

    Mostra o piso de comparação. Em desbalanceamento 1:77, acerta 98.7% por
    sorte mas tem F1=0, expondo a inutilidade de accuracy como métrica.
    """
    y_pred = np.zeros(len(y_test), dtype=int)
    metrics = _metrics_from_predictions(y_test, y_pred, y_prob=None, name="majority_class")
    print(f"[Baseline naive — majority class]  F1={metrics['f1']:.4f} | Acc≈{1 - y_test.mean():.4f}")
    return metrics


def evaluate_rule_baseline(
    df_test: pd.DataFrame,
    y_test: pd.Series,
    rule_col: str = "n_criticos_30m",
    threshold: int = 3,
) -> dict:
    """Baseline de regra de negócio: prever DontGo se >= N alarmes críticos em 30min.

    Codifica a heurística tradicional usada por operação antes do modelo ML.
    Justifica o investimento em ML — superar essa regra é a hipótese H0.
    """
    if rule_col not in df_test.columns:
        raise KeyError(f"Coluna '{rule_col}' não encontrada em df_test")
    y_pred = (df_test[rule_col].fillna(0).values >= threshold).astype(int)
    # Score "probabilístico" derivado da contagem normalizada — só para AUC
    y_prob = (df_test[rule_col].fillna(0).clip(0, 20).values / 20.0)
    name = f"rule_n{rule_col}_ge_{threshold}"
    metrics = _metrics_from_predictions(y_test, y_pred, y_prob=y_prob, name=name)
    metrics["rule_col"] = rule_col
    metrics["rule_threshold"] = threshold
    print(
        f"[Baseline regra — {rule_col} >= {threshold}]  "
        f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )
    return metrics


def find_best_rule_threshold(
    df_val: pd.DataFrame,
    y_val: pd.Series,
    rule_col: str = "n_criticos_30m",
    n_range: range | None = None,
) -> int:
    """Otimiza o threshold N da regra de negócio no conjunto de validação."""
    n_range = n_range or range(1, 21)
    best_n, best_f1 = 1, -1.0
    for n in n_range:
        y_pred = (df_val[rule_col].fillna(0).values >= n).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_n = f1, n
    print(f"Regra ótima no val: {rule_col} >= {best_n}  (F1={best_f1:.4f})")
    return best_n


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float | None = None,
    C: float = 0.1,
    max_iter: int = 1000,
) -> Pipeline:
    """Logistic Regression com regularização L1 + StandardScaler.

    Família linear — contrasta com LightGBM (boosting) e Random Forest (bagging).
    Penalty L1 atua como seleção de features automática.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if scale_pos_weight is None:
        spw = n_neg / max(n_pos, 1)
    else:
        spw = scale_pos_weight
    class_weight = {0: 1.0, 1: float(spw)}
    print(f"[LogReg L1] class_weight={class_weight} | C={C}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            solver="saga",
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train.fillna(0), y_train)
    return pipe


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 20,
    scale_pos_weight: float | None = None,
) -> RandomForestClassifier:
    """Random Forest — família bagging, contraste com boosting (LightGBM).

    Usa class_weight balanceado para corrigir desbalanceamento.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    if scale_pos_weight is None:
        spw = n_neg / max(n_pos, 1)
    else:
        spw = scale_pos_weight
    class_weight = {0: 1.0, 1: float(spw)}
    print(f"[RandomForest] n_estimators={n_estimators} | max_depth={max_depth} | class_weight={class_weight}")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=50,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train.fillna(0), y_train)
    return rf


def evaluate_sklearn_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    name: str = "sklearn_model",
) -> dict:
    """Avaliação padronizada para modelos sklearn (pipeline ou estimator)."""
    y_prob = model.predict_proba(X_test.fillna(0))[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = _metrics_from_predictions(y_test, y_pred, y_prob=y_prob, name=name)
    metrics["threshold"] = threshold
    print(
        f"[{name}]  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
        f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}"
    )
    return metrics


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Tabela comparativa Markdown-ready de métricas de modelos.

    Args:
        results: lista de dicts retornados por evaluate_* / _metrics_from_predictions.
    Returns:
        DataFrame ordenado por F1 desc, colunas: model, F1, Precision, Recall, ROC-AUC, PR-AUC.
    """
    rows = []
    for r in results:
        rows.append({
            "Modelo": r.get("model", "?"),
            "F1": r.get("f1"),
            "Precision": r.get("precision"),
            "Recall": r.get("recall"),
            "ROC-AUC": r.get("roc_auc"),
            "PR-AUC": r.get("pr_auc"),
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação multi-horizonte (Sprint 3 — BONUS-1: antecedência 1-4h)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_at_horizon(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test_by_horizon: dict[int, pd.Series],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Avalia o mesmo modelo contra targets de horizontes diferentes (60/120/240min).

    Args:
        model: modelo treinado com target_60m.
        X_test: features de teste (mesmas usadas no treino).
        y_test_by_horizon: dict {horizonte_min: y_test_serie}.
    Returns:
        DataFrame com F1/Precision/Recall/PR-AUC por horizonte.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for h, y_h in y_test_by_horizon.items():
        rows.append({
            "horizonte_min": h,
            "F1": f1_score(y_h, y_pred, zero_division=0),
            "Precision": precision_score(y_h, y_pred, zero_division=0),
            "Recall": recall_score(y_h, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_h, y_prob),
            "PR-AUC": average_precision_score(y_h, y_prob),
            "n_positivos": int(y_h.sum()),
        })
    df = pd.DataFrame(rows).sort_values("horizonte_min").reset_index(drop=True)
    return df
