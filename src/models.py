"""Gold → Outputs: treinamento, avaliação e explicabilidade do modelo preditivo."""

from pathlib import Path
import pickle
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
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
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ── Split temporal ────────────────────────────────────────────────────────────

def temporal_train_test_split(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Divide o gold dataset em treino (Jan-Abr), validação (Mai) e teste (Jun).

    O split é feito estritamente pelo mês do campo Data_Evento, garantindo que
    dados futuros nunca influenciem o treino.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (pandas, para LightGBM)
    """
    month_col = df["Data_Evento"].dt.month()

    mask_train = month_col.is_in([1, 2, 3, 4])
    mask_val = month_col.is_in([5])
    mask_test = month_col.is_in([6])

    def _split(mask: pl.Series) -> tuple[pd.DataFrame, pd.Series]:
        subset = df.filter(mask)
        X = subset.select(feature_cols).to_pandas()
        y = subset[target_col].cast(pl.Int8).to_pandas()
        return X, y

    X_train, y_train = _split(mask_train)
    X_val, y_val = _split(mask_val)
    X_test, y_test = _split(mask_test)

    print(f"Treino  (Jan-Abr): {len(X_train):>9,} registros | positivos: {y_train.sum():,} ({y_train.mean():.2%})")
    print(f"Validação   (Mai): {len(X_val):>9,} registros | positivos: {y_val.sum():,} ({y_val.mean():.2%})")
    print(f"Teste        (Jun): {len(X_test):>9,} registros | positivos: {y_test.sum():,} ({y_test.mean():.2%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Treinamento ───────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMClassifier:
    """Treina LightGBM com scale_pos_weight calculado dos dados de treino.

    Usa early stopping no conjunto de validação para evitar overfitting.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {spw:.1f}  (neg={n_neg:,} / pos={n_pos:,})")

    params = {**_LGB_PARAMS, "scale_pos_weight": spw}
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
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
) -> dict:
    """Orquestra todo o pipeline ML: Gold → treino → avaliação → SHAP → salvamento.

    Args:
        silver_months: meses a incluir. None = todos os 6.
        rebuild_gold: força re-geração do gold mesmo que o arquivo já exista.

    Returns:
        dict com model, metrics, shap_values, feature_cols, threshold.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gold dataset
    gold_path = GOLD_DIR / "gold_features.parquet"
    if gold_path.exists() and not rebuild_gold:
        print(f"Carregando gold existente: {gold_path}")
        df, top_alarm_ids = pl.read_parquet(str(gold_path)), None
    else:
        print("Gerando gold dataset...")
        df, top_alarm_ids = build_feature_matrix(silver_months=silver_months, save=True)

    print(f"Gold: {len(df):,} registros × {len(df.columns)} colunas")

    # 2. Feature columns
    feature_cols = get_feature_columns(df)
    print(f"Features selecionadas: {len(feature_cols)}")

    # 3. Split temporal
    print("\n--- Split Temporal ---")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_test_split(
        df, feature_cols
    )

    # 4. Treino
    print("\n--- Treinamento LightGBM ---")
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    # 5. Threshold ótimo no val set
    best_threshold = find_best_threshold(model, X_val, y_val, metric="f1")

    # 6. Avaliação no test set
    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)

    # 7. SHAP
    print("\n--- SHAP Values ---")
    shap_vals, X_shap = compute_shap_values(model, X_test)

    # 8. Salvar modelo
    save_model(model)

    return {
        "model": model,
        "metrics": metrics,
        "shap_values": shap_vals,
        "X_shap": X_shap,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "df": df,
    }
