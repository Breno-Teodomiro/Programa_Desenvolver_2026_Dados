"""SHAP waterfall de uma predição individual (CM 5.3).

Seleciona o verdadeiro positivo de maior convicção do conjunto de teste
(Jun/2025), decompõe a predição feature a feature com SHAP e salva:
- outputs/figures/shap_waterfall.png  (figura para o relatório)
- outputs/gold/shap_waterfall_example.json  (metadados do caso p/ legenda)

Só as linhas positivas do teste são carregadas (~35K), preservando RAM.
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import shap

ROOT = Path(__file__).parent.parent
GOLD_JUN = ROOT / "outputs" / "gold" / "gold_jun.parquet"
MODEL_PATH = ROOT / "outputs" / "gold" / "lgbm_dontgo.pkl"
METRICS_PATH = ROOT / "outputs" / "reports" / "model_metrics.json"
FIG_PATH = ROOT / "outputs" / "figures" / "shap_waterfall.png"
OUT_JSON = ROOT / "outputs" / "gold" / "shap_waterfall_example.json"

TARGET_COL = "is_dont_go_next_60m"


def main() -> None:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    feature_cols = list(model.feature_name_)
    with open(METRICS_PATH) as f:
        threshold = json.load(f)["optimal_threshold"]

    # Apenas os positivos do teste: suficiente para achar um TP e barato em RAM
    df = (
        pl.scan_parquet(str(GOLD_JUN))
        .filter(pl.col(TARGET_COL) == 1)
        .select(feature_cols + ["TAG", "Data_Evento"])
        .collect()
    )
    X = df.select([pl.col(c).cast(pl.Float32) for c in feature_cols]).to_pandas()
    proba = model.predict_proba(X)[:, 1]

    idx = int(proba.argmax())
    if proba[idx] < threshold:
        raise RuntimeError("Nenhum verdadeiro positivo acima do threshold no teste.")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X.iloc[[idx]])[0]
    # Saída binária do LGBM: usa a dimensão da classe positiva se existir
    if explanation.values.ndim == 2:
        explanation = explanation[:, 1]

    plt.figure()
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "tag": df["TAG"][idx],
        "data_evento": str(df["Data_Evento"][idx]),
        "probability": float(proba[idx]),
        "threshold": threshold,
        "n_test_positives": df.height,
        "top_features": [
            {"feature": feature_cols[i], "shap": float(explanation.values[i])}
            for i in abs(explanation.values).argsort()[::-1][:5]
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✔ Waterfall salvo em: {FIG_PATH}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
