"""Sprint 10 — Ensemble Random Forest + LightGBM.

Motivação: o Sprint 1 mostrou que Random Forest e LightGBM empatam tecnicamente
(F1=0.6741 ambos), mas erram de formas diferentes (árvores independentes vs.
boosting sequencial). O F1 do benchmark (~0,67) parece ser o TETO para o
feature set atual — testar um ensemble é a tentativa metodologicamente limpa de
romper esse teto SEM adicionar novas features (que causam distribution shift).

Três estratégias de combinação, todas com threshold F1-ótimo calibrado na
validação (Mai) e avaliadas no teste (Jun):
  1. Soft-voting        — média simples das probabilidades.
  2. Média ponderada    — peso w∈[0,1] otimizado na validação.
  3. Stacking           — meta-learner (LogisticRegression) sobre [p_rf, p_lgbm],
                          treinado na validação e avaliado no teste.

Hipótese nula honesta: se o teto é o feature set (não o algoritmo), o ensemble
NÃO supera o melhor modelo-base de forma relevante. Documentamos o resultado seja
qual for — ganho marginal é informação tão valiosa quanto ganho expressivo.

Salva: outputs/gold/ensemble.json + figuras em outputs/figures/.
"""

from pathlib import Path
import json
import pickle
import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).parent))
from models import (
    _load_split, load_model, train_random_forest,
    TRAIN_MONTHS, VAL_MONTHS, TEST_MONTHS,
)
from features import get_feature_columns

ROOT = Path(__file__).parent.parent
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Threshold que maximiza F1 — vetorizado via sort + cumsum (O(n log n)).

    Evita chamar sklearn.f1_score centenas de vezes sobre arrays de milhões de
    linhas (gargalo da versão ingênua, sobretudo no grid de média ponderada).
    """
    y_true = y_true.astype(np.int32)
    order = np.argsort(-y_prob, kind="stable")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.5, 0.0
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    i = int(np.argmax(f1))
    return float(p_sorted[i]), float(f1[i])


def _metrics_at(y_true, y_prob, thr) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
    }


def main():
    gold_files = sorted(GOLD_DIR.glob("gold_*.parquet"))
    schema = pl.read_parquet(str(gold_files[0]), n_rows=0)
    feature_cols = get_feature_columns(schema)
    print(f"Features (gold): {len(feature_cols)}")

    # ── Splits ───────────────────────────────────────────────────────────────
    print("\n== Carregando splits ==")
    X_tr, y_tr = _load_split(gold_files, feature_cols, TRAIN_MONTHS, neg_sample_ratio=5)
    print(f"  treino (5:1): {len(X_tr):,} | {y_tr.sum():,} pos")
    X_val, y_val = _load_split(gold_files, feature_cols, VAL_MONTHS)
    X_test, y_test = _load_split(gold_files, feature_cols, TEST_MONTHS)
    y_val = y_val.to_numpy(); y_test = y_test.to_numpy()
    print(f"  val (Mai): {len(X_val):,} | {int(y_val.sum()):,} pos")
    print(f"  test (Jun): {len(X_test):,} | {int(y_test.sum()):,} pos")

    # ── Modelos-base ──────────────────────────────────────────────────────────
    print("\n== LightGBM (modelo principal salvo) ==")
    lgbm = load_model("lgbm_dontgo")
    lgbm_feats = list(getattr(lgbm, "feature_name_", feature_cols))
    for c in lgbm_feats:
        if c not in X_val.columns:
            X_val[c] = 0.0; X_test[c] = 0.0
    p_lgbm_val = lgbm.predict_proba(X_val[lgbm_feats].fillna(0))[:, 1]
    p_lgbm_test = lgbm.predict_proba(X_test[lgbm_feats].fillna(0))[:, 1]

    print("== Random Forest (retreino reproduzível, params Sprint 1) ==")
    rf = train_random_forest(X_tr, y_tr, n_estimators=200, max_depth=20, scale_pos_weight=5.0)
    p_rf_val = rf.predict_proba(X_val[feature_cols].fillna(0))[:, 1]
    p_rf_test = rf.predict_proba(X_test[feature_cols].fillna(0))[:, 1]

    # ── Baselines (modelos isolados) ───────────────────────────────────────────
    print("\n== Baselines isolados (threshold F1-ótimo na val) ==")
    results = {}
    for name, pv, pt in [("LightGBM", p_lgbm_val, p_lgbm_test),
                         ("RandomForest", p_rf_val, p_rf_test)]:
        thr, f1v = _best_f1_threshold(y_val, pv)
        results[name] = _metrics_at(y_test, pt, thr)
        print(f"  {name:<14} thr={thr:.3f}  F1_val={f1v:.4f}  F1_test={results[name]['F1']:.4f}")

    best_base = max(results, key=lambda k: results[k]["F1"])
    best_base_f1 = results[best_base]["F1"]

    # ── Ensemble 1: soft-voting (média simples) ─────────────────────────────────
    print("\n== Ensemble 1: Soft-voting (média simples) ==")
    sv_val = 0.5 * p_rf_val + 0.5 * p_lgbm_val
    sv_test = 0.5 * p_rf_test + 0.5 * p_lgbm_test
    thr, f1v = _best_f1_threshold(y_val, sv_val)
    results["Soft-voting"] = _metrics_at(y_test, sv_test, thr)
    print(f"  thr={thr:.3f}  F1_val={f1v:.4f}  F1_test={results['Soft-voting']['F1']:.4f}")

    # ── Ensemble 2: média ponderada (peso otimizado na val) ─────────────────────
    print("\n== Ensemble 2: Média ponderada (otimiza peso na val) ==")
    weights = np.linspace(0, 1, 51)  # w = peso do LightGBM
    best_w, best_wf1, best_wthr = 0.5, -1.0, 0.5
    for w in weights:
        pv = w * p_lgbm_val + (1 - w) * p_rf_val
        thr, f1v = _best_f1_threshold(y_val, pv)
        if f1v > best_wf1:
            best_wf1, best_w, best_wthr = f1v, float(w), thr
    wa_test = best_w * p_lgbm_test + (1 - best_w) * p_rf_test
    results["Média ponderada"] = _metrics_at(y_test, wa_test, best_wthr)
    results["Média ponderada"]["weight_lgbm"] = best_w
    print(f"  w_lgbm*={best_w:.2f}  thr={best_wthr:.3f}  F1_val={best_wf1:.4f}  "
          f"F1_test={results['Média ponderada']['F1']:.4f}")

    # ── Ensemble 3: stacking (meta-learner logístico na val) ─────────────────────
    print("\n== Ensemble 3: Stacking (LogisticRegression meta na val) ==")
    Z_val = np.column_stack([p_rf_val, p_lgbm_val])
    Z_test = np.column_stack([p_rf_test, p_lgbm_test])
    meta = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    meta.fit(Z_val, y_val)
    st_val = meta.predict_proba(Z_val)[:, 1]
    st_test = meta.predict_proba(Z_test)[:, 1]
    thr, f1v = _best_f1_threshold(y_val, st_val)
    results["Stacking"] = _metrics_at(y_test, st_test, thr)
    results["Stacking"]["meta_coef"] = {"rf": float(meta.coef_[0][0]),
                                        "lgbm": float(meta.coef_[0][1]),
                                        "intercept": float(meta.intercept_[0])}
    print(f"  coef[rf={meta.coef_[0][0]:.3f}, lgbm={meta.coef_[0][1]:.3f}]  "
          f"thr={thr:.3f}  F1_val={f1v:.4f}  F1_test={results['Stacking']['F1']:.4f}")
    print("  ⚠ stacking treina o meta na MESMA val usada p/ threshold → leve viés otimista")

    # ── Resumo ──────────────────────────────────────────────────────────────────
    best_ens = max(["Soft-voting", "Média ponderada", "Stacking"], key=lambda k: results[k]["F1"])
    delta = results[best_ens]["F1"] - best_base_f1
    print("\n" + "═" * 64)
    print(f"  Melhor modelo-base : {best_base} (F1={best_base_f1:.4f})")
    print(f"  Melhor ensemble    : {best_ens} (F1={results[best_ens]['F1']:.4f})")
    print(f"  ► Δ F1 ensemble vs base: {delta:+.4f} ({100*delta/best_base_f1:+.1f}%)")
    print("═" * 64)

    # ── Figura 1: barras F1 ───────────────────────────────────────────────────
    order = ["LightGBM", "RandomForest", "Soft-voting", "Média ponderada", "Stacking"]
    f1s = [results[k]["F1"] for k in order]
    colors = ["#10B981", "#8B5CF6", "#3B82F6", "#F59E0B", "#EF4444"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(order, f1s, color=colors)
    ax.axhline(best_base_f1, color="#10B981", linestyle="--", alpha=0.6,
               label=f"Melhor base ({best_base}) = {best_base_f1:.4f}")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
    ax.set_ylabel("F1-Score (teste Jun/2025)")
    ax.set_title("Sprint 10 — Ensemble RF + LightGBM vs. Modelos-Base\n"
                 "Threshold F1-ótimo calibrado na validação (Mai/2025)")
    ax.set_ylim(0, max(f1s) * 1.18)
    ax.legend()
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ensemble_f1_comparison.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'ensemble_f1_comparison.png'}")

    # ── Figura 2: curvas PR ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, pt, color in [("LightGBM", p_lgbm_test, "#10B981"),
                            ("RandomForest", p_rf_test, "#8B5CF6"),
                            ("Soft-voting", sv_test, "#3B82F6"),
                            ("Stacking", st_test, "#EF4444")]:
        prec, rec, _ = precision_recall_curve(y_test, pt)
        ap = average_precision_score(y_test, pt)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=color, linewidth=2)
    ax.axhline(float(y_test.mean()), color="k", linestyle="--", alpha=0.4,
               label=f"Baseline ({y_test.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall — Ensemble vs. Base (Teste Jun/2025)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ensemble_pr_curves.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'ensemble_pr_curves.png'}")

    # ── Persistência ──────────────────────────────────────────────────────────
    payload = {
        "results": results,
        "best_base_model": best_base,
        "best_base_f1": best_base_f1,
        "best_ensemble": best_ens,
        "best_ensemble_f1": results[best_ens]["F1"],
        "delta_f1": delta,
        "delta_f1_pct": 100 * delta / best_base_f1,
        "ceiling_broken": bool(delta > 0.005),
        "rf_params": {"n_estimators": 200, "max_depth": 20, "scale_pos_weight": 5.0},
    }
    out_json = GOLD_DIR / "ensemble.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(GOLD_DIR / "ensemble_meta_learner.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"\n✓ Resultados salvos em {out_json}")
    print("\n══════════════ SPRINT 10 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
