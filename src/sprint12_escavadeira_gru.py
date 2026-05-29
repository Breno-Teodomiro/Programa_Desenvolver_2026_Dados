"""Sprint 12 — PoC de modelo sequencial (GRU) para a escavadeira LeTourneau.

Motivação: o LightGBM agregado é cego à escavadeira (Recall=0,14, Sprint 2) e um
modelo dedicado com o mesmo feature set falhou (Sprint 5). O Sprint 11 mostrou que
existem alarmes de altíssimo lift (Channel Forced 44×, PTO>90°C 28×) que o
fingerprint por frequência descarta. Hipótese desta PoC: um modelo SEQUENCIAL,
que enxerga a ORDEM bruta dos alarmes (não a agregação binária top-30), pode
capturar a assinatura temporal que precede o Don't Go na escavadeira — incluindo
naturalmente esses alarmes raros de alto lift, pois todos entram no vocabulário.

Arquitetura (pequena, adequada a CPU): Embedding(alarmes) → GRU → Linear → sigmoid.
Cada ponto de decisão vê a sequência dos K alarmes anteriores (do mesmo equipamento).

Split temporal do projeto: treino jan-abr / val Mai (threshold) / teste Jun.
Avaliação HONESTA no Jun completo (7,35M eventos, apenas 162 positivos — o mesmo
distribution shift severo que derrubou o Sprint 5).

Baseline a superar: LightGBM geral na escavadeira → Recall=0,14, F1=0,015 (Sprint 2).

Salva: outputs/gold/escavadeira_gru.json + figuras + modelo .pt.

Dependência opcional (não está no uv.lock para não bloquear o pipeline principal
com a variante CUDA). Instalar a versão CPU antes de rodar este sprint:
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

from pathlib import Path
import json
import sys
import time
import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

ROOT = Path(__file__).parent.parent
SILVER_GLOB = str(ROOT / "outputs" / "silver" / "silver_*.parquet")
GOLD_DIR = ROOT / "outputs" / "gold"
FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ESCAVADEIRA = "LeTourneau L 1850"
SEQ_LEN = 32          # nº de alarmes anteriores na sequência
EMB_DIM = 24
HIDDEN = 48
EPOCHS = 4
BATCH = 1024
NEG_RATIO_TRAIN = 5   # undersampling de negativos no treino
NEG_RATIO_VAL = 20    # subsample de negativos na val (só p/ calibrar threshold)
TARGET = "is_dont_go_next_60m"

TRAIN_MONTHS = {1, 2, 3, 4}
VAL_MONTH = {5}
TEST_MONTH = {6}


class AlarmGRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)            # h: (1, B, hidden)
        return self.head(h[-1]).squeeze(-1)  # (B,)


def build_sequences():
    """Carrega escavadeira, ordena por TAG/tempo e monta janelas deslizantes de
    alarmes. Treino/val são subamostrados nos negativos; teste é completo."""
    print("== Carregando escavadeira (silver jan-jun) ==")
    df = (
        pl.scan_parquet(SILVER_GLOB)
        .filter(pl.col("Tag_Frota") == ESCAVADEIRA)
        .select([
            "TAG", "Data_Evento", "Id_Alarme",
            pl.col(TARGET).cast(pl.Int8).alias("y"),
            pl.col("Data_Evento").dt.month().alias("mes"),
        ])
        .sort(["TAG", "Data_Evento"])
        .collect()
    )
    print(f"  {len(df):,} eventos | {int(df['y'].sum()):,} positivos")

    # Vocabulário de alarmes (0 = PAD)
    alarms = df["Id_Alarme"].unique().to_list()
    vocab = {a: i + 1 for i, a in enumerate(sorted(alarms))}
    vocab_size = len(vocab) + 1
    print(f"  vocabulário: {len(vocab)} alarmes (+PAD) → vocab_size={vocab_size}")
    df = df.with_columns(pl.col("Id_Alarme").replace_strict(vocab, default=0).alias("aidx"))

    # Para cada TAG: janelas deslizantes (left-pad com PAD=0)
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for tag, g in df.group_by("TAG", maintain_order=True):
        aidx = g["aidx"].to_numpy().astype(np.int32)
        y = g["y"].to_numpy().astype(np.int8)
        mes = g["mes"].to_numpy()
        padded = np.concatenate([np.zeros(SEQ_LEN - 1, dtype=np.int32), aidx])
        sw = sliding_window_view(padded, SEQ_LEN)        # (n, SEQ_LEN) — view

        for name, months, neg_ratio in [
            ("train", TRAIN_MONTHS, NEG_RATIO_TRAIN),
            ("val", VAL_MONTH, NEG_RATIO_VAL),
            ("test", TEST_MONTH, None),
        ]:
            mask = np.isin(mes, list(months))
            pos_idx = np.where(mask & (y == 1))[0]
            neg_idx = np.where(mask & (y == 0))[0]
            if neg_ratio is not None and len(neg_idx) > len(pos_idx) * neg_ratio:
                rng = np.random.default_rng(42)
                neg_idx = rng.choice(neg_idx, size=max(len(pos_idx) * neg_ratio, 1), replace=False)
            sel = np.concatenate([pos_idx, neg_idx])
            splits[name][0].append(sw[sel].copy())
            splits[name][1].append(y[sel])

    out = {}
    for name in splits:
        X = np.concatenate(splits[name][0]).astype(np.int32)
        Y = np.concatenate(splits[name][1]).astype(np.float32)
        # embaralha
        perm = np.random.default_rng(7).permutation(len(X))
        out[name] = (X[perm], Y[perm])
        print(f"  {name}: {len(X):,} seqs | {int(Y.sum()):,} pos ({Y.mean():.3%})")
    return out, vocab_size


def predict_proba(model, X, batch=4096):
    model.eval()
    probs = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).long()
            probs[i:i + batch] = torch.sigmoid(model(xb)).numpy()
    return probs


def _best_f1_threshold(y, p):
    order = np.argsort(-p, kind="stable")
    ys = y[order].astype(np.int64)
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P = int(y.sum())
    if P == 0:
        return 0.5, 0.0
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return float(p[order][i]), float(f1[i])


def main():
    t0 = time.time()
    data, vocab_size = build_sequences()
    Xtr, Ytr = data["train"]; Xval, Yval = data["val"]; Xte, Yte = data["test"]

    model = AlarmGRU(vocab_size, EMB_DIM, HIDDEN)
    pos_w = torch.tensor([(Ytr == 0).sum() / max((Ytr == 1).sum(), 1)], dtype=torch.float32)
    print(f"\n== Treino GRU (pos_weight={pos_w.item():.2f}) ==")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xtr_t = torch.from_numpy(Xtr).long()
    Ytr_t = torch.from_numpy(Ytr)
    n = len(Xtr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            logit = model(Xtr_t[idx])
            loss = crit(logit, Ytr_t[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        # val AUC por época
        pv = predict_proba(model, Xval)
        auc = roc_auc_score(Yval, pv) if Yval.sum() > 0 else float("nan")
        print(f"  época {ep+1}/{EPOCHS}  loss={tot/n:.4f}  val_ROC_AUC={auc:.4f}  ({time.time()-t0:.0f}s)")

    # ── Threshold F1-ótimo na val ──────────────────────────────────────────
    p_val = predict_proba(model, Xval)
    thr, f1v = _best_f1_threshold(Yval, p_val)
    print(f"\n== Threshold F1-ótimo (val Mai): {thr:.4f} (F1_val={f1v:.4f}) ==")

    # ── Avaliação HONESTA no teste Jun completo ────────────────────────────
    print("== Avaliação no teste (Jun completo) ==")
    p_te = predict_proba(model, Xte)
    y_pred = (p_te >= thr).astype(int)
    metrics = {
        "threshold": thr,
        "F1": float(f1_score(Yte, y_pred, zero_division=0)),
        "precision": float(precision_score(Yte, y_pred, zero_division=0)),
        "recall": float(recall_score(Yte, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(Yte, p_te)) if Yte.sum() > 0 else None,
        "PR_AUC": float(average_precision_score(Yte, p_te)) if Yte.sum() > 0 else None,
        "n_test": int(len(Yte)), "n_pos_test": int(Yte.sum()),
        "TP": int(((Yte == 1) & (y_pred == 1)).sum()),
        "FP": int(((Yte == 0) & (y_pred == 1)).sum()),
        "FN": int(((Yte == 1) & (y_pred == 0)).sum()),
    }
    # Baseline LightGBM geral na escavadeira (Sprint 2, fleet_segmentation.json)
    baseline = {"F1": 0.0153, "precision": 0.0081, "recall": 0.1358, "PR_AUC": 0.0102, "ROC_AUC": 0.79}

    print(f"  GRU   : Recall={metrics['recall']:.3f}  Prec={metrics['precision']:.3f}  "
          f"F1={metrics['F1']:.3f}  PR-AUC={metrics['PR_AUC']:.4f}  ROC-AUC={metrics['ROC_AUC']:.3f}")
    print(f"  LGBM  : Recall={baseline['recall']:.3f}  Prec={baseline['precision']:.3f}  "
          f"F1={baseline['F1']:.3f}  PR-AUC={baseline['PR_AUC']:.4f}")
    print(f"  TP={metrics['TP']} de {metrics['n_pos_test']} positivos | FP={metrics['FP']:,}")

    beat_recall = metrics["recall"] > baseline["recall"]
    beat_prauc = (metrics["PR_AUC"] or 0) > baseline["PR_AUC"]
    print(f"\n  ► Supera baseline em Recall? {'SIM' if beat_recall else 'NÃO'} | "
          f"em PR-AUC? {'SIM' if beat_prauc else 'NÃO'}")

    # ── Figura: PR curve GRU vs baseline ───────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    if Yte.sum() > 0:
        prec, rec, _ = precision_recall_curve(Yte, p_te)
        a1.plot(rec, prec, color="#EF4444", linewidth=2,
                label=f"GRU sequencial (AP={metrics['PR_AUC']:.4f})")
    a1.axhline(float(Yte.mean()), color="k", linestyle="--", alpha=0.4,
               label=f"Base rate ({Yte.mean():.5f})")
    a1.scatter([baseline["recall"]], [baseline["precision"]], color="#8B5CF6", s=80, zorder=5,
               label=f"LightGBM geral (R={baseline['recall']:.2f})")
    a1.set_xlabel("Recall"); a1.set_ylabel("Precision")
    a1.set_title("Precision-Recall — Escavadeira (Teste Jun/2025)")
    a1.legend(loc="upper right"); a1.grid(alpha=0.3)

    mets = ["recall", "PR_AUC", "ROC_AUC"]
    gru_v = [metrics[m] or 0 for m in mets]
    bas_v = [baseline[m] for m in mets]
    x = np.arange(len(mets)); w = 0.38
    a2.bar(x - w/2, bas_v, w, label="LightGBM geral", color="#8B5CF6")
    a2.bar(x + w/2, gru_v, w, label="GRU sequencial", color="#EF4444")
    a2.set_xticks(x); a2.set_xticklabels(["Recall", "PR-AUC", "ROC-AUC"])
    a2.set_title("GRU sequencial vs. LightGBM geral — Escavadeira")
    a2.legend(); a2.grid(axis="y", alpha=0.3)
    for xi, (b, gv) in enumerate(zip(bas_v, gru_v)):
        a2.text(xi - w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
        a2.text(xi + w/2, gv, f"{gv:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "escavadeira_gru_vs_baseline.png", dpi=150)
    plt.close()
    print(f"  ✓ {FIGURES_DIR / 'escavadeira_gru_vs_baseline.png'}")

    # ── Persistência ─────────────────────────────────────────────────────────
    payload = {
        "architecture": {"seq_len": SEQ_LEN, "emb_dim": EMB_DIM, "hidden": HIDDEN,
                         "vocab_size": vocab_size, "epochs": EPOCHS,
                         "neg_ratio_train": NEG_RATIO_TRAIN},
        "test_metrics_gru": metrics,
        "baseline_lgbm_escavadeira": baseline,
        "beats_baseline_recall": bool(beat_recall),
        "beats_baseline_prauc": bool(beat_prauc),
        "elapsed_s": time.time() - t0,
        "note_distribution_shift": "Jun (teste) tem apenas 162 positivos vs ~108k no treino — "
                                   "mesmo shift severo do Sprint 5.",
    }
    with open(GOLD_DIR / "escavadeira_gru.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    torch.save(model.state_dict(), GOLD_DIR / "escavadeira_gru.pt")
    print(f"\n✓ Resultados salvos | tempo total {time.time()-t0:.0f}s")
    print("\n══════════════ SPRINT 12 COMPLETO ══════════════")
    return payload


if __name__ == "__main__":
    main()
