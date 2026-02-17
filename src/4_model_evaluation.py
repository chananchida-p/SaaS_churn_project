import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

PRED_PATH = "outputs/test_predictions.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_roc(y_true, p_lr, p_rf, out_path):
    fpr_lr, tpr_lr, _ = roc_curve(y_true, p_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, p_rf)

    auc_lr = roc_auc_score(y_true, p_lr)
    auc_rf = roc_auc_score(y_true, p_rf)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={auc_lr:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, title, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No churn", "Churn"])
    plt.yticks([0, 1], ["No churn", "Churn"])

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def threshold_table(y_true, p, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)

    rows = []
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def main():
    df = pd.read_csv(PRED_PATH)

    # columns created by Step 3
    y_true = df["y_true"].astype(int).values
    p_lr = df["p_churn_logreg"].values
    p_rf = df["p_churn_rf"].values

    # 1) ROC curve
    roc_path = os.path.join(OUT_DIR, "roc_curve_test.png")
    plot_roc(y_true, p_lr, p_rf, roc_path)

    # 2) Confusion matrix at threshold 0.5 (RF)
    y_pred_05 = (p_rf >= 0.5).astype(int)
    cm_05 = confusion_matrix(y_true, y_pred_05)
    cm05_path = os.path.join(OUT_DIR, "confusion_matrix_rf_0_5.png")
    plot_confusion_matrix(cm_05, "Confusion Matrix (RF, threshold=0.5)", cm05_path)

    # 3) Threshold tuning (RF)
    table = threshold_table(y_true, p_rf)
    table_path = os.path.join(OUT_DIR, "threshold_tuning_rf.csv")
    table.to_csv(table_path, index=False)

    best = table.iloc[0].to_dict()
    best_t = best["threshold"]

    # 4) Confusion matrix at best threshold (by F1)
    y_pred_best = (p_rf >= best_t).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)
    cmb_path = os.path.join(OUT_DIR, "confusion_matrix_rf_best.png")
    plot_confusion_matrix(cm_best, f"Confusion Matrix (RF, best F1 threshold={best_t})", cmb_path)

    # 5) Write a short summary text for your report
    auc_lr = roc_auc_score(y_true, p_lr)
    auc_rf = roc_auc_score(y_true, p_rf)

    summary = []
    summary.append("MODEL EVALUATION SUMMARY (Test Set)\n")
    summary.append(f"Logistic Regression AUC: {auc_lr:.3f}\n")
    summary.append(f"Random Forest AUC:       {auc_rf:.3f}\n")
    summary.append("\nRandom Forest confusion matrix @ threshold 0.5:\n")
    summary.append(str(cm_05) + "\n")
    summary.append("\nThreshold tuning (Random Forest): best threshold by F1\n")
    summary.append(
        f"Best threshold={best_t}, precision={best['precision']:.3f}, recall={best['recall']:.3f}, f1={best['f1']:.3f}\n"
    )
    summary.append("Confusion matrix at best threshold:\n")
    summary.append(str(cm_best) + "\n")

    summary_path = os.path.join(OUT_DIR, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("".join(summary))

    print("Saved evaluation outputs:")
    print("-", roc_path)
    print("-", cm05_path)
    print("-", table_path)
    print("-", cmb_path)
    print("-", summary_path)


if __name__ == "__main__":
    main()
