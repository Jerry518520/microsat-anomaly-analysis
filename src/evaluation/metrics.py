"""
评估指标模块
"""

from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)


def compute_all_metrics(y_true, y_pred, y_scores=None):
    """计算所有评估指标"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_scores is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
    return metrics


def print_classification_report(y_true, y_pred):
    """打印分类报告"""
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))


def print_confusion_matrix(y_true, y_pred):
    """打印混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
