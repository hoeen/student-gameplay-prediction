import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def get_metric(targets, preds):
    targets, preds = targets.reshape(-1), preds.reshape(-1)
    f1 = f1_score(targets, np.where(preds >= 0.5, 1, 0), average='macro')
    pre = precision_score(targets, np.where(preds >= 0.5, 1, 0), average='macro')
    rec = recall_score(targets, np.where(preds >= 0.5, 1, 0), average='macro')
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

    return f1, pre, rec, auc, acc
