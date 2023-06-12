import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

def score_threshold(results):
    true = np.hstack([np.concatenate(r[0]) for r in results])
    oof = np.hstack([np.concatenate(r[1]) for r in results])

    scores = []; thresholds = []
    best_score = 0; best_threshold = 0

    for threshold in np.arange(0.2, 0.9, 0.01):
        preds = (oof.reshape(-1) > threshold).astype('int')
        m = f1_score(true.reshape(-1), preds, average='macro')
        scores.append(m)
        thresholds.append(threshold)
        if m > best_score:
            best_score = m
            best_threshold = threshold

    print(f'When using optimal threshold = {best_threshold:.2f}...')
    for k in range(18):
        m = f1_score(np.concatenate(results[k][0]), (np.concatenate(results[k][1]) > best_threshold).astype('int'), average = 'macro')
        print(f'Q{k}: F1 = {m:.5f}')
    m = f1_score(true.reshape(-1), (oof.reshape(-1) > best_threshold).astype('int'), average = 'macro')
    print(f'==> Overall F1 = {m:.5f}')