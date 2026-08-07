# -*- coding: utf-8 -*-
"""Statistical helpers kept separate from plotting code."""

import numpy as np


def _finite_1d(values, name='values', minimum=1):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f'{name} 必须是一维数组且至少包含 {minimum} 个值')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} 包含 NaN 或无穷值')
    return array


def confidence_interval(values, confidence=0.95):
    """Return sample mean and a two-sided Student-t confidence interval."""
    from scipy import stats

    array = _finite_1d(values, minimum=2)
    if not 0 < confidence < 1:
        raise ValueError('confidence 必须位于 0 和 1 之间')
    mean = float(array.mean())
    sem = float(stats.sem(array))
    margin = float(stats.t.ppf((1 + confidence) / 2, array.size - 1) * sem)
    return mean, mean - margin, mean + margin


def pareto_mask(values, minimize=True):
    """Return a mask identifying non-dominated rows of an objective matrix."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError('values 必须是至少包含两个目标的非空二维数组')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('values 包含 NaN 或无穷值')
    directions = np.asarray(minimize if np.ndim(minimize) else [minimize] * matrix.shape[1])
    if directions.shape != (matrix.shape[1],):
        raise ValueError('minimize 长度必须等于目标数量')
    normalized = matrix.copy()
    normalized[:, ~directions.astype(bool)] *= -1
    efficient = np.ones(matrix.shape[0], dtype=bool)
    for index, point in enumerate(normalized):
        dominated = np.all(normalized <= point, axis=1) & np.any(normalized < point, axis=1)
        dominated[index] = False
        if np.any(dominated):
            efficient[index] = False
    return efficient


def classification_curves(y_true, y_score, positive_label=1):
    """Compute ROC and precision-recall curves without coupling them to plots."""
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    )

    labels = np.asarray(y_true)
    scores = _finite_1d(y_score, name='y_score', minimum=2)
    if labels.ndim != 1 or labels.size != scores.size:
        raise ValueError('y_true 与 y_score 必须是一维等长数组')
    binary = labels == positive_label
    if np.unique(binary).size != 2:
        raise ValueError('y_true 必须同时包含正类和负类')
    fpr, tpr, roc_thresholds = roc_curve(binary, scores)
    precision, recall, pr_thresholds = precision_recall_curve(binary, scores)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'roc_auc': float(roc_auc_score(binary, scores)),
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'average_precision': float(average_precision_score(binary, scores)),
        'prevalence': float(binary.mean()),
    }


def standardize_matrix(values, method='zscore'):
    """Standardize criteria columns for multi-criteria visual comparison."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError('values 必须是有限非空二维数组')
    if method == 'zscore':
        scale = matrix.std(axis=0, ddof=0)
        scale[scale == 0] = 1
        return (matrix - matrix.mean(axis=0)) / scale
    if method == 'minmax':
        span = np.ptp(matrix, axis=0)
        span[span == 0] = 1
        return (matrix - matrix.min(axis=0)) / span
    raise ValueError("method 必须是 'zscore' 或 'minmax'")
