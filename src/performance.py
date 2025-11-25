import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stylia as st
from stylia import (
    ONE_COLUMN_WIDTH,
    TWO_COLUMNS_WIDTH,
    NamedColorMaps,
    NamedColors,
    ContinuousColorMap,
)
from sklearn.metrics import roc_curve, auc
import json

root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(root, "..","models"))

def plot_yhat(ax,name):
    with open(os.path.join(model_dir, f"report_{name}.json"), 'r') as f:
        results = json.load(f)
    y_hat = np.array(results["0"]['y_hat'])
    y_true = np.array(results["0"]['y_true'])
    y_hats_1 = y_hat[y_true == 1]
    y_hats_0 = y_hat[y_true == 0]
    np.random.shuffle(y_hats_0)
    np.random.shuffle(y_hats_1)
    if len(y_hats_1)>100:
        y_hats_1[:100]
    if len(y_hats_0)>100:
        y_hats_0[:100]
    for y in zip(y_hats_0):
        jitter = np.random.uniform(-0.1, 0.1)
        color = NamedColors().gray
        ax.scatter([0 + jitter], [y], color=color, alpha=0.5)

    for y in zip(y_hats_1):
        jitter = np.random.uniform(-0.1, 0.1)
        color = NamedColors().red
        ax.scatter([1 + jitter], [y], color=color, alpha=0.5)

    ax.set_xticks([0,1])
    ax.set_xticklabels(["Neg", "Pos"])
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(name)

def plot_roc_curve(ax, name):
    with open(os.path.join(model_dir, f"report_{name}.json"), 'r') as f:
        results = json.load(f)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in results.keys():
        y_hat = results[i]['y_hat']
        y_true = results[i]['y_true']
        fpr, tpr, _ = roc_curve(y_true, y_hat)
        roc_auc = auc(fpr, tpr)
        tpr_interp = np.interp(mean_fpr, fpr, tpr, left=0.0, right=1.0)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, lw=1.5, color="#50285a",
            label=f"AUC = {mean_auc:.2f} ± {std_auc:.2f})")
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    alpha=0.25,color="#50285a" )
    ax.set_title(f"ROC Curve {name}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="lower right", fontsize=8)