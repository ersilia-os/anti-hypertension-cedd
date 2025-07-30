import os
import sys
import pandas as pd
import numpy as np
import json
import lazyqsar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import collections

model_type = "random_forest"
desc = "chemeleon"
train_file = sys.argv[1]
model_folder = sys.argv[2]

if not os.path.exists(model_folder):
    os.mkdir(model_folder)

train = pd.read_csv(train_file)
smiles = train["smiles"]
y = train["activity"]

print("Crossvalidation")
folds = 5
roc_aucs = []
report = collections.defaultdict(dict)
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
chemeleon = lazyqsar.descriptors.ChemeleonDescriptor()
X = chemeleon.transform(smiles)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model = lazyqsar.LazyBinaryClassifier(model_type=model_type)
    model.fit(X_train, y_train)
    y_hat = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_hat)
    roc_auc = auc(fpr, tpr)
    print(f"Fold {fold}","AUROC", roc_auc)
    report[fold]["y_true"] = list(y_test)
    report[fold]["y_hat"] = list(y_hat)
    report[fold]["roc_auc"] = roc_auc
    roc_aucs += [roc_auc]

mean_roc = np.mean(roc_aucs)
st_dev = np.std(roc_aucs)
print("MEAN AUC: ", mean_roc, st_dev)

print("training final model")
model = lazyqsar.LazyBinaryClassifier(model_type=model_type)
model.fit(X, y)
model.save_model(model_folder)

with open(os.path.join(model_folder,"report.json"), "w") as f:
    json.dump(report, f, indent=2)


""" Params for large models:
min_seen_across_partitions=1, 
num_trials=20, 
base_num_splits=1, 
max_samples=10000
"""