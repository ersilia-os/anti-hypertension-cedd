from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import collections
from tqdm import tqdm
import json
import numpy as np
import os
import lazyqsar as lq
import pandas as pd

data_dir = "../data/"
model_dir = "../models/"
figure_dir = "../figures/"
N_FOLDS = 5

def cross_validate(smiles_list, y, name, descriptor, method):

    cross_validation_data = collections.defaultdict(list)

    for _ in tqdm(range(N_FOLDS)):
        smiles_train, smiles_test, y_train, y_test = train_test_split(
            smiles_list,y, test_size=0.2, stratify=y
        )

        model = lq.LazyBinaryQSAR(model_type=method, descriptor_type=descriptor)
        model.fit(smiles_train, y_train)
        y_pred = model.predict_proba(smiles_test)

        fpr, tpr, thr = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        J = tpr - fpr

        best_thr_index = np.argmax(J)
        best_thr = thr[best_thr_index]

        cross_validation_data["roc_auc"] += [roc_auc]
        cross_validation_data["thr"] += [best_thr]
        cross_validation_data["y_hat"] += [list(y_pred)]
        cross_validation_data["y"] += [list(y_test)]

    with open(
        os.path.join(model_dir, f"{name}.json"),
        "w",
    ) as f:
        json.dump(cross_validation_data, f, indent=4)

neg_sd = pd.read_csv(os.path.join(data_dir, "chembl", "chembl_smi.csv"))
neg_sd = neg_sd.sample(frac=0.3, random_state=42)
neg_np = pd.read_csv(os.path.join(data_dir, "coconut", "coconut_smi.csv"))
neg_np = neg_np.sample(frac=0.3, random_state=42)
neg = pd.concat([neg_sd, neg_np], axis=0)
neg["activity"] = 0
neg = neg[["canonical_smiles", "activity"]]


# ACE
df = pd.read_csv(os.path.join(data_dir, "processed", "ace.csv"))
df = df[["canonical_smiles", "activity"]]
all = pd.concat([df, neg], axis=0)
cross_validate(all["canonical_smiles"], all["activity"].tolist(), "ace_morgan", "morgan", "zsrandomforest")
cross_validate(all["canonical_smiles"], all["activity"].tolist(), "ace_mordred", "mordred", "zsrandomforest")

## CA
df = pd.read_csv(os.path.join(data_dir, "processed", "ca.csv"))
df = df[["canonical_smiles", "activity"]]
all = pd.concat([df, neg], axis=0)
cross_validate(all["canonical_smiles"], all["activity"].tolist(), "ca_morgan", "morgan", "zsrandomforest")
cross_validate(all["canonical_smiles"], all["activity"].tolist(), "ca_mordred", "mordred", "zsrandomforest")


#K
df = pd.read_csv(os.path.join(data_dir, "processed", "k.csv"))
df = df[["canonical_smiles", "activity"]]
all = pd.concat([df, neg], axis=0)
cross_validate(all["canonical_smiles"], all["activity"].tolist(), "k_morgan", "morgan", "zsrandomforest")
cross_validate(all["canonical_smiles"], all["activity"].tolist(), "k_mordred", "mordred", "zsrandomforest")