import os
import sys
import pandas as pd
import json
import lazyqsar
from sklearn.metrics import roc_curve, auc

pred_file = sys.argv[1]
model_folder = sys.argv[2]
pred_folder = sys.argv[3]

pred_data=pd.read_csv(pred_file)
smiles = pred_data["smiles"]
y_test = pred_data["activity"]

results={}

print(f"Length of the X sample: {len(smiles)}")
model = lazyqsar.LazyBinaryClassifier.load_model(model_folder)
chemeleon = lazyqsar.descriptors.ChemeleonDescriptor()
X = chemeleon.transform(smiles)
print(X.shape)
y_hat = model.predict_proba(X=X)
fpr, tpr, _ = roc_curve(y_test, y_hat)
roc_auc = auc(fpr, tpr)
print("AUROC", roc_auc)
results["y_test"]=y_test
results["y_hat"]=y_hat
results["roc_auc"] = roc_auc

with open(os.path.join(pred_folder,"results.json"), "w") as f:
    json.dump(results, f, indent=2)