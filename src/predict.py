import os
import sys
import pandas as pd
import json
import lazyqsar
from sklearn.metrics import roc_curve, auc

pred_file = sys.argv[1]
model_folder = sys.argv[2]
pred_folder = sys.argv[3]

if not os.path.exists(pred_folder):
    os.mkdir(pred_folder)

pred_data=pd.read_csv(pred_file)
smiles = pred_data["smiles"]

results={}

print(f"Length of the X sample: {len(smiles)}")
model = lazyqsar.LazyBinaryClassifier.load_model(model_folder)
chemeleon = lazyqsar.descriptors.ChemeleonDescriptor()
X = chemeleon.transform(smiles)
print(X.shape)
y_hat = model.predict_proba(X=X)[:,1]
results["model"] = str(model_folder.split("/")[-1])
results["pred_set"] = str(pred_file.split("/")[-1])
results["y_hat"]=list(y_hat)

with open(os.path.join(pred_folder,"results.json"), "w") as f:
    json.dump(results, f, indent=2)