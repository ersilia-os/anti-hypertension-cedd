import os
import sys
import pandas as pd
from lazyqsar.qsar import LazyBinaryQSAR


root = os.path.dirname(os.path.abspath(__file__))

dataset = sys.argv[1]
model_name = sys.argv[2]
pred_file = os.path.join(root, "..","data", "prediction_sets", f"{dataset}.csv")

df = pd.read_csv(pred_file)
x_test = df["smiles"].tolist()


model_folder = os.path.join(root, "..", "models", f"{model_name}.zip")
model = LazyBinaryQSAR.load(model_folder)
y_pred = model.predict_proba(smiles_list=x_test)[:, 1]
df["y_pred"] = y_pred

df.to_csv(os.path.join(root, "..","data", "lq_predictions", f"{dataset}_{model_name}.csv"), index=False)
