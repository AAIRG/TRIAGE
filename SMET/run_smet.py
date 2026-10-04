# Evaluate the trained model on our data
import time
from datetime import datetime
# Start timer
print("Starting timer")
start_time = time.time()
import pandas as pd
import os

from settings import ROOT_DIR
from SMET import map_text, get_clf_model, get_emb_model, get_attack_ids
import json
# stop timer
print("Module imports done")
print("--- %s seconds ---" % (time.time() - start_time))

"""
Script to run SMET
"""

# select train or test
SPLIT = "train"

# Path variables
OUTPUT_DIR = os.path.join(str(ROOT_DIR), "models/uncategorized_mapping/SMET_output")
PATH_ID2LABEL = os.path.join(ROOT_DIR, 'SMET/id2ATT&CK_V2.json')
PATH_LR_MODEL = os.path.join(ROOT_DIR, 'SMET/LR_ATT&CK_model_V2.pkl')

cve_base = pd.read_csv(os.path.join(ROOT_DIR, "data/pre_processed/cve_base.csv"))
train_cves = pd.read_csv(os.path.join(ROOT_DIR, "data/pre_processed/cves_train.csv"))
test_cves = pd.read_csv(os.path.join(ROOT_DIR, "data/pre_processed/cves_test.csv"))

df_train = pd.merge(cve_base, train_cves, on="CVE ID")
df_train = df_train.loc[:, ["CVE ID", "description"]]
df_test = pd.merge(cve_base, test_cves, on="CVE ID")
df_test = df_test.loc[:, ["CVE ID", "description"]]

if SPLIT == "train":
    print(f"df_train.shape: {df_train.shape}")
    cve_ids = list(df_train["CVE ID"])
    cve_desc = list(df_train["description"])

elif SPLIT == "test":
    print(f"df_test.shape: {df_test.shape}")
    cve_ids = list(df_test["CVE ID"])
    cve_desc = list(df_test["description"])
else:
    print("SPLIT must be either 'train' or 'test'")

# Load models before mapping
id_to_label = get_attack_ids(PATH_ID2LABEL)
LR_model = get_clf_model(PATH_LR_MODEL)
emb_model = get_emb_model()

# For each cve id map its description to all attack ids. Attack ids are ranked by confidence.
history = []
count = 0
for cve_id, desc in  zip(cve_ids, cve_desc):
    # Get ranked confidence scores for each attack id
    mapping = map_text(desc, LR_model, emb_model, id_to_label, CVE=True)
    # Get attack ids and confidence scores in separate lists
    mapping = list(zip(*mapping))
    attack_ids = list(mapping[0])
    conf_scores = list(mapping[1])
    history_i = {"target_cve": cve_id, "predictions": attack_ids, "confidence": conf_scores}

    history.append(history_i)
    if count % 10 == 0:
        print(f"Mapping for count: {count} done")
        # print(f"cve_id: {cve_id}")
        # print(f"cve_id: {cve_id}")
        # print(f"history_i: {history_i}")
    count += 1
    # print(f"history: {history}")

# Save JSON file of all mappings
history_json = json.dumps(history, indent=2)
# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Save JSON file of all mappings with timestamp
with open(os.path.join(OUTPUT_DIR, f"{SPLIT}_mappings_{timestamp}.json"), "w") as json_file:
    json_file.write(history_json)