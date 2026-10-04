import os
import pandas as pd
from settings import ROOT_DIR
from sklearn.model_selection import train_test_split
from constants import SPLIT_RATIO, SEED_VALUE

# Script to make a train-test split from the labeled CVEs

# Read attack labels
df_attack_labels = pd.read_csv(os.path.join(ROOT_DIR, "data/pre_processed/labeled_cve_to_attack.csv"))

# take out CVEs
cves_attack_labels = df_attack_labels['CVE ID'].unique()

# Make train test split
cves_train, cves_test = train_test_split(cves_attack_labels, test_size=SPLIT_RATIO,
                                         random_state=SEED_VALUE)

# Make dataframe
cves_train = pd.DataFrame(cves_train, columns=['CVE ID'])
cves_test = pd.DataFrame(cves_test, columns=['CVE ID'])

# Save


