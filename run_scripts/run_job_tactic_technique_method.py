import os
from settings import CUSTOM_ROOT_DIR
os.environ["DSPY_CACHEDIR"] = os.path.join(CUSTOM_ROOT_DIR, "dspy_cache")
import dspy
import re

from lib.core.other_methodology_mappers import get_tactic_prompt
from lib.load_data import BaseDataHandler
from lib.utils import parse_arguments, configure_dspy, create_experiment_dir, \
    compute_and_log_metrics, log_experiment_parameters, \
    log_completed_run, save_pred_to_csv
import neptune
from config import NeptuneConfig
import pandas as pd
from config import openai_api_key


# NB: This script includes mapper logic and LLM calls.


os.environ["OPENAI_API_KEY"] = openai_api_key

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

args = parse_arguments()
mapping_method = "tactic_technique"
configure_dspy(model=args.path_model, temperature=args.temperature)
data_handler = BaseDataHandler(args, mapping_method)

# Used for logging
num_cves = len(data_handler.cves_split) if args.num_cves is None else args.num_cves

neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_1, api_token=NeptuneConfig.API_TOKEN,  # Configurations
                               mode=args.neptune_mode)
log_experiment_parameters(neptune_run, mapping_method, args=args)
log_completed_run(neptune_run, completed=False)

experiment_dir = create_experiment_dir(neptune_run)

qa = dspy.ChainOfThought("question -> answer")

# Train or test CVEs

labeled_cves = data_handler.cves_split
labeled_cves = labeled_cves['CVE ID']

cve_to_desc = data_handler.cve_id_to_desc



# tactic_descriptions and tactic_to_technique are moved to separate file:
# #    tactic_technique_mapping.csv.
# Taken from the first sentence in the descriptions of Enterprise tactics. See e.g https://attack.mitre.org/tactics/TA0008/
tactic_descriptions = {"Initial Access": "The adversary is trying to get into your network.",
                      "Execution": "The adversary is trying to run malicious code.",
                      "Privilege Escalation": "The adversary is trying to gain higher-level "
                                              "permissions.",
                      "Defense Evasion": "The adversary is trying to avoid being detected.",
                      "Credential Access": "The adversary is trying to steal account names and "
                                           "passwords.",
                      "Lateral Movement": "The adversary is trying to move through your environment."}

# From methodology
tactic_to_technique = {"Initial Access": "T1190",
                       "Execution": "T1203",
                       "Privilege Escalation": "T1068",
                       "Defense Evasion": "T1211",
                       "Credential Access": "T1212",
                       "Lateral Movement": "T1210"}

# multi class approach
cve_to_tactic = {}
tactics = list(tactic_descriptions.keys())
for cve in labeled_cves:
    cve_desc = cve_to_desc[cve]
    instruction_tactic = get_tactic_prompt(tactic_descriptions, cve_desc)
    pred = qa(question=instruction_tactic)
    pred_answer = pred.answer
    print(f"pred.answer: {pred_answer}")
    cve_to_tactic[cve] = pred_answer
print(cve_to_tactic)
# make as dataframe
df_cve_to_tactic = pd.DataFrame(cve_to_tactic.items(), columns=['CVE ID', 'tactic'])
# Save
save_pred_to_csv(df_cve_to_tactic, experiment_dir, mapping_method, args.tmp_dir)

# Postprocess and save final output
# Sometimes, the model answer contains more than just the tactic

def extract_keyword(text):
    for keyword in tactics:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            return keyword
    return None

# Apply the function to extract the desired tactic words
df_cve_to_tactic['tactic_extracted'] = df_cve_to_tactic['tactic'].apply(extract_keyword)
# Drop old tactic
df_cve_to_tactic = df_cve_to_tactic.drop(columns=['tactic'])
# rename the new tactic
df_cve_to_tactic = df_cve_to_tactic.rename(columns={'tactic_extracted': 'tactic'})

# now add column with techniques
df_pred_final = df_cve_to_tactic.copy()
df_pred_final['attack_id'] = df_cve_to_tactic['tactic'].map(tactic_to_technique)
df_pred_final['mapping_type'] = "exploitation_technique"
print(f"df_cve_to_technique: {df_pred_final}")
# Save with experiment ID

# Compute scores
if args.neptune_mode == "async":
    df_attack_labels = data_handler.df_attack_labels
    attack_techniques = data_handler.attack_techniques
    compute_and_log_metrics(df_pred_final, df_attack_labels, mapping_type="exploitation_technique",
                          attack_techniques=attack_techniques, average="micro",
                          neptune_run=neptune_run)
    compute_and_log_metrics(df_pred_final, df_attack_labels, mapping_type="exploitation_technique",
                          attack_techniques=attack_techniques, average="macro",
                          neptune_run=neptune_run)
    compute_and_log_metrics(df_pred_final, df_attack_labels, mapping_type="exploitation_technique",
                          attack_techniques=attack_techniques, average="weighted",
                          neptune_run=neptune_run)


# NB: The output is currently not saved.

log_completed_run(neptune_run, completed=True)
