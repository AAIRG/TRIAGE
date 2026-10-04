import os

from settings import ROOT_DIR, CUSTOM_ROOT_DIR

os.environ["DSPY_CACHEDIR"] = os.path.join(CUSTOM_ROOT_DIR, "dspy_cache")
import json
from lib.load_data import BaseDataHandler
from lib.core.exp_affected_object import Job
import logging

from lib.utils import (parse_arguments, configure_dspy,
                       create_experiment_dir,
                       retrieve_techniques_from_affected_object,
                       compute_and_log_metrics,
                       log_completed_run, save_pred_to_csv, log_experiment_parameters)
import pandas as pd
from config import openai_api_key, NeptuneConfig
import neptune

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Suppress LiteLLM logging to avoid a large number of output cells
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

args = parse_arguments()
mapping_method = "affected_object"
configure_dspy(model=args.path_model, temperature=args.temperature)

data_handler = BaseDataHandler(args, mapping_method)
labeled_data = data_handler.df_attack_labels
attack_techniques = data_handler.attack_techniques

neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_1, api_token=NeptuneConfig.API_TOKEN,
                               mode=args.neptune_mode)
log_experiment_parameters(neptune_run, mapping_method, args)
experiment_dir = create_experiment_dir(neptune_run)
log_completed_run(neptune_run, completed=False)
#
job = Job(data_handler)
# # Store to csv
intermediate_predictions = job.run_experiment()
save_pred_to_csv(intermediate_predictions, experiment_dir, mapping_method, args.tmp_dir)

# Map attack techniques
affected_object_mapping = data_handler.affected_object_mapping
count = 0
all_techniques = []
for row in intermediate_predictions.itertuples():
    cve_id = row[1]
    intermediate_pred = row[2]
    techniques = retrieve_techniques_from_affected_object(affected_object_mapping,
                                                          intermediate_pred,
                                                          use_parent_techniques=args.use_parent_techniques)
    all_techniques.append(techniques)
final_data = {"CVE ID": job.cve_ids, "attack_id": all_techniques}
df_final = pd.DataFrame(final_data)
df_final = df_final.explode(column="attack_id", ignore_index=True)
df_final.loc[:, "mapping_type"] = "exploitation_technique"
df_final = df_final.loc[:, ["CVE ID", "mapping_type", "attack_id"]]
print(df_final.head())
compute_and_log_metrics(df_final, labeled_data, attack_techniques=attack_techniques,
                        mapping_type="exploitation_technique", neptune_run=neptune_run,
                       average="micro")
compute_and_log_metrics(df_final, labeled_data, attack_techniques=attack_techniques,
                        mapping_type="exploitation_technique", neptune_run=neptune_run,
                       average="macro")
compute_and_log_metrics(df_final, labeled_data, attack_techniques=attack_techniques,
                        mapping_type="exploitation_technique", neptune_run=neptune_run,
                       average="weighted")

reasoning_history = json.dumps(job.reasoning_history, indent=2)
with open(os.path.join(experiment_dir, f"reasoning_history.json"), "w") as json_file:
    json_file.write(reasoning_history)

with open(os.path.join(experiment_dir, f"question_history.txt"), "w") as file:
    for element in job.question_history:
        file.write(element + "\n\n")
log_completed_run(neptune_run, completed=True)