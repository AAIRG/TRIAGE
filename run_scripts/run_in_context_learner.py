import os
from settings import CUSTOM_ROOT_DIR

os.environ["DSPY_CACHEDIR"] = os.path.join(CUSTOM_ROOT_DIR, "dspy_cache")

import json
import pandas as pd
from lib.load_data import BaseDataHandler
from lib.core.in_context_learner import Job
from lib.utils import (parse_arguments, configure_dspy,
                       compute_and_log_metrics, create_experiment_dir,
                       compute_and_log_ranking_metrics,
                       log_completed_run, save_pred_to_json,
                       log_experiment_parameters)
from config import openai_api_key, NeptuneConfig
import neptune

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

args = parse_arguments()
mapping_method = "in_context_learner"
configure_dspy(model=args.path_model, temperature=args.temperature)
data_handler = BaseDataHandler(args, mapping_method)

neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_1, api_token=NeptuneConfig.API_TOKEN,
                               mode=args.neptune_mode)
experiment_dir = create_experiment_dir(neptune_run)
num_cves = len(data_handler.cves_split) if args.num_cves is None else args.num_cves
# TODO: fix this for the test set


# Not implemented yet
# if args.include_tactic_prompt:
#     neptune_run["intermediate_predictions_tactic"] = match_exp_id(args.path_tactic_level_predictions)
neptune_run["use_ranking_approach"] = True if args.use_ranking_approach else False
log_experiment_parameters(neptune_run, mapping_method, args, data_handler=data_handler)
log_completed_run(neptune_run, completed=False)

job = Job(data_handler, neptune_run, args)
all_predictions = job.run_experiment()
df_attack_labels = data_handler.df_attack_labels
# Filter df_attack_labels by mapping type
df_attack_labels_type = df_attack_labels.loc[df_attack_labels["mapping_type"] == args.mapping_type]
print(df_attack_labels_type)

history = []
for i in range(len(job.cve_ids)):
    target_cve = job.cve_ids[i]
    df_attack_labels_sub = df_attack_labels_type.loc[df_attack_labels_type["CVE ID"] == target_cve]
    true_attack_ids = list(df_attack_labels_sub["attack_id"])
    pred_attack_ids = job.answers[i]
    # demo_cves = job.demo_history[i]
    reasoning = job.reasoning_history[i]
    # exp_method_inter_pred = job.exp_method_intermediate_pred[i]
    history_i = {
        "target_cve": target_cve,
        "predictions": pred_attack_ids,
        "true labels": true_attack_ids,
        # "exploitation_method": exp_method_inter_pred,
        "reasoning": reasoning
        # "demonstration_cves": demo_cves
    }
    history.append(history_i)


if args.use_ranking_approach:
    compute_and_log_ranking_metrics(history, neptune_run)

else:
    df_pred_final = pd.DataFrame(all_predictions)
    # Ensure 'attack_id' is a list for each row
    df_pred_final['attack_id'] = df_pred_final['attack_id'].apply(lambda x: eval(x) if
    isinstance(x, str) else x)

    # explode if multiple techniques
    df_pred_final = df_pred_final.explode('attack_id', ignore_index=True)

    # save output
    df_pred_final.to_csv(os.path.join(experiment_dir, f"pred.csv"), index=False)

    df_pred_eval = df_pred_final.copy()

    # Add column mapping_type "exploitation_technique"
    df_pred_eval.loc[:, "mapping_type"] = args.mapping_type

    df_pred_eval = df_pred_eval.loc[:, ["CVE ID", "mapping_type", "attack_id"]]


    # Log both micro, macro and weighted averaging scores
    compute_and_log_metrics(df_pred_eval, data_handler.df_attack_labels, data_handler.attack_techniques,
                            mapping_method=mapping_method, mapping_type=args.mapping_type,
                            neptune_run=neptune_run, average="micro")
    compute_and_log_metrics(df_pred_eval, data_handler.df_attack_labels, data_handler.attack_techniques,
                            mapping_method=mapping_method, mapping_type=args.mapping_type,
                            neptune_run=neptune_run, average="macro")
    compute_and_log_metrics(df_pred_eval, data_handler.df_attack_labels, data_handler.attack_techniques,
                            mapping_method=mapping_method, mapping_type=args.mapping_type,
                            neptune_run=neptune_run, average="weighted")


# Save history
history = json.dumps(history, indent=4)

# save reasoning history, context and question.
save_pred_to_json(history, experiment_dir, mapping_method, args.tmp_dir, args.mapping_type)


with open(os.path.join(experiment_dir, f"question_history.txt"), "w") as file:
    for element in job.question_history:
        file.write(element + "\n\n")

with open(os.path.join(experiment_dir, f"context.txt"), "w") as file:
    file.write(job.context)

log_completed_run(neptune_run, completed=True)