from lib.utils import parse_arguments, compute_and_log_metrics, \
    log_experiment_parameters, create_experiment_dir, log_completed_run
import os
from lib.load_data import BaseDataHandler
from lib.combined_approach import Job
import neptune
from config import NeptuneConfig
from settings import ROOT_DIR
import pandas as pd

"""
Combine methodology predictions from individual methodology mappers
"""

args = parse_arguments()
mapping_method = "methodology_combine"
data_handler = BaseDataHandler(args, mapping_method)
# Track with neptune
neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_1, api_token=NeptuneConfig.API_TOKEN,
                               mode=args.neptune_mode)

log_experiment_parameters(neptune_run, mapping_method, args)
experiment_dir = create_experiment_dir(neptune_run)

job = Job(data_handler)
# Track number of CVEs which is loaded from existing experiments
neptune_run["num_cves"].log(len(job.cves_split))
pred_techniques = job.run_experiment()
all_cves = job.cves_split["CVE ID"].tolist()
data = {"CVE ID": all_cves,
        "attack_id": pred_techniques}
df = pd.DataFrame(data)
df_pred_final = df.explode('attack_id')
df_pred_final.loc[:, "mapping_type"] = args.mapping_type
df_pred_final = df_pred_final.loc[:, ["CVE ID", "mapping_type", "attack_id"]]
compute_and_log_metrics(df_pred_final, data_handler.df_attack_labels,
                        data_handler.attack_techniques,
                        mapping_method=mapping_method, mapping_type=args.mapping_type,
                        neptune_run=neptune_run)

df_pred_final.to_csv(str(os.path.join(ROOT_DIR, experiment_dir, "predictions.csv")))
log_completed_run(neptune_run, completed=True)