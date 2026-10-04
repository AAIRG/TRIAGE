from lib.load_data import BaseDataHandler
from lib.utils import (parse_arguments,
                                              match_exp_id_from_path,
                                              compute_and_log_ranking_metrics,
                                              create_experiment_dir)
from lib.uncategorized_triage import Job
import os
import json

from config import NeptuneConfig
import neptune

"""
Uncategorized approach combining predictions from three mapping types.
This approach can be compared with SMET. 
"""
args = parse_arguments()
mapping_method = "uncategorized_triage"
data_handler = BaseDataHandler(args, mapping_method)
neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_2, api_token=NeptuneConfig.API_TOKEN,
                               mode=args.neptune_mode)
experiment_dir = create_experiment_dir(neptune_run, categorized_mapping=False)
neptune_run["mapping_method"].log(mapping_method)
num_cves = len(data_handler.cves_split) if args.num_cves is None else args.num_cves
neptune_run["num_cves"].log(num_cves)
neptune_run["data_split"] = args.data_split
neptune_run["exploitation_technique_id"] = (match_exp_id_from_path(
    args.path_exploitation_technique_predictions))
neptune_run["primary_impact_id"] = match_exp_id_from_path(args.path_primary_impact_predictions)
neptune_run["secondary_impact_id"] = match_exp_id_from_path(args.path_secondary_impact_predictions)
neptune_run["exclude_secondary_impact"] = False if not args.exclude_secondary_impact else True
job = Job(data_handler, args)
job.run_experiment()
history = job.history
compute_and_log_ranking_metrics(history, neptune_run)

history = json.dumps(history, indent=4)
with open(os.path.join(experiment_dir, f"history.json"), "w") as json_file:
    json_file.write(history)

# print(history)
