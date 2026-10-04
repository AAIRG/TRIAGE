from lib.load_data import BaseDataHandler
from lib.utils import (parse_arguments, create_experiment_dir,
                                              compute_and_log_ranking_metrics)

import json
from lib.smet_ranking import Job
from config import NeptuneConfig
import neptune
import os

"""
Evaluate SMET predictions
"""

args = parse_arguments()
mapping_method = "smet"
data_handler = BaseDataHandler(args, mapping_method)
neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_2, api_token=NeptuneConfig.API_TOKEN,
                               mode=args.neptune_mode)
experiment_dir = create_experiment_dir(neptune_run, categorized_mapping=False)
neptune_run["mapping_method"].log(mapping_method)
num_cves = len(data_handler.cves_split) if args.num_cves is None else args.num_cves
neptune_run["num_cves"].log(num_cves)
neptune_run["data_split"] = args.data_split
neptune_run["exclude_secondary_impact"] = False if not args.exclude_secondary_impact else True
job = Job(data_handler, args)
job.run_experiment()
history = job.history
compute_and_log_ranking_metrics(history, neptune_run)

history = json.dumps(history, indent=4)
with open(os.path.join(experiment_dir, f"history.json"), "w") as json_file:
    json_file.write(history)