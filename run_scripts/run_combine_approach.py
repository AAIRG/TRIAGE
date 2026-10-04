from lib.utils import (parse_arguments, compute_and_log_ranking_metrics, \
    match_exp_id_from_path, create_experiment_dir, log_completed_run, save_pred_to_json,
                       log_experiment_parameters)
import json
from lib.load_data import BaseDataHandler
from lib.core.combined_approach import Job
import neptune
from config import NeptuneConfig


args = parse_arguments()
mapping_method = "combined_approach"
data_handler = BaseDataHandler(args, mapping_method)

# Track with neptune
neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_1, api_token=NeptuneConfig.API_TOKEN,
                               mode=args.neptune_mode)
experiment_dir = create_experiment_dir(neptune_run)
log_experiment_parameters(neptune_run, mapping_method, args, data_handler=data_handler)

if args.include_vul_type:
    neptune_run["intermediate_predictions_vulnerability_type"] = match_exp_id_from_path(
            args.path_vulnerability_type_predictions) if args.path_vulnerability_type_predictions else None
if args.include_functionality:
    neptune_run["intermediate_predictions_functionality"] = match_exp_id_from_path(
            args.path_functionality_predictions) if args.path_functionality_predictions else None
if args.include_exploitation_method:
    neptune_run["intermediate_predictions_exploitation_method"] = match_exp_id_from_path(
            args.path_exploitation_method_predictions) if args.path_exploitation_method_predictions else None
if args.include_affected_object:
    neptune_run["affected_object_predictions"] = match_exp_id_from_path(
            args.path_affected_object_predictions) if args.path_affected_object_predictions else None
if args.include_in_context_learner:
    neptune_run["in_context_learner_predictions"] = match_exp_id_from_path(
        args.path_in_context_learner_predictions) \
        if args.path_in_context_learner_predictions else None

job = Job(data_handler)
# Track number of CVEs which is loaded from existing experiments
neptune_run["num_cves"].log(len(job.cves_split))
history = job.run_experiment()
compute_and_log_ranking_metrics(history, neptune_run)

# Save history
history = json.dumps(history, indent=4)
save_pred_to_json(history, experiment_dir, mapping_method, args.tmp_dir, args.mapping_type)


log_completed_run(neptune_run, completed=True)
