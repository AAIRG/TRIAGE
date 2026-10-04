import json
import os
from pathlib import Path

import dspy
import numpy as np
import pandas as pd
import argparse
import re
from config import NeptuneConfig
import neptune
from config import lambda_api_key, openai_api_key

from settings import ROOT_DIR
from sklearn.metrics import precision_score, recall_score, f1_score
from constants import SEED_VALUE

import logging

# Suppress LiteLLM logging to avoid a large number of output cells
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Handle output model output and compute performance scores

# Utilities for mapping intermediate labels, binarize predictions and compute
#   performance scores

def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_techniques_base", type=str, default=
                        "data/pre_processed/attack_techniques_base.csv",
                        help="Path to file with relevant data for all enterprise techniques.")
    parser.add_argument("--path_methodology_techniques", type=str, default=
    "data/pre_processed/methodology_attack_techniques.csv",
                        help="File contains attack techniques specified in CVE Mapping "
                             "Methodology (CMM).")
    parser.add_argument("--path_functionality_mapping", type=str,
                        default= "data/pre_processed/functionality_mapping.csv",
                        help="CMM mappings for functionality.")
    parser.add_argument("--path_vulnerability_type_mapping", type=str, default=
                        "data/pre_processed/vulnerability_type_mapping.csv",
                        help="CMM mappings for vulnerability type.")
    parser.add_argument("--path_exploitation_method_mapping", type=str,
                        default="data/pre_processed/exploitation_techniques_mapping.json",
                        help="CMM mappings for the exploitation technique method.")
    parser.add_argument("--path_affected_object_mapping", type=str,
                        default="data/pre_processed/affected_object.json",
                        help="CMM mappings for the affected object types.")
    parser.add_argument("--path_cwe_id_to_description", type=str,
                        default="data/pre_processed/cwe_id_to_description.csv")
    parser.add_argument("--path_cwe_id_to_name", type=str,
                        default="data/pre_processed/cwe_id_to_name.csv")
    parser.add_argument("--path_vulnerability_type_to_cwe", type=str,
                        default="data/pre_processed/vul_type_to_cwe.csv",
                        help="Mapping vulnerability types (CMM) to CWE.")
    parser.add_argument("--use_functionality_demos", action="store_true",
                        help="Use functionality demos")
    parser.add_argument('--path_functionality_labels', type=str,
                        default="data/pre_processed/functionality_labels.csv",
                        help="Demonstration examples with mappings from vulnerability to "
                             "functionality.")
    parser.add_argument('--path_cve_base', type=str,
                        default="data/pre_processed/cve_base.csv",
                        help="CVEs considered for the project with relevant data.")
    parser.add_argument('--data_split', type=str, default="train",
                        help="Select 'train' or 'test' split.")
    parser.add_argument("--neptune_mode", type=str, default="debug",
                        help="Log to neptune")
    parser.add_argument('--num_cves', type=int,
                        help="Make a subset of the data by selecting the number of CVEs.")
    parser.add_argument("--random_seed", type=int, default=SEED_VALUE,
                        help="Change to make a different a subset of the data based. Used "
                             "together with 'num_cves'.")
    parser.add_argument("--use_parent_techniques", action="store_true",
                        help="Only consider parent techniques. E.g lift sub-techniques to their "
                             "corresponding parent technique.")
    parser.add_argument('--compute_metrics', action="store_true")
    parser.add_argument('--logging_interval', type=int, default=10)
    parser.add_argument("--mapping_type", type=str,
                        help="Determines which mapping type to use. Relevant for "
                             "in_context_learner and combined_approach.")
    # Below specifies paths to existing predictions for individual mapping_methods
    parser.add_argument("--path_vulnerability_type_predictions", type=str)
    parser.add_argument("--path_exploitation_method_predictions", type=str,
                        help="Refers to the 'exploitation_technique' mapping method. Not to be "
                             "confused with the mapping type with the same name.")
    parser.add_argument("--path_tactic_level_predictions", type=str)
    parser.add_argument("--path_functionality_predictions", type=str)
    parser.add_argument("--path_affected_object_predictions", type=str)
    parser.add_argument("--path_in_context_learner_predictions", type=str)
    # Next three args is used in the uncategorized approach where we combine the final predictions
    #   across mapping types
    parser.add_argument("--path_exploitation_technique_predictions", type=str,
                        help="Refers to the 'exploitation_technique' mapping type not to be "
                             "confused with the mapping method with the same name.")
    parser.add_argument("--path_primary_impact_predictions", type=str)
    parser.add_argument("--path_secondary_impact_predictions", type=str)
    parser.add_argument("--dir_SMET_predictions", type=str,
                        default="models/uncategorized_mapping/SMET_output")
    parser.add_argument("--include_intermediate_labels", action="store_true",
                        help="Include intermediate labels in the final mapping")
    parser.add_argument("--num_demonstrations", type=int, default=None,
                        help="Number of demonstrations to use for in-context learning. If None, "
                             "all available examples will be used.")
    parser.add_argument("--path_model", type=str, default="openai/gpt-4o-mini",
                        help="Path to the LM model. Example with lambda: "
                             "'openai/llama3.3-70b-instruct-fp8'")
    parser.add_argument("--dspy_module", type=str, default="chain_of_thought")
    # add relevant arguments for two step approach. Select options to include
    parser.add_argument("--include_cwe", action="store_true")
    parser.add_argument("--include_cvss", action="store_true")
    parser.add_argument("--include_in_context_learner", action="store_true")
    parser.add_argument("--include_vul_type", action="store_true")
    parser.add_argument("--include_functionality", action="store_true")
    parser.add_argument("--include_exploitation_method", action="store_true")
    parser.add_argument("--include_affected_object", action="store_true")
    parser.add_argument("--include_tactic", action="store_true")
    parser.add_argument("--include_in_context_prompt", action="store_true")
    parser.add_argument("--include_attack_descriptions", action="store_true")
    parser.add_argument("--embedding_model", type=str, default="paraphrase-MiniLM-L6-v2",
                       help="Name of the Sentence transformer model. Relevant only with "
                            "--use_similar_demonstrations "
                            "Other examples: basel/ATTACK-BERT")
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--use_ranking_approach', action='store_true',
                        help="Option to use the ranked approach instead of the unranked.")
    parser.add_argument("--exclude_secondary_impact", action="store_true")
    parser.add_argument("--instance_ip", type=str, default=None,
                        help="The IP of the running instance. For example a GPU instance from "
                             "lambda.ai")
    parser.add_argument("--api_port", type=str, default=None,
                        help="Should be provided together above argument. The port of the running "
                             "instance.")
    parser.add_argument("--tmp_dir", type=str, default="models/tmp_output",
                        help="Temporary place to store output of individual experiments.")



    if 'JPY_PARENT_PID' in os.environ:
        print("Running in Jupyter notebook")
        if args_list is None:
            args_list = []
        args = parser.parse_args(args_list)
    else:
        print("Running in terminal")
        args = parser.parse_args()
    return args

def process_labels(df, mapping_type, techniques=None):
    """
    Process predicted and true attack techniques and binarize them
    Args:
        df:
        mapping_type:
        techniques:

    Returns:

    """
    # df = df.drop(columns=["capability_description"])
    # print("Processing labels")
    # print(f"df.columns: {df.columns}")
    all_cves = df["CVE ID"].unique()
    df_mapping_type = df.loc[df["mapping_type"] == mapping_type]
    # print(f"df_mapping_type.columns: {df_mapping_type.columns}")
    cves_category = df_mapping_type["CVE ID"].unique()
    # The cves that are not in the category
    diff_cves = np.setdiff1d(all_cves, cves_category)
    # Concatenate empty rows for the cves that are not in the category
    if len(diff_cves) > 0:
        df_empty = pd.DataFrame({"CVE ID": diff_cves})
        df_mapping_type = pd.concat([df_mapping_type, df_empty])
    # Columns no longer needed
    # df_mapping_type = df_mapping_type.drop(columns=["mapping_type", "attack_name"])
    df_mapping_type = df_mapping_type.loc[: , ["CVE ID", "attack_id"]]
    # print(f"df_mapping_type.columns: {df_mapping_type.columns}")
    df_category_bin = binarize_df(df_mapping_type, target_column="attack_id", techniques=techniques,
                                  group_by_cve=True)
    return df_category_bin


def binarize_df(df, target_column="attack_id", techniques=None, group_by_cve=False, prefix='',
                prefix_sep=''):
    """
    Binarize dataframe
    Get dummies of attack techniques and group by CVE IDs such that one row corresponds to one
    CVE ID.
    Args:
        df: DataFrame to be binarized.
        target_column: Column to be converted into dummy variables.
        techniques: DataFrame containing all possible techniques.
        group_by_cve: Boolean indicating whether to group by CVE ID.
        prefix: Prefix for dummy variable columns.
        prefix_sep: Separator for prefix in dummy variable columns.

    Returns:
        Binarized DataFrame.
    """
    df = pd.get_dummies(df, prefix=prefix, prefix_sep=prefix_sep, columns=[target_column], dtype=int)
    if group_by_cve:
        df = df.groupby(['CVE ID'], as_index=False).max()
    # Adjust binary encoded data according to all possible techniques. Techniques not present in
    #   original df will be added with 0 values
    if techniques is not None:
        techniques_set = set(techniques["attack_id"])
        missing_techniques = techniques_set - set(df.columns)
        missing_techniques = list(missing_techniques)

        # Create a DataFrame with missing columns initialized to 0
        missing_columns_df = pd.DataFrame(0, index=df.index, columns=missing_techniques)

        # Concatenate the missing columns DataFrame to the original DataFrame
        df = pd.concat([df, missing_columns_df], axis=1)
        # Sort columns
        df = df.reindex(sorted(df.columns), axis=1)
    return df


def preprocess_inputs(df_pred, df_attack_labels, techniques, mapping_type):

    """
    Preprocess inputs for computing scores.

    Args:
        df_pred:
        df_attack_labels:
        techniques:
        mapping_type:

    Returns:

    """
    # print("Preprocessing inputs")
    # print(f"df_attack_labels.columns: {df_attack_labels.columns}")
    df_category_labels = process_labels(df_attack_labels, mapping_type=mapping_type,
                                        techniques=techniques)
    df_mapped_labels = process_labels(df_pred, mapping_type=mapping_type, techniques=techniques)

    return df_category_labels, df_mapped_labels


def compute_attack_scores(df_pred, df_attack_labels, mapping_type, techniques, average,
                          neptune_run=None):
    """
    Map CVE to attack techniques based on functionalities and compute scores.

    Args:
        average: averaging option for multiclass/multilabel
        neptune_run:
        df_pred: Final predictions of attack techniques
        df_attack_labels: Labels of CVE to attack techniques
        techniques: all relevant attack techniques (preprocessed as a set)
        mapping_type: primary_impact, secondary_impact, exploitation_technique

    Returns:
        pre, rec, f1: Precision, recall and F1 score
    """
    # Input preprocessing
    df_category_labels, df_mapped = preprocess_inputs(df_pred, df_attack_labels, techniques,
                                                      mapping_type)

    # print(f"df_category_labels.shape: {df_category_labels.shape}")
    # print(f"df_mapped.shape: {df_mapped.shape}")
    # print(f"df_category_labels.columns: {df_category_labels.columns}")

    # if shape is unequal, add rows with only zeros to the other variable
    # Find difference in attack ids
    category_attack_ids = set(df_category_labels.columns)
    mapped_attack_ids = set(df_mapped.columns)
    # making a copy before further processing
    df_mapped_2 = df_mapped.copy()
    df_category_labels_2 = df_category_labels.copy()

    # Code below should probably be removed. This case should not happen.
    # diff_1 = category_attack_ids - mapped_attack_ids
    # if len(diff_1) > 0:
    #     print(f"attack_ids: {diff_1} exist in df_category_labels but not in df_mapped. Adding "
    #           f"zero values.")
    #     # add column with zeros
    #     for attack_id in list(diff_1):
    #         df_mapped_2[attack_id] = 0
    diff_2 = mapped_attack_ids - category_attack_ids
    if len(diff_2) > 0:
        print(f"attack_ids: {diff_2} exist in df_mapped but not in df_category_labels. Adding "
              f"zero values.")
        for attack_id in list(diff_2):
            df_category_labels_2[attack_id] = 0


    # Compute scores
    pre, rec, f1 = compute_scores(df_category_labels_2, df_mapped_2, average=average)
    print(f"Performance on {mapping_type}: pre: {pre}, rec: {rec}, f1: {f1}")
    if neptune_run is not None:
        log_metrics(pre, rec, f1, mapping_type=mapping_type, neptune_run=neptune_run,
                    average=average)

    return pre, rec, f1

def compute_scores(df_true, df_pred, average="micro"):
    """
    Compute performance scores
    Args:
        average:
        df_true:
        df_pred:

    Returns:

    """

    # Sort by CVE ID and reset index
    df_pred = df_pred.sort_values(by="CVE ID", ignore_index=True)
    df_true = df_true.sort_values(by="CVE ID", ignore_index=True)

    # drop CVE ID column
    df_pred = df_pred.drop(columns=["CVE ID"])
    df_true = df_true.drop(columns=["CVE ID"])

    # Compute metrics
    # print(f"df_pred.values: {df_pred.values}")
    precision = precision_score(df_true.values, df_pred.values, average=average)
    recall = recall_score(df_true.values, df_pred.values, average=average)
    f1 = f1_score(df_true.values, df_pred.values, average=average)

    return precision, recall, f1


def compute_and_log_metrics(df_pred, df_attack_labels, attack_techniques, mapping_method=None,
                            mapping_type=None, neptune_run=None, average="micro"):

    # Filter out CVEs in attack_labels not in df_pred_intermediate
    df_attack_labels = df_attack_labels.loc[df_attack_labels['CVE ID'].isin(df_pred['CVE ID'])]
    df_attack_labels.reset_index(drop=True, inplace=True)

    if mapping_method == "exploitation_technique":
        # Mapping type and mapping method is the same for exploitation_technique
        compute_attack_scores(df_pred, df_attack_labels, mapping_type="exploitation_technique",
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)
    elif mapping_method == "functionality":
        compute_attack_scores(df_pred, df_attack_labels, mapping_type="primary_impact",
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)
        compute_attack_scores(df_pred, df_attack_labels, mapping_type="secondary_impact",
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)
    elif mapping_method == "vulnerability_type":
        compute_attack_scores(df_pred, df_attack_labels, mapping_type="exploitation_technique",
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)
        compute_attack_scores(df_pred, df_attack_labels, mapping_type="primary_impact",
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)
        compute_attack_scores(df_pred, df_attack_labels, mapping_type="secondary_impact",
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)
    else:
        compute_attack_scores(df_pred, df_attack_labels, mapping_type=mapping_type,
                              techniques=attack_techniques, average=average,
                              neptune_run=neptune_run)


def log_metrics(precision, recall, f1, mapping_type, neptune_run, average):
    if mapping_type == "exploitation_technique":
        neptune_run[f'metrics/exploitation_technique/{average}_precision'].log(precision)
        neptune_run[f'metrics/exploitation_technique/{average}_recall'].log(recall)
        neptune_run[f'metrics/exploitation_technique/{average}_f1'].log(f1)
    elif mapping_type == "primary_impact":
        neptune_run[f'metrics/primary_impact/{average}_precision'].log(precision)
        neptune_run[f'metrics/primary_impact/{average}_recall'].log(recall)
        neptune_run[f'metrics/primary_impact/{average}_f1'].log(f1)
    elif mapping_type == "secondary_impact":
        neptune_run[f'metrics/secondary_impact/{average}_precision'].log(precision)
        neptune_run[f'metrics/secondary_impact/{average}_recall'].log(recall)
        neptune_run[f'metrics/secondary_impact/{average}_f1'].log(f1)

def log_experiment_parameters(neptune_run, mapping_method, args, data_handler=None, parent_id=None):
    """
   Log experiment parameters and optionally child experiment details to a Neptune run.

    Args:
        neptune_run: Neptune run object used for logging experiment parameters.
        args (argparse.Namespace): Parsed command-line arguments containing configuration settings.
        mapping_method: Mapping method used for predictions.
        parent_id (str, optional): ID of the parent experiment, used when logging child experiments.
        data_handler

    Logs:
        - Mapping method used for predictions.
        - Number of CVEs processed.
        - Path to the model used.
        - Version of labeled data.
        - Data split type (train/test).
        - Whether parent techniques are used.
        - Random seed value for reproducibility.
        - Temperature of the LLM.
        - Type of mapping applied (if specified).
        - Parent experiment ID if logging child experiments.
        - For methodology_combine approach:
            - Experiment IDs for intermediate predictions (vulnerability type, exploitation method,
            affected object, tactic level, functionality).
        - For combined_approach or methodology_combine approaches:
            - Whether vulnerability type is included.
            - Whether exploitation method is included.
            - Whether affected object is included.
            - Whether tactic is included.
            - Whether functionality is included.
        - For combined_approach or in_context_learner approaches:
            - Whether CWE information is included.
            - Whether CVSS information is included.
            - Embedding model used.
            - Whether attack descriptions are included.
            - Number of demonstrations used.
       """
    neptune_run["mapping_method"].log(mapping_method)
    if not mapping_method == "methodology_combine" or mapping_method == "combined_approach":
        neptune_run["num_cves"].log(args.num_cves)
    neptune_run["path_model"].log(args.path_model)
    neptune_run["data_split"] = args.data_split
    neptune_run["use_parent_techniques"] = args.use_parent_techniques
    neptune_run["seed_value"] = args.random_seed
    neptune_run["temperature"] = args.temperature
    if args.mapping_type is not None:
        neptune_run["mapping_type"].log(args.mapping_type)
    if parent_id is not None:
        neptune_run["parent_id"].log(parent_id)
    if mapping_method == "methodology_combine":
        neptune_run["intermediate_predictions_vulnerability_type"] = match_exp_id_from_path(
            args.path_vulnerability_type_predictions) if args.include_vul_type else None
        neptune_run["intermediate_predictions_exploitation_method"] = match_exp_id_from_path(
            args.path_exploitation_method_predictions) if args.include_exploitation_method else None
        neptune_run["affected_object_predictions"] = match_exp_id_from_path(
            args.path_affected_object_predictions) if args.include_affected_object else None
        neptune_run["tactic_level_predictions"] = match_exp_id_from_path(
            args.path_tactic_level_predictions) if args.include_tactic else None
        neptune_run["functionality_predictions"] = match_exp_id_from_path(
            args.path_functionality_predictions) if args.include_functionality else None
    if mapping_method == "combined_approach" or mapping_method == "methodology_combine":
        neptune_run["include_vul_type"] = args.include_vul_type
        neptune_run["include_exploitation_method"] = args.include_exploitation_method
        neptune_run["include_affected_object"] = args.include_affected_object
        neptune_run["include_tactic"] = args.include_tactic
        neptune_run["include_functionality"] = args.include_functionality
    if mapping_method == "combined_approach" or mapping_method == "in_context_learner":
        neptune_run["include_cwe"] = True if args.include_cwe else False
        neptune_run["include_cvss"] = True if args.include_cvss else False
        neptune_run["embedding_model"] = args.embedding_model
        neptune_run["include_attack_descriptions"] = args.include_attack_descriptions
        if args.num_demonstrations is None:
            if args.data_split == "train":
                num_demonstrations = len(data_handler.cves_train) - 1
            elif args.data_split == "test":
                num_demonstrations = len(data_handler.cves_train)
        else:
            num_demonstrations = args.num_demonstrations
        neptune_run["num_demonstrations"].log(num_demonstrations)




def log_completed_run(neptune_run, completed):
    # Useful to catch if the experiment is cancelled
    # completed is boolean
    neptune_run["completed"] = completed


def log_child_experiment(df_pred_final, df_attack_labels, attack_techniques, mapping_type, args,
                         parent_id, mapping_method):
    neptune_run = neptune.init_run(project=NeptuneConfig.PROJECT_1,
                                   api_token=NeptuneConfig.API_TOKEN,
                                   mode=args.neptune_mode)
    compute_and_log_metrics(df_pred_final, df_attack_labels, attack_techniques,
                            mapping_type=mapping_type, neptune_run=neptune_run, average="micro")
    compute_and_log_metrics(df_pred_final, df_attack_labels, attack_techniques,
                            mapping_type=mapping_type, neptune_run=neptune_run, average="macro")
    compute_and_log_metrics(df_pred_final, df_attack_labels, attack_techniques,
                            mapping_type=mapping_type, neptune_run=neptune_run, average="weighted")
    log_experiment_parameters(neptune_run, mapping_method, args, parent_id=parent_id)


def map_intermediate_labels(mapping, df_pred_intermediate, mapping_method, mapping_type=None,
                            include_intermediate_labels=False):
    """
    Create final labels by converting intermediate labels.
    Args:
        mapping_type: Exploitation technique, primary impact or secondary impact
        mapping_method:
        df_pred_intermediate:
        mapping: From intermediate predictions to attack techniques. For example from vulnerability
        type to attack techniques
        include_intermediate_labels:

    Returns:

    """
    if df_pred_intermediate is None:
        return None
    if mapping_method == "vulnerability_type":
        mapping = mapping.rename(columns={
            'Vulnerability Type': 'pred_vul_type',
            'sublabel': 'mapping_sublabel'})
        try:
            df_pred = pd.merge(df_pred_intermediate, mapping, how='left', on='pred_vul_type')
            # filter for entries with sublabels
            df_pred_sublabels = df_pred.dropna(subset=['mapping_sublabel'])
            # get indices to drop, drop row if sublabel is different from mapping_sublabel
            drop_indices = df_pred_sublabels.loc[df_pred_sublabels["sublabel"] !=
                                                 df_pred_sublabels["mapping_sublabel"]].index
            df_pred_2 = df_pred.drop(drop_indices)
            # Drop nan values in attack id, when vulnerability type does not map to attack id
            # df_pred_2 = df_pred_2.dropna(subset=['attack_id'])
            df_pred_2 = df_pred_2.reset_index(drop=True)
            if include_intermediate_labels:
                df_pred_final = df_pred_2.loc[:, ["CVE ID", "pred_vul_type", "mapping_type",
                                                  "attack_id"]]
            else:
                df_pred_final = df_pred_2.loc[:, ["CVE ID", "mapping_type", "attack_id"]]
        except ValueError:
            # Handle case with no relevant vul types. e.g. only NaN/None values
            df_pred_final = None

    elif mapping_method == "functionality":
        df_pred = pd.merge(df_pred_intermediate, mapping, how='left', on='Functionality')
        if include_intermediate_labels:
            df_pred = df_pred.rename(columns={"Functionality": "pred_functionality"})
            df_pred_final = df_pred.loc[:, ["CVE ID", "pred_functionality", "mapping_type",
                                            "attack_id"]]
        else:
            df_pred_final = df_pred.loc[:, ["CVE ID", "mapping_type", "attack_id"]]
    else:
        RaiseNotImplementedError("Mapping method not implemented")
    if mapping_type:
        df_pred_final = df_pred_final.loc[df_pred_final["mapping_type"] == mapping_type]
    return df_pred_final


def create_experiment_dir(neptune_run, categorized_mapping=True):
    """
    Create output directory for experiments. E.g predictions.
    Args:
        neptune_run:

    Returns:

    """
    nept_id = neptune_run["sys/id"].fetch()
    if categorized_mapping:
        experiment_dir = os.path.join(ROOT_DIR, f'models/categorized_mapping/experiments/{nept_id}')
    else:
        experiment_dir = os.path.join(ROOT_DIR, f'models/uncategorized_mapping/experiment'
                                                f's/{nept_id}')
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    return experiment_dir



def configure_dspy(model, temperature=0.2, cache=False, cache_in_memory=False):
    if model == "openai/gpt-4o-mini":
        lm = dspy.LM(model=model, api_key=openai_api_key, temperature=temperature,
                     max_tokens=16384, cache=cache, cache_in_memory=cache_in_memory)
    else:
        # serverless variant using lambda:
        api_base = "https://api.lambda.ai/v1"
        lm = dspy.LM(model=model, api_key=lambda_api_key, api_base=api_base,
                     temperature=temperature,
                     max_tokens=20000,
                     cache=cache, cache_in_memory=cache_in_memory)
    print(f"lm: {lm.model}")
    dspy.configure(lm=lm)

def match_exp_id_from_path(path):
    # pattern to extract experiment id from path
    pattern_exp_id = r"(CVET2-\d+)"
    match = re.search(pattern_exp_id, path)
    exp_id = match.group(1)
    return exp_id


def remove_digits(attack_id):
    return re.sub(r'\.\d{3}', '', attack_id)


def map_to_parent(df_attack_labels, attack_techniques_base):
    """
    use attack techniques to map labels to its parents
    Returns:
    """
    attack_dict = attack_techniques_base.set_index('attack_id')['attack_name'].to_dict()
    # drop attack_name column
    df_attack_labels_2 = df_attack_labels.drop(columns="attack_name")
    # remove digits using regular expression
    df_attack_labels_3 = df_attack_labels_2.copy()
    df_attack_labels_3["attack_id"] = df_attack_labels_3['attack_id'].apply(
        remove_digits)
    # Map attack names using attack_dict
    df_attack_labels_3['attack_name'] = df_attack_labels_3[
        'attack_id'].map(attack_dict)
    return df_attack_labels_3


def get_full_attack_name(attack_techniques, attack_id):
    attack_name = attack_techniques.loc[attack_techniques["attack_id"] == attack_id, "attack_name"].item()
    parent_name = attack_techniques.loc[attack_techniques["attack_id"] == attack_id, "parent_name"]

    if parent_name.isna().item():
        full_name = attack_name
    else:
        parent_name = parent_name.item()
        full_name = f"{parent_name}: {attack_name}"

    return full_name


def get_all_cvss_features(row):
    cvss_v2_features = get_cvss_features(row, version="cvssV2")
    cvss_v3_features = get_cvss_features(row, version="cvssV3")
    all_cvss_features = cvss_v2_features + cvss_v3_features
    return all_cvss_features


def retrieve_techniques_from_affected_object(data, affected_object,
                                             use_parent_techniques=True,
                                             mapping_type="exploitation_technique"):
    """
     Retrieve attack techniques associated with a specific affected object from the provided data.

    Args:
        data (dict): Data containing items with affected objects and their associated details.
        affected_object (str): The name of the affected object to search for.
        use_parent_techniques (bool, optional): If True, map techniques to their parent by removing sub-technique digits. Defaults to True.
        mapping_type (str, optional): The type of mapping to use for extracting techniques.
            Can be "exploitation_technique", "primary_impact", or "secondary_impact".
            For "primary_impact" or "secondary_impact", uses "example_impact" as the mapping type.

    Returns:
        list or None: A list of unique attack techniques associated with the affected object, or None if no match is found.
    """
    if mapping_type == "primary_impact" or mapping_type == "secondary_impact":
        mapping_type = "example_impact"

    for item in data['items']:
        if item['affected_object'] == affected_object:
            # Collect all exploitation_technique values
            techniques = []
            for detail in item['details']:
                #print(f"detail[mapping_type]: {detail[mapping_type]}")
                if detail[mapping_type] is None:
                    continue
                else:
                    techniques.extend(detail[mapping_type])
                # print(f"techniques: {techniques}")
            if use_parent_techniques:
                techniques = [remove_digits(technique) for technique in techniques]
            # Remove duplicates
            techniques = list(set(techniques))
            return techniques
    # Return None if no match is found
    return None


def postprocess(entry):
    """
    Postprocesses the predictions and ground truth labels for a single CVE entry.
    Args:
        entry: JSON entry with predictions and ground truth labels

    """
    # print(f"Postprocessing {entry}")
    if len(entry["true labels"]) == 0:
        true_labels = {"N/A"}
    else:
        true_labels = set(entry["true labels"])
    predictions = entry["predictions"]
    predictions = ["N/A" if item in ["None", None] else item for item in predictions]

    return true_labels, predictions


def compute_predictive_quality_metrics(data, k=10):
    """
    See: https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems#predictive-quality-metrics
    Args:
        data: JSON file with predictions and ground truth for each CVE
        k: cut off point

    Returns: precision@k, recall@k

    """
    num_relevant_at_k_total = 0
    num_relevant_total = 0

    for entry in data:
        true_labels, predictions = postprocess(entry)
        predictions = predictions[:k]
        relevant_at_k = true_labels.intersection(set(predictions))
        num_relevant_at_k = len(relevant_at_k)
        num_relevant = len(true_labels)
        num_relevant_at_k_total += num_relevant_at_k
        num_relevant_total += num_relevant

    precision_at_k_total = num_relevant_at_k_total / (k * len(data))
    recall_at_k_total = num_relevant_at_k_total / num_relevant_total
    return precision_at_k_total, recall_at_k_total


def compute_classwise_recall_at_k(data, k=10):
    """
    Computes classwise recall@k for each unique class in the dataset.

    Args:
        data: JSON file with predictions and ground truth for each CVE.
        k: Cut off point.

    Returns:
        A dictionary where keys are class labels and values are recall@k for each class.
    """
    classwise_recall = {}

    # Collect all unique true labels across the dataset
    all_classes = set()
    for entry in data:
        true_labels, _ = postprocess(entry)
        all_classes.update(true_labels)

    # Initialize recall counters for each class
    for cls in all_classes:
        classwise_recall[cls] = {'num_relevant_at_k': 0, 'num_relevant_total': 0}

    # Calculate recall@k for each class
    for entry in data:
        true_labels, predictions = postprocess(entry)
        predictions = predictions[:k]

        for cls in all_classes:
            if cls in true_labels:
                classwise_recall[cls]['num_relevant_total'] += 1
                if cls in predictions:
                    classwise_recall[cls]['num_relevant_at_k'] += 1

    # Compute recall@k for each class
    for cls in classwise_recall:
        num_relevant_at_k = classwise_recall[cls]['num_relevant_at_k']
        num_relevant_total = classwise_recall[cls]['num_relevant_total']
        classwise_recall[cls] = num_relevant_at_k / num_relevant_total if num_relevant_total > 0 else 0.0

    return classwise_recall


def average_precision_at_k(relevant_items, retrieved_items, k=None):
    """
    Calculate the average precision at k.

    :param relevant_items: A set of relevant item identifiers.
    :param retrieved_items: A list of retrieved item identifiers, ordered by relevance.
    :param k: The number of top items to consider.
    :return: The average precision at k.
    """
    if not relevant_items:
        return 0.0

    # Total number of relevant items in the ground truth
    # This is the correct denominator for AP
    total_relevant_in_ground_truth = len(relevant_items)

    if not k:
        k = len(retrieved_items)

    retrieved_items = retrieved_items[:k]
    num_relevant_found = 0
    sum_precision = 0.0

    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            sum_precision += precision_at_i
            # Workaround. LLM produces multiple N/A values inflating scores. The line below
            #   corrects this behavior
            if item == "N/A":
                break
    return sum_precision / total_relevant_in_ground_truth


def mean_average_precision(data, k=None):
    """
    Calculate the mean average precision for a set of queries.

    :param data: JSON file with predictions and ground truth for each CVE.
    :param k: The number of top items to consider.
    :return: The mean average precision score.
    """

    ap_sum = 0.0
    for entry in data:
        relevant_items, retrieved_items = postprocess(entry)
        ap_sum += average_precision_at_k(relevant_items, retrieved_items, k)

    return ap_sum / len(data)

def estimate_token_count(prompt):
    # Count the number of characters in the prompt
    num_characters = len(prompt)

    # Estimate the number of tokens (average 4 characters per token)
    estimated_tokens = num_characters / 4

    return estimated_tokens

def load_csv(path):
    if path:
        full_path = str(os.path.join(ROOT_DIR, path))
        sanitize_file_path(full_path)
        print(f"loading from path: {full_path}")
        df = pd.read_csv(full_path)
    else:
        df = None
    return df


def load_json(path):
    if path:
        full_path = os.path.join(ROOT_DIR, path)
        sanitize_file_path(full_path)
        with open(full_path, 'r') as file:
            data = json.load(file)
    else:
        data = None
    return data

def sanitize_file_path(full_path):
    if not full_path.startswith(os.path.abspath(ROOT_DIR)):
        raise ValueError("Invalid file path: Potential Directory traversal!")


def get_entry_value_of_predictions(target_cve, key, predictions):
    """
    For a given CVE, get the entry value of a key. For example get predictions or true labels.
    Args:
        predictions: JSON file with model predictions and true values
        target_cve:
        key: Decides the output. E.g "predictions" or "true labels"

    Returns:
        Values from key

    """
    for entry in predictions:
        if entry.get("target_cve") == target_cve:
            return entry.get(key)
    return None


def create_attack_csv(attack_techniques, include_attack_descriptions):
    """
    Creates a csv with attack features.

    Args:
        attack_techniques: A list of attack techniques to include
        include_attack_descriptions: Option to include descriptions

    Returns:

    """
    all_attack_ids = list(attack_techniques.loc[:, "attack_id"])
    attack_data = []
    for attack_id in all_attack_ids:
        full_attack_name = get_full_attack_name(attack_techniques, attack_id)
        attack_data.append((attack_id, full_attack_name))
    all_attack_names_df = pd.DataFrame(attack_data, columns=["attack_id",
                                                             "attack_name"])
    if include_attack_descriptions:
        all_attack_names_df = all_attack_names_df.merge(attack_techniques.loc[
                                                        :, ["attack_id", "description"]],
                                                        on="attack_id")
    all_attack_names_csv = all_attack_names_df.to_csv(index=False)
    return all_attack_names_csv


def get_role():
    return ("You are a cybersecurity analyst specialised in applying MITRE's ATT&CK Framework "
            "to label vulnerability descriptions.")


def get_dspy_module(dspy_module):
    if dspy_module == "chain_of_thought":
        qa = dspy.ChainOfThought('context, question -> answer')
    elif dspy_module == "predict":
        qa = dspy.Predict('question -> answer')
    else:
        raise NotImplementedError
    return qa


def compute_and_log_ranking_metrics(history, neptune_run):
    pre_at_5, rec_at_5 = compute_predictive_quality_metrics(history, k=5)
    pre_at_10, rec_at_10 = compute_predictive_quality_metrics(history, k=10)
    pre_at_20, rec_at_20 = compute_predictive_quality_metrics(history, k=20)
    map_score = mean_average_precision(history)
    neptune_run[f"metrics/Precision@5"] = pre_at_5
    neptune_run[f"metrics/Recall@5"] = rec_at_5
    neptune_run[f"metrics/Precision@10"] = pre_at_10
    neptune_run[f"metrics/Recall@10"] = rec_at_10
    neptune_run[f"metrics/Precision@20"] = pre_at_20
    neptune_run[f"metrics/Recall@20"] = rec_at_20
    neptune_run[f"metrics/MAP"] = map_score
    print(f"MAP: {map_score}")
    print(f"Precision@{10}: {pre_at_10}")
    print(f"Recall@{10}: {rec_at_10}")


def save_pred_to_csv(df, experiment_dir, mapping_method, tmp_dir):
    """
    Save output predictions to the experiment directory.
    Args:
        df: DataFrame with predictions
        ...

    """
    df.to_csv(os.path.join(experiment_dir, "pred.csv"), index=False)
    # Save temporary output to be used when running the full pipeline
    df.to_csv(os.path.join(ROOT_DIR, tmp_dir, f"{mapping_method}_pred.csv"), index=False)

def save_pred_to_json(history, experiment_dir, mapping_method, tmp_dir, mapping_type):
    """
    Save output predictions to the experiment directory.
    Args:
        history: JSON dump with predictions, true labels and reasoning
        ...
    """
    with open(os.path.join(experiment_dir, f"history.json"), "w") as json_file:
        json_file.write(history)
    # Save temporary output to be used when running the full pipeline
    with open(os.path.join(ROOT_DIR, tmp_dir, f"{mapping_method}_{mapping_type}.json"),
              "w") as json_file:
        json_file.write(history)


def transform_string(input_str):
    # Check if the input string is already in list format
    if input_str.startswith('[') and input_str.endswith(']'):
        # Use regex to find elements and add quotes around them
        transformed_str = re.sub(r'(?<=\[)(.*?)(?=\])', lambda x: ', '.join(
            f'"{item.strip()}"' for item in x.group(0).split(',')), input_str)
    else:
        # Assume it's a comma-separated string and transform it into a list format
        items = [f'"{item.strip()}"' for item in input_str.split(',')]
        transformed_str = f"[{', '.join(items)}]"

    return transformed_str


def match_pattern_from_sentence(text):
    # Updated pattern to match quoted/unquoted technique codes and None
    pattern = r"'?T\d{4}'?|None"
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    # Convert matches to a list of evaluated elements
    extracted_list = [match.strip("'") if match != 'None' else None for match in matches]
    return extracted_list


def replace_func(value, replacements):
    """
    Apply replacements to a given value based on the provided mappings.

    Args:
        value: The value to be modified.
        replacements (dict): A dictionary mapping old values to new values.

    Returns:
        The modified value with replacements applied.
    """
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value

def replace_values(df):
    """
    Replace specific string values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.

    Returns:
        pd.DataFrame: The modified DataFrame with replaced values.
    """
    # Define the replacement mappings for general values
    general_replacements = {
        '_': '-',
        '@': '-at-'
    }

    # Define the replacement mappings for column names
    column_replacements = {
        '10': 'ten',
        '5': 'five'
    }

    # Rename columns using both general and column-specific replacements
    new_columns = {}
    for col in df.columns:
        # Apply general replacements first
        new_col_name = replace_func(col, general_replacements)
        # Apply column-specific replacements if applicable
        new_col_name = replace_func(new_col_name, column_replacements)
        new_columns[col] = new_col_name

    df = df.rename(columns=new_columns)

    # Apply the general replacement function to each element in the DataFrame
    for col in df.columns:
        df[col] = df[col].apply(lambda x: replace_func(x, general_replacements))

    return df


def get_metric_name(metric, average):
    metric_name = f"{average}_{metric}"
    return metric_name


def filter_metric_columns(df, metric, average):
    """
    Filters columns in the DataFrame that match the specified metric and average.

    Args:
        df (pd.DataFrame): The DataFrame containing metric columns.
        metric (str): The metric to filter by. e.g. f1
        average (str): The averaging method to filter by.

    Returns:
        list: A list of column names that match the metric and average.
    """
    metric_name = get_metric_name(metric, average)
    metric_cols = [col for col in df.columns if metric_name in col]
    if average == "micro":
        extra_metric_name = f"/{metric}"
        extra_metric_cols = [col for col in df.columns if extra_metric_name in col]
        metric_cols.extend(extra_metric_cols)
    return metric_cols


def aggregate_column_values(df, metric_name, cols):
    """
    Aggregates values from specified columns into a new column and drops the old columns.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        metric_name (str): The name of the new aggregated column.
        cols (list): The list of columns to aggregate.

    Returns:
        pd.DataFrame: The modified DataFrame with aggregated column values.
    """
    df.loc[:, metric_name] = df[cols].bfill(axis=1).iloc[:, 0]
    # now we can drop the old columns
    # print(cols)
    df = df.drop(columns=cols)
    return df


def process_metrics(df, metrics=None, average_options=None):
    """
    Processes the DataFrame to aggregate metric columns based on specified metrics and averages.

    Args:
        df (pd.DataFrame): The DataFrame containing raw metric data.
        metrics (list, optional): List of metrics to process. Defaults to ["f1", "precision", "recall"].
        average_options (list, optional): List of averaging methods to use. Defaults to ["micro", "macro", "weighted"].

    Returns:
        pd.DataFrame: A new DataFrame with aggregated metric columns.
    """
    df_copy = df.copy()
    if metrics is None:
        metrics = ["f1", "precision", "recall"]
    if average_options is None:
        average_options = ["micro", "macro", "weighted"]
    for metric in metrics:
        for average in average_options:
            filter_cols_test = filter_metric_columns(df_copy, metric=metric, average=average)
            metric_name = get_metric_name(metric=metric, average=average)
            # print(metric_name)
            # print(filter_cols_test)
            # print(metric_name)
            df_copy = aggregate_column_values(df_copy, metric_name=metric_name, cols=filter_cols_test)
    return df_copy


def get_cvss_features(df, version=None):
    """
    Extract CVSS related columns from processed dataframe
    """
    cvss_columns = []
    if version == "cvssV2":
        column_names = ['cvssV2', 'baseMetricV2']
    elif version == "cvssV3":
        column_names = ['cvssV3', 'baseMetricV3']
    else:
        column_names = ['cvss', 'baseMetric']
    for column in df.columns:
        if column.startswith(column_names[0]) or column.startswith(column_names[1]):
            cvss_columns.append(column)
    return cvss_columns


