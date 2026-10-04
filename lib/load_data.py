import os
import pandas as pd
import json

from settings import ROOT_DIR
from lib.utils import (map_to_parent, remove_digits, map_intermediate_labels, load_json, load_csv,
                                              create_attack_csv, sanitize_file_path)


def load_exploit_method_mapping(args):
    json_data = load_json(args.path_exploitation_method_mapping)
    if args.use_parent_techniques:
        json_data = process_techniques_in_dict(json_data)
    return json_data


def process_techniques_in_dict(data):
    """
    Processes a dictionary of questions to normalize attack technique ids by removing sub-technique digits.

    For each question in the input data, this function applies the `remove_digits` utility to all technique ids,
    both at the question level and within each answer (if present). This ensures that all technique IDs are mapped to their parent.

    Args:
        data (dict): A dictionary containing a list of questions, each with possible 'techniques' and/or 'answers' fields.

    Returns:
        dict: The modified data dictionary with normalized technique ids.
    """
    for question in data['questions']:
        # Process answers if they exist
        if 'answers' in question:
            for answer in question['answers']:
                answer['techniques'] = [remove_digits(t) for t in answer['techniques']]

        # Process techniques directly under the question
        if 'techniques' in question:
            question['techniques'] = [remove_digits(t) for t in question['techniques']]

    return data


class BaseDataHandler:
    """
    Reading and processing data
    """
    def __init__(self, args, mapping_method=None):
        self.use_parent_techniques = args.use_parent_techniques
        self.mapping_type = args.mapping_type
        self.mapping_method = mapping_method
        self.attack_techniques_base = pd.read_csv(str(os.path.join(ROOT_DIR,
                                                             args.path_techniques_base)),
                                index_col=False)
        self.df_attack_labels = self.load_attack_labels()

        # Using enterprise techniques
        self.attack_techniques = self.attack_techniques_base.copy()
        self.attack_techniques = self.attack_techniques.loc[self.attack_techniques[
            "attack_matrix"] == "enterprise-attack"]
        if self.use_parent_techniques:
            # Drop subtechniques
            self.attack_techniques = self.attack_techniques.loc[self.attack_techniques[
                "category"] == "technique"]

        self.cve_base = self.load_cve_base(args)
        self.validate_cves_in_attack_labels()
        self.data_split = args.data_split
        self.num_cves = args.num_cves
        self.random_seed = args.random_seed
        self.vulnerability_type_predictions = None
        self.functionality_predictions = None
        self.exploitation_method_predictions = None
        self.tactic_level_predictions = None
        self.affected_object_predictions = None
        self.in_context_learner_pred = None
        # Loading values for above variables
        self.include_in_context_learner = args.include_in_context_learner
        self.include_vul_type = args.include_vul_type
        self.include_functionality = args.include_functionality
        self.include_exploitation_method = args.include_exploitation_method
        self.include_tactic = args.include_tactic
        self.include_affected_object = args.include_affected_object
        self.include_cwe = args.include_cwe
        self.include_cvss = args.include_cvss
        self.load_existing_predictions(args)
        self.cves_train = self.load_cves("train")
        self.cves_test = self.load_cves("test")
        self.cves_split = self.get_cves_split()
        self.cve_id_to_desc = self.get_cve_id_to_desc(args.data_split)
        self.df_functionality_mapping = self.load_functionality_mapping(args)
        self.use_functionality_demos = args.use_functionality_demos
        self.functionality_labels = self.load_functionality_labels(args.path_functionality_labels)
        self.df_vulnerability_mapping = self.load_vulnerability_mapping(args)
        self.df_cwe_id_to_name = pd.read_csv(str(os.path.join(ROOT_DIR, args.path_cwe_id_to_name)))
        self.df_cwe = self.load_cwe_data(args.path_cwe_id_to_description)
        self.df_vul_type_to_cwe = pd.read_csv(str(os.path.join(ROOT_DIR, args.path_vulnerability_type_to_cwe)))
        self.cwe_name_to_methodology = self.get_methodology_name()
        self.vul_dict = self.get_vul_dict()
        self.func_name_to_desc = self.get_func_name_to_desc()
        self.exploitation_method_mapping = (
            load_exploit_method_mapping(args))
        self.affected_object_mapping = load_json(args.path_affected_object_mapping)
        self.vulnerability_type_predictions_mapped = map_intermediate_labels(
            self.df_vulnerability_mapping, self.vulnerability_type_predictions,
            "vulnerability_type",
            self.mapping_type)
        self.num_demonstrations = args.num_demonstrations
        self.all_attack_names_csv = create_attack_csv(self.attack_techniques, args.include_attack_descriptions)
        self.exclude_secondary_impact = args.exclude_secondary_impact


    def get_cves_split(self):
        if self.data_split == "train":
            cves_split = self.cves_train
        else:
            cves_split = self.cves_test
        if self.num_cves:
            cves_split = cves_split.sample(n=self.num_cves, random_state=self.random_seed)
        return cves_split


    def load_existing_predictions(self, args):
        # Included by default, Change to JSON
        if self.include_in_context_learner:
            if args.path_in_context_learner_predictions:
                self.in_context_learner_pred = load_json(args.path_in_context_learner_predictions)
            else:
                self.in_context_learner_pred = load_json(os.path.join(args.tmp_dir,
                                                                      f"in_context_learner_"
                                                                      f"{self.mapping_type}.json"))
        if self.include_vul_type:
            if args.path_vulnerability_type_predictions:
                self.vulnerability_type_predictions = load_csv(args.path_vulnerability_type_predictions)
            else:
                self.vulnerability_type_predictions = load_csv(os.path.join(args.tmp_dir,
                                                                            f"vulnerability_type_pred.csv"))
        if self.include_functionality:
            if args.path_functionality_predictions:
                self.functionality_predictions = load_csv(args.path_functionality_predictions)
            else:
                self.functionality_predictions = load_csv(os.path.join(args.tmp_dir, f"functionality_pred.csv"))
        if self.include_exploitation_method:
            if args.path_exploitation_method_predictions:
                self.exploitation_method_predictions = load_csv(args.path_exploitation_method_predictions)
            else:
                self.exploitation_method_predictions = load_csv(os.path.join(args.tmp_dir,
                                                                         f"exploitation_technique_pred.csv"))
        if self.include_affected_object:
            if args.path_affected_object_predictions:
                self.affected_object_predictions = load_csv(args.path_affected_object_predictions)
            else:
                self.affected_object_predictions = load_csv(os.path.join(args.tmp_dir,
                                                                         f"affected_object_pred.csv"))
        if self.include_tactic:
            if args.path_tactic_level_predictions:
                self.tactic_level_predictions = load_csv(args.path_tactic_level_predictions)
            else:
                self.tactic_level_predictions = load_csv(os.path.join(args.tmp_dir,
                                                                         f"tactic_technique_pred.csv"))


    def load_cve_base(self, args):
        cve_base = load_csv(args.path_cve_base)
        # Drop unnecessary cvss columns
        drop_columns = ["cvssV2.version", "cvssV2.vectorString", "baseMetricV2.acInsufInfo",
                        "cvssV3.version", "cvssV3.vectorString"]
        cve_base = cve_base.drop(columns=drop_columns)
        return cve_base


    def get_methodology_name(self):
        df_merged = self.df_vul_type_to_cwe.merge(self.df_cwe_id_to_name, on="CWE ID",
                                                        how="left")
        # make dict CWE name to methodology
        cwe_to_vul_type = dict(zip(df_merged['CWE Name'], df_merged['Vulnerability Type']))

        return cwe_to_vul_type


    def load_cves(self, data_split):
        if data_split == "train":
            cves_split = load_csv("data/pre_processed/cves_train.csv")
        elif data_split == "test":
            cves_split = load_csv("data/pre_processed/cves_test.csv")
        else:
            raise NotImplementedError
        return cves_split


    def load_attack_labels(self):
        df_attack_labels = load_csv("data/pre_processed/labeled_cve_to_attack.csv")
        if self.use_parent_techniques:
            df_attack_labels = map_to_parent(df_attack_labels, self.attack_techniques_base)

        return df_attack_labels


    def get_cve_id_to_desc(self, data_split=None):
        cve_id_to_desc = dict(zip(self.cve_base['CVE ID'], self.cve_base['description']))
        # if data_split == "train":
        #     cve_id_to_desc = {cve_id: cve_id_to_desc[cve_id] for cve_id in self.cves_split[("CVE "
        #                                                                                     "ID")]}
        return cve_id_to_desc

    def validate_cves_in_attack_labels(self):
        """
        Validate that cves in the attack labels exist in the base of all cves
        Returns:

        """
        # Check that cves_attack_labels is a subset of cve_base
        all_cves = self.cve_base['CVE ID'].unique()
        cves_attack = self.df_attack_labels['CVE ID'].unique()

        # Check that cves_attack is a subset of cves_nvd
        assert set(cves_attack).issubset(
            set(all_cves)), "CVEs in df_attack_labels are not a subset of cve base"

    def load_functionality_mapping(self, args):
        df_mapping = load_csv(args.path_functionality_mapping)
        # Take out Install app from the mapping
        df_mapping = df_mapping.loc[df_mapping['Functionality'] != "Install App"]
        if self.use_parent_techniques:
            df_mapping["attack_id"] = df_mapping["attack_id"].apply(remove_digits)
        return df_mapping

    def load_vulnerability_mapping(self, args):
        df_mapping = load_csv(args.path_vulnerability_type_mapping)
        if self.use_parent_techniques:
            df_mapping["attack_id"] = df_mapping["attack_id"].apply(remove_digits)
        return df_mapping

    def load_functionality_labels(self, path_functionality_labels):
        # Read functionality labels, this are the demonstrations
        df_func_labels = load_csv(path_functionality_labels)
        # Add target value. All are positive examples
        df_func_labels['target_value'] = 1
        return df_func_labels

    def load_cwe_data(self, path_cwe_id_to_description):
        df_cwe_id_to_desc = load_csv(path_cwe_id_to_description)
        df_cwe = self.df_cwe_id_to_name.merge(df_cwe_id_to_desc, on='CWE ID')
        return df_cwe

    def get_vul_dict(self):
        vul_dict = dict(
            zip(self.df_cwe['CWE Name'], self.df_cwe['Description']))
        return vul_dict

    def get_func_name_to_desc(self):
        """
        Create a dictionary of functionality names and corresponding descriptions
        Returns:

        """
        func_name_to_desc = dict(zip(self.df_functionality_mapping['Functionality'],
                                     self.df_functionality_mapping['Description']))
        return func_name_to_desc


