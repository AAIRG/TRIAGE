import re

from lib.utils import (retrieve_techniques_from_affected_object,
                       get_entry_value_of_predictions)
from lib.core.other_methodology_mappers import retrieve_exp_mapping, get_sub_qs_w_ans


def get_exploit_method_pred(cve, df_pred, exploitation_method_mapping):
    row_1 = df_pred.loc[df_pred['CVE ID'] ==
                             cve]
    # Identify columns containing the value "NO"
    columns_to_drop = row_1.columns[row_1.isin(['NO']).any()]

    # Drop those columns
    row_cleaned = row_1.drop(columns=columns_to_drop)

    # Drop columns with NaN values, this can be the sub questions
    row_cleaned_2 = row_cleaned.dropna(axis='columns', how='all')

    row_cleaned_2 = row_cleaned_2.drop(columns="CVE ID")

    # Check if sub questions columns is present and if so, store the values of these in a dictionary
    sub_q_dict = get_sub_qs_w_ans(row_cleaned_2)
    # Now remove the subquestions which should not be iterated over in the for loop below
    row_cleaned_3 = row_cleaned_2.drop(columns=sub_q_dict.keys())

    if row_cleaned_3.empty:
        techniques = []
        return techniques
    # Iterate relevant columns to retrieve techniques
    # iterating over positive short questions
    for col in row_cleaned_3.columns:
        if col == "Mal. file?":
            # Check where the file came from
            option = sub_q_dict["File come from?"]
            techniques = retrieve_exp_mapping(exploitation_method_mapping, q_short=col,
                                              option=option)
        elif col == "Mal. link?":
            option = sub_q_dict["Link come from?"]
            techniques = retrieve_exp_mapping(exploitation_method_mapping, q_short=col,
                                              option=option)
        else:
            techniques = retrieve_exp_mapping(exploitation_method_mapping, q_short=col)
    return techniques


def get_tactic_technique_pred(df_tactic_pred):
    tactic_to_technique = {"Initial Access": "T1190",
                           "Execution": "T1203",
                           "Privilege Escalation": "T1068",
                           "Defense Evasion": "T1211",
                           "Credential Access": "T1212",
                           "Lateral Movement": "T1210"}
    # Apply the function to extract the desired tactic words
    tactics  = tactic_to_technique.keys()
    def extract_keyword(text):
        for keyword in tactics:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                return keyword
        return None
    df_tactic_pred['tactic_extracted'] = df_tactic_pred['tactic'].apply(extract_keyword)
    # Drop old tactic
    df_tactic_pred = df_tactic_pred.drop(columns=['tactic'])
    # rename the new tactic
    df_tactic_pred = df_tactic_pred.rename(columns={'tactic_extracted': 'tactic'})

    # now add column with techniques
    df_pred_final = df_tactic_pred.copy()
    df_pred_final['attack_id'] = df_pred_final['tactic'].map(tactic_to_technique)
    df_pred_final['mapping_type'] = "exploitation_technique"
    return df_pred_final


class Job:
    def __init__(self, data_handler):
        self.mapping_type = data_handler.mapping_type
        self.mapping_method = data_handler.mapping_method
        self.use_parent_techniques = data_handler.use_parent_techniques
        self.include_vul_type = data_handler.include_vul_type
        self.include_exploitation_method = data_handler.include_exploitation_method
        self.include_affected_object = data_handler.include_affected_object
        self.exploitation_method_mapping = data_handler.exploitation_method_mapping
        self.affected_object_mapping = data_handler.affected_object_mapping
        self.in_context_learner_pred = data_handler.in_context_learner_pred
        self.df_exp_pred = data_handler.exploitation_method_predictions
        self.df_affected_object_pred = data_handler.affected_object_predictions
        self.df_vul_final = data_handler.vulnerability_type_predictions_mapped
        self.cves_split = self.get_combined_cves()

    def get_combined_cves(self):
        """
        Collects and combines all unique CVE IDs from included prediction sources.

        Returns:
            list: Combined set of all unique CVE IDs from the included sources.
        """
        cve_sets = []
        if self.include_exploitation_method:
            cve_sets.append(set(self.df_exp_pred["CVE ID"]))
        if self.include_affected_object:
            cve_sets.append(set(self.df_affected_object_pred["CVE ID"]))
        if self.include_vul_type:
            cve_sets.append(set(self.df_vul_final["CVE ID"]))
        # Always include in_context_learner_pred
        cve_sets.append(self.get_all_in_context_learner_cves())
        combined_cves = set().union(*cve_sets)
        combined_cves = list(combined_cves)
        return combined_cves

    def get_all_in_context_learner_cves(self):
        """
        Retrieve all CVE IDs from the JSON file of in_context_learner_predictions.

        Returns:
            List of all CVE IDs present in the predictions.
        """
        in_context_learner_cves = [entry.get("target_cve") for entry in
                                 self.in_context_learner_pred if "target_cve" in entry]
        in_context_learner_cves = set(in_context_learner_cves)
        return in_context_learner_cves


    def get_methodology_techniques(self, cve):
        """
        NB: Intermediate labels are already mapped to attack techniques for vulnerability type but
        not for affected object or exploitation technique
        Args:
            cve:

        Returns:

        """
        exp_techniques = None
        vul_techniques = None
        affected_object_techniques = None
        if self.include_exploitation_method:
            exp_techniques = get_exploit_method_pred(cve, self.df_exp_pred,
                                                     self.exploitation_method_mapping)
        if self.include_affected_object:
            affected_object = self.df_affected_object_pred.loc[self.df_affected_object_pred["CVE ID"] ==
                                                           cve]["pred"].values
            affected_object_techniques = retrieve_techniques_from_affected_object(
                self.affected_object_mapping, affected_object, self.use_parent_techniques, self.mapping_type)
        if self.include_vul_type:
            df_vul_sub = self.df_vul_final.loc[self.df_vul_final["CVE ID"] == cve]
            vul_techniques = list(df_vul_sub["attack_id"])
        methodology_techniques = set()
        for techniques in [exp_techniques, vul_techniques, affected_object_techniques]:
            if techniques is not None:
                methodology_techniques.update(techniques)
        return methodology_techniques

    def heuristic_combine(self, cve, methodology_techniques, in_context_techniques, true_labels):
        diff = methodology_techniques - set(in_context_techniques)
        combined_predictions = in_context_techniques.copy()
        if diff:
            # Find none
            found_none = False
            for technique in in_context_techniques:
                if technique is None:
                    found_none = True
                    pos_none = in_context_techniques.index(technique)
                    # remove none from combined predictions
                    combined_predictions.pop(pos_none)
                    # We are only interested in the first "None"
                    break
            replace_techniques = list(diff)
            replace_pos = len(combined_predictions) - 1
            count = 0
            for technique in replace_techniques:
                count += 1
                combined_predictions[replace_pos] = technique
                replace_pos -= 1
                # Possible future extension to prioritize which methods to use.
            # Insert None back if it exists
            if found_none:
                combined_predictions.insert(pos_none, None)
                # print(f"CVE: {cve}")
                # print(f"combined_predictions: {combined_predictions}")
                # print(f"diff: {diff}")
        history_element = {
            "target_cve": cve,
            "methodology_techniques": list(methodology_techniques),
            "in_context_techniques": in_context_techniques,
            "diff": list(diff),
            "predictions": combined_predictions,
            "true labels": true_labels
        }
        return history_element


    def run_experiment(self):
        history = []
        for cve in self.cves_split:
            methodology_techniques = self.get_methodology_techniques(cve)
            if self.mapping_method == "combined_approach":
                in_context_techniques_raw = get_entry_value_of_predictions(cve, "predictions",
                                                                         self.in_context_learner_pred)
                in_context_techniques = [None if technique in ["None", None] else technique for
                                       technique in in_context_techniques_raw]
                true_labels = get_entry_value_of_predictions(cve, "true labels",
                                                             self.in_context_learner_pred)
                history_element = self.heuristic_combine(cve, methodology_techniques,
                                                         in_context_techniques, true_labels)
                history.append(history_element)
            elif self.mapping_method == "methodology_combine":
                history.append(list(methodology_techniques))
            else:
                ValueError("Mapping method not recognized.")
        return history

