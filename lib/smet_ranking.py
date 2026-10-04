from lib.utils import get_entry_value_of_predictions, load_json, parse_arguments
import os

"""
Load SMET predictions and true labels
"""

class Job:
    def __init__(self, data_handler, args):
        self.mapping_method = data_handler.mapping_method
        self.data_split = data_handler.data_split
        self.df_cve_split = data_handler.cves_split
        self.cve_ids = list(self.df_cve_split["CVE ID"])
        self.attack_techniques_base = data_handler.attack_techniques_base
        self.df_attack_labels = data_handler.df_attack_labels
        self.exclude_secondary_impact = args.exclude_secondary_impact
        self.uncategorized_pred_data = self.load_pred_path(args.dir_SMET_predictions)
        # Dspy
        self.history = []

    def load_pred_path(self, dir_SMET_predictions):
        if self.data_split == "train":
            pred = load_json(os.path.join(dir_SMET_predictions, "train_mappings.json"))
        elif self.data_split == "test":
            pred = load_json(os.path.join(dir_SMET_predictions, "test_mappings.json"))
        else:
            raise ValueError("Invalid data split")
        return pred


    def run_experiment(self):
        # Postprocessing to simplify analysis and computation of performance metrics
        for cve in self.cve_ids:
            pred = get_entry_value_of_predictions(cve, "predictions", self.uncategorized_pred_data)
            df_attack_labels_sub = self.df_attack_labels.loc[self.df_attack_labels["CVE ID"] ==
                                                            cve]
            if self.exclude_secondary_impact:
                df_attack_labels_sub = df_attack_labels_sub.loc[df_attack_labels_sub[
                    "mapping_type"] != "secondary_impact"]
            true_attack_ids = list(df_attack_labels_sub["attack_id"])
            history_i = {
                "target_cve": cve,
                "predictions": pred,
                "true labels": true_attack_ids
                }
            self.history.append(history_i)