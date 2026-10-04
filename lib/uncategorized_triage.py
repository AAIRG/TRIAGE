from lib.utils import get_entry_value_of_predictions, load_json

def add_unique_element(index, lst, result, seen):
    """
    Add an element from the list to the result if it hasn't been seen.
    """
    if index < len(lst):
        element = lst[index]
        if element not in seen:
            result.append(element)
            seen.add(element)

def merge(l1, l2, l3=None, cut_off=None):
    """
    Merge three lists by interleaving their elements in a round-robin
    fashion, skipping elements that have already been included.
    """

    result = []
    seen = set()  # trade memory for efficiency using an extra set
    if l3 is not None:
        max_len = max(len(l1), len(l2), len(l3))
    else:
        max_len = max(len(l1), len(l2))
    for i in range(max_len):
        add_unique_element(i, l1, result, seen)
        add_unique_element(i, l2, result, seen)
        if l3 is not None:
            add_unique_element(i, l3, result, seen)
    if cut_off:
        result = result[:cut_off]

    return result



class Job:

    def __init__(self, data_handler, args):
        self.mapping_method = data_handler.mapping_method
        self.df_cve_split = data_handler.cves_split
        self.cve_ids = list(self.df_cve_split["CVE ID"])
        self.attack_techniques_base = data_handler.attack_techniques_base
        self.df_attack_labels = data_handler.df_attack_labels
        self.exclude_secondary_impact = data_handler.exclude_secondary_impact
        # Read experiment predictions for each mapping type
        self.exp_pred_data = load_json(args.path_exploitation_technique_predictions)
        self.prim_pred_data = load_json(args.path_primary_impact_predictions)
        self.sec_pred_data = load_json(args.path_secondary_impact_predictions)
        # Dspy
        self.dspy_module = args.dspy_module
        self.history = []


    def run_experiment(self):
        for cve in self.cve_ids:
            exp_pred = get_entry_value_of_predictions(cve, "predictions", self.exp_pred_data)
            prim_pred = get_entry_value_of_predictions(cve, "predictions", self.prim_pred_data)
            sec_pred = get_entry_value_of_predictions(cve, "predictions", self.sec_pred_data)
            df_attack_labels_sub = self.df_attack_labels.loc[self.df_attack_labels["CVE ID"] ==
                                                            cve]
            if self.exclude_secondary_impact:
                merged_pred = merge(exp_pred, prim_pred)
                # filter out secondary impact predictions
                df_attack_labels_sub = df_attack_labels_sub.loc[df_attack_labels_sub[
                    "mapping_type"] != "secondary_impact"]
            else:
                merged_pred = merge(exp_pred, prim_pred, sec_pred)
            true_attack_ids = list(df_attack_labels_sub["attack_id"])
            history_i = {
                "target_cve": cve,
                "predictions": merged_pred,
                "true labels": true_attack_ids
                }
            self.history.append(history_i)