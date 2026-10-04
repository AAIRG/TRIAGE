import pandas as pd
import json
import ast

from sentence_transformers import SentenceTransformer

from lib.utils import (get_full_attack_name, get_all_cvss_features,
                       estimate_token_count, get_role,
                       get_dspy_module, transform_string,
                       match_pattern_from_sentence)

def compute_similarity(target_description, other_descriptions, model_name=None):
    """

    Args:
        model_name: Name of the Sentence transformer model
        target_description: str
        other_descriptions: list[str]

    Returns:

    """
    if model_name:
        model = SentenceTransformer(model_name)
    else:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    target_emb = model.encode(target_description)
    other_embs = model.encode(other_descriptions)
    similarities = [model.similarity(target_emb, other_emb) for other_emb in other_embs]
    return similarities


class MetaInfoCVE:
    """
    The class is designed to handle and process Common Vulnerabilities and Exposures (CVE) data
    to be used in a prompt.
    """
    def __init__(self, data_handler):
        self.cve_base = data_handler.cve_base
        self.df_cwe_id_to_name = data_handler.df_cwe_id_to_name
        self.all_attack_names_csv = data_handler.all_attack_names_csv
        self.cwe_name_to_methodology = data_handler.cwe_name_to_methodology
        self.df_vulnerability_mapping = data_handler.df_vulnerability_mapping
        self.mapping_type = data_handler.mapping_type
        self.attack_techniques = data_handler.attack_techniques
        self.include_cwe = data_handler.include_cwe
        self.include_cvss = data_handler.include_cvss


    def meta_info_prompt(self, cve, all_attack_names_csv, attack_techniques=None):
        row = self.cve_base.loc[self.cve_base["CVE ID"] == cve, :]
        header = (
            f"The vulnerability to be labelled has the following description:\n"
            f"----\n"
            f"{row['description'].item()}\n"
            "----"
        )
        # print(header)
        # NB: Not sure if this should be placed here
        attack_techniques_data = ("The data in CSV format below describes the attack techniques "
                                  "to choose "
                                  "from:\n "
                                  "----\n"
                                  f"{all_attack_names_csv}\n"
                                  "----")
        # print("ATTACK TECHNIQUES DATA")
        # print(attack_techniques_data)
        if self.include_cwe:
            cwe_id = row["CWE ID"].item()
            df_cwe = self.df_cwe_id_to_name.loc[self.df_cwe_id_to_name["CWE ID"] == cwe_id]
            if df_cwe.empty:
                vul_type_prompt = ("There is no info about the CWE of this vulnerability and therefore "
                                 "no info about the methodology type.")
            else:
                cwe_name = df_cwe["CWE Name"].item()
                vul_type_prompt = (f"The following CWE applies to this vulnerability:\n"
                            "----\n"
                            f"{cwe_id}: {cwe_name}\n"
                            "----\n")
            vul_type_prompt = f"\n{vul_type_prompt}\n"
        else:
            vul_type_prompt = ""

        #print("METHODOLOGY PROMPT")
        #print(methodology_prompt)
        if self.include_cvss:
            cvss_features = get_all_cvss_features(row)
            cvss_context = ["The CVE has the following CVSS features:", "----"]
            for i in cvss_features:
                cvss_context.append(f" - {i}: {row[i].item()}")
            cvss_context.append("----")
            cvss_context = "\n".join(cvss_context)
            cvss_context = f"\n{cvss_context}\n"
        else:
            cvss_context = ""
        # cvss_context = "\n----".join(cvss_context)
        all_context = (f"\n{header}\n{attack_techniques_data}{vul_type_prompt}"
                       f"{cvss_context}")
        return all_context

class InContextLearner:
    """
    The class is designed to facilitate in-context learning by generating prompts that include
    examples of labeled vulnerabilities with their associated attack mappings
    """
    def __init__(self, data_handler):
        self.df_attack_labels = data_handler.df_attack_labels
        self.cve_base = data_handler.cve_base
        self.df_cves_train = data_handler.cves_train
        self.attack_techniques = data_handler.attack_techniques
        self.mapping_type = data_handler.mapping_type
        self.num_demonstrations = data_handler.num_demonstrations
        self.mapping_method = data_handler.mapping_method

    def get_similar_demonstrations(self, cve, other_cves, embedding_model):
        """
        Find the most similar CVEs to a target CVE based on description embeddings.

        Args:
            cve (str): The CVE ID of the target vulnerability.
            other_cves (list): List of other CVE IDs to compare against.
            embedding_model (str): Name of the sentence transformer model to use for similarity computation.

        Returns:
            list: A list of CVE IDs corresponding to the most similar demonstrations, sorted by similarity.
        """
        target_description = self.cve_base.loc[self.cve_base['CVE ID'] ==
                                               cve, 'description'].values[0]
        # Get descriptions of other cves
        other_descriptions = self.cve_base.loc[
            self.cve_base['CVE ID'].isin(other_cves), 'description'].values
        # Compute similarity
        similarities = compute_similarity(target_description, other_descriptions,
                                          model_name=embedding_model)
        # create a dataframe with CVE and similarity (compared to target)
        df_similarity = pd.DataFrame({'CVE ID': list(other_cves), 'similarity': similarities})
        # Sort by similarity
        df_similarity = df_similarity.sort_values(by='similarity', ascending=False)
        # Get the most relevant CVEs
        top_n = df_similarity.iloc[:self.num_demonstrations]
        relevant_cves = top_n['CVE ID'].tolist()
        return relevant_cves

    def get_demos(self, cve, embedding_model):
        """
        Retrieve demonstration CVEs (In-Context learning examples) for a given target CVE,
        optionally selecting the most similar ones.

        Args:
            cve (str): The CVE ID of the target vulnerability.
            embedding_model (str): Name of the sentence transformer model to use for similarity computation.

        Returns:
            list: A list of CVE IDs to be used as demonstrations.
        """
        other_cves = set(self.df_cves_train["CVE ID"]) - {cve}
        other_cves = list(other_cves)
        # NB: if below is "None", it means that all possible demonstrations are used.
        if self.num_demonstrations is not None:
            relevant_cves = self.get_similar_demonstrations(cve, other_cves, embedding_model)
        else:
            relevant_cves = other_cves
        return relevant_cves

    def in_context_prompt(self, relevant_cves):
        header = (f"\nThe following is a JSON with examples. Each entry corresponds to a "
                  f"vulnerability which has a description and associated attack techniques by "
                  f"mapping type: \n "
                  "---- \n")
        uncategorized_header = (f"\nThe following is a JSON with examples. Each entry corresponds "
                                f"to a vulnerability which has a description and associated attack "
                                f"techniques: \n "
                                "---- \n")
        ending = "\n----\n"
        cves_json = self.create_json(relevant_cves)
        cves_json = json.dumps(cves_json, indent=4)
        if self.mapping_method == "uncategorized_ranking":
            # This method does not consider the mapping type
            prompt = uncategorized_header + cves_json + ending
        else:
            prompt = header + cves_json + ending
        return prompt

    def create_json(self, cve_ids, uncategorized=False):
        """
        Creates a JSON representation of in-context examples.
        Includes CVE ID, CVE descriptions and associated attack techniques.

        Args:
            uncategorized: Excluding mapping type labels.
            cve_ids (list[str]): A list of CVE IDs to process and include in the JSON output.

        Returns:
            dict: A dictionary where each key is a CVE ID, and each value is a dictionary containing
            the CVE's description and a mapping of attack techniques categorized by their mapping types.
        """
        df_attack_labels = self.df_attack_labels
        cve_base = self.cve_base
        attack_techniques = self.attack_techniques

        cves_json = {}

        for cve_id in cve_ids:
            cves_json[cve_id] = {}
            # # Add description
            description = cve_base.loc[cve_base['CVE ID'] == cve_id, 'description'].values[0]
            # Replace all whitespace in the description with a single space
            description = ' '.join(description.split())
            cves_json[cve_id]['description'] = description
            attack_techniques_json = {}

            # Populate attack techniques
            cve_attacks = df_attack_labels[df_attack_labels['CVE ID'] == cve_id]

            for mapping_type, group in cve_attacks.groupby('mapping_type'):
                attack_list = [{'attack_id': row['attack_id'], 'attack_name': get_full_attack_name(
                    attack_techniques, row['attack_id'])} for index, row in group.iterrows()]
                attack_techniques_json[mapping_type] = attack_list

            if uncategorized:
                attack_list = [{'attack_id': row['attack_id'], 'attack_name':
                    get_full_attack_name(attack_techniques, row['attack_id'])} for index, row in
                      uncategorized_attacks.iterrows()]
                attack_techniques_json = attack_list
            cves_json[cve_id]['attack_techniques'] = attack_techniques_json

        return cves_json


class Job:
    """
    Handle experiment
    """
    def __init__(self, data_handler, neptune_run, args):
        self.df_cve_split = data_handler.cves_split
        self.cve_ids = list(self.df_cve_split["CVE ID"])
        self.meta_info_cve = MetaInfoCVE(data_handler)
        self.in_context_learner = InContextLearner(data_handler)
        self.attack_techniques = data_handler.attack_techniques
        self.all_attack_names_csv = data_handler.all_attack_names_csv
        self.mapping_type = data_handler.mapping_type
        self.dspy_module = args.dspy_module
        self.qa = get_dspy_module(self.dspy_module)
        self.neptune_run = neptune_run
        self.reasoning_history = []
        self.answers = []
        self.demo_history = []
        self.question_history = []
        self.context = get_role()
        self.include_in_context_prompt = args.include_in_context_prompt
        self.embedding_model = args.embedding_model
        self.use_ranking_approach = args.use_ranking_approach


    def get_task_description_basic(self):
        header = (f"An attack consists of three steps which corresponds to the following mapping "
                  f"types: \n"
                  "- Exploitation Technique - the method (technique) used to exploit the "
                  "vulnerability\n"
                  "- Primary Impact - the initial benefit (impact) gained through exploitation of the vulnerability\n"
                  "- Secondary Impact - what the adversary can do by gaining the benefit of the "
                  "primary impact\n")
        task = ("Given a vulnerability, your task is to determine the relevant attack "
                  f"techniques of type {self.mapping_type}."
                  f"There can be cases with multiple relevant attack techniques, "
                  f"and cases without any attack technique for the given type."
                  " In the remainder you will first receive the vulnerability to label, "
                  "then an overview of attack techniques to use as labels. ")
        prompt = header + task
        if self.include_in_context_prompt:
            optional =  "Finally, you receive examples of correctly labeled vulnerabilities."
            prompt += optional
        return prompt

    def specify_output_basic(self):
        return ("Your task is to determine the relevant attack techniques of type"
                f" {self.mapping_type} for the given vulnerability. Provide a list of one or more attack "
                f"techniques based on their IDs, for example ['T1068', 'T1190'].")


    def get_task_description_ranking_approach(self):
        header = (f"An attack consists of three steps which corresponds to the following mapping "
                  f"types: \n"
                  "- Exploitation Technique - the method (technique) used to exploit the "
                  "vulnerability\n"
                  "- Primary Impact - the initial benefit (impact) gained through exploitation of the vulnerability\n"
                  "- Secondary Impact - what the adversary can do by gaining the benefit of the "
                  "primary impact\n")
        task = ("Given a vulnerability, your task is to determine the relevant attack "
                  f"techniques of type {self.mapping_type}."
                  f"You should output the top 10 most relevant attack techniques in descending "
                  f"order."
                  f" An empty label indicating no relevant attack technique can be included among "
                  f"the top 10 where appropriate."
                  " In the remainder you will first receive the vulnerability to label, "
                  "then an overview of attack techniques to use as labels. ")
        prompt = header + task
        if self.include_in_context_prompt:
            optional =  "Finally, you receive examples of correctly labeled vulnerabilities."
            prompt += optional
        return prompt


    def specify_output_ranking_approach(self):
        return ("\nYour task is to determine the relevant attack techniques of type"
        f" {self.mapping_type} for the given vulnerability. Provide a ranked list of either ten "
                f"technique IDs or nine technique IDs and one 'None' value indicating empty label. "
                f"The 'None' value should be ranked similarly as the other techniques when "
                f"applicable. "
                f"Here is an example of predicted output "
                f"['T1068', None, 'T1168', 'T1290', 'T1078', 'T1180', 'T1010', "
                f"'T1435', 'T1320', 'T1100']. The output should have the same format used in "
                f"the example, and an extra explanation should not be included.")


    def run_experiment(self):
        """
        Iterate over cves (cves train). Get components from the method classes above.
        Returns:

        """
        all_predictions = []
        count = 0
        length = len(self.cve_ids)
        context = self.context
        for cve in self.cve_ids:
            if self.use_ranking_approach:
                question = self.get_task_description_ranking_approach()
            else:
                question = self.get_task_description_basic()
            count += 1
            # context = ""
            print(f"count: {count} of {length}")
            print(f"Evaluating: {cve}")
            row_data = {"CVE ID": cve}
            cve_meta_info = self.meta_info_cve.meta_info_prompt(cve, self.all_attack_names_csv,
                                                                self.attack_techniques)
            # print(cve_meta_info)
            question = f"{question}\n{cve_meta_info}"
            if self.include_in_context_prompt:
            # get demos
                relevant_cves = self.in_context_learner.get_demos(cve, self.embedding_model)
                self.demo_history.append(relevant_cves)
                in_context_prompt = self.in_context_learner.in_context_prompt(relevant_cves)

                question += in_context_prompt
            # Make last specification of the question
            if self.use_ranking_approach:
                question += self.specify_output_ranking_approach()
            else:
                question += self.specify_output_basic()
            #print(question)

            # Using dspy predict module
            if self.dspy_module == "predict":
                prompt = f"{context}\n\n{question}"
                # print(prompt)
                pred = self.qa(question=prompt)
            # Using dspy COT
            else:
                pred = self.qa(question=question, context=context)
                self.reasoning_history.append(pred.reasoning)
            print(f"pred.answer: {pred.answer}")
            # Evaluate if an answer as a string representation e.g list is returned
            if pred.answer:
                try:
                    answer = ast.literal_eval(pred.answer)
                except SyntaxError:
                    print("Encountered a SyntaxError")
                    matched_pattern = match_pattern_from_sentence(pred.answer)
                    # Convert None values to 'None' strings for proper evaluation
                    formatted_patterns = [str(item) if item is not None else 'None' for item in
                                          matched_pattern]
                    # Ensure matched_pattern is a string representation of a list
                    transformed_str = transform_string(', '.join(formatted_patterns))
                    print(f"transformed_str: {transformed_str}")
                    answer = ast.literal_eval(transformed_str)
                except ValueError:
                    print("Encountered a ValueError")
                    # Transform string
                    try:
                        transformed_str = transform_string(pred.answer)
                        print(f"transformed_str: {transformed_str}")
                        answer = ast.literal_eval(
                            transformed_str)  # Evaluate the transformed string

                    except ValueError:
                        print("Incorrect output format! Continuing...")
                        answer = None
            else:    # Handles empty "" and None
                answer = None
            print(f"final answer: {answer}")
            self.answers.append(answer)
            row_data.update({"attack_id": answer})
            all_predictions.append(row_data)
            # Store the first element of context
            if count == 1:
                self.question_history.append(question)
            self.question_history.append(question)
            input_token_count = estimate_token_count(question)
            # print(f"Token count: {input_token_count}")
        return all_predictions
