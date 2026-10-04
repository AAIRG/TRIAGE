import pandas as pd
import itertools
import time
import dspy

from lib.utils import map_intermediate_labels, \
    compute_and_log_metrics, remove_digits

"""
This module is responsible for the following methodology mappers:
- Vulnerability Type
- Functionality
- Tactic
- Exploitation Technique
"""

class VulnerabilityType:
    """
    All methods and specific parameters from vulnerability type method
    """

    def __init__(self, data_handler):
        self.cve_id_to_desc = data_handler.cve_id_to_desc
        self.df_cwe = data_handler.df_cwe
        self.vul_dict = data_handler.vul_dict
        self.question_history = []
        self.reasoning_history = []

    def predict_cwe(self, input_cves, qa, qa_2):
        """

        Function to predict CWEs from CVE descriptions. These CWEs are later converted to
        vulnerability types.

        Args:
            input_cves:
            qa:

        Returns:

        """

        multi_vul_type_instruction = (
            f"The following vulnerability types (keys) with corresponding "
            f"descriptions (values) exist: {self.vul_dict}")

        output = {}
        input_cves = input_cves['CVE ID']
        count = 0
        instruction_2 = None
        pred_2 = None
        for i, split_cve_id in input_cves.items():
            # print(f"Count: {count} of {len(input_cves)}")
            count += 1
            # print(f"test CVE ID: {split_cve_id}")
            split_cve_desc = self.cve_id_to_desc[split_cve_id]
            instruction = (f"{multi_vul_type_instruction}. Given a new CVE with description: "
                           f"{split_cve_desc}. Which vulnerability type does this CVE map to? "
                           f"provide only the vulnerability type. If no vulnerability type "
                           f"applies, answer with 'N/A'")
            pred = qa(question=instruction)
            # print(f"pred.answer: {pred.answer}")

            # NB: consider changing these to vulnerability types later
            context = instruction + pred.answer
            if pred.answer == ("Improper Neutralization of Input During Web Page Generation "
                               "('Cross-site Scripting')"):
                instruction_2 = ("Select whether the vulnerability is a stored or other "
                                 "type of Cross-site Scripting vulnerability. Answer with "
                                 "'STORED' or 'OTHER'")
                pred_2 = qa_2(question=instruction_2, context=context)
                output[split_cve_id] = [pred.answer, pred_2.answer]
            elif pred.answer == "Overly Restrictive Account Lockout Mechanism":
                instruction_2 = ("Select whether the vulnerability is related to mobile devices "
                                 "or not. Answer with 'MOBILE' or 'OTHER'")
                pred_2 = qa_2(question=instruction_2, context=context)
                output[split_cve_id] = [pred.answer, pred_2.answer]
            elif pred.answer == "Cryptographic Issues":
                instruction_2 = ("Select whether the vulnerability is related to "
                                 "OPTION 1: Credential storage or transmission"
                                 "OPTION 2: Transmitting over network"
                                 "OPTION 3: Sensitive information storage"
                                 "Answer with 'OPTION 1', 'OPTION 2' or "
                                 "'OPTION 3' depending on the relevant option. Only one of "
                                 "the options should be selected.")
                pred_2 = qa_2(question=instruction_2, context=context)
                output[split_cve_id] = [pred.answer, pred_2.answer]
            else:
                output[split_cve_id] = [pred.answer, None]
            # NB: The first question is really what is interesting.
            self.question_history.append(instruction)
            self.reasoning_history.append(split_cve_id)
            self.reasoning_history.append(pred.reasoning)
            if instruction_2 is not None:
                self.question_history.append(instruction_2)
                self.reasoning_history.append(pred_2.reasoning)

        return output


class Functionality:
    """
    All methods and specific parameters from functionality method
    """
    def __init__(self, data_handler):
        self.attack_techniques = data_handler.attack_techniques
        self.df_cve_base = data_handler.cve_base
        self.cve_id_to_desc = data_handler.cve_id_to_desc
        self.df_functionality_mapping = data_handler.df_functionality_mapping
        self.functionality_labels = data_handler.functionality_labels
        self.use_functionality_demos = data_handler.use_functionality_demos

        self.func_name_to_desc = data_handler.func_name_to_desc
        self.question_history = []
        self.reasoning_history = []

    def create_df_functionality_combinations(self, df):
        # Create dataset with negative examples
        cve_func_combinations = create_combinations(df, self.df_functionality_mapping)
        # Make dataframe
        df_func_combinations = pd.DataFrame(cve_func_combinations, columns=['CVE ID',
                                                                                'Functionality'])
        return df_func_combinations

    def get_demo_examples(self, col_name, col_value, max_examples=5, target=None):
        """
        NB: Currently tested on the Functionality class
        Get a demonstration example based on a column value. If target is 1, get a positive example,
        else (if 0) get a negative example. If random is True, get a random example, otherwise get
        the first.
        """
        # Create combinations of CVE and functionality among the functionality labels

        all_cves = set(self.functionality_labels["CVE ID"])
        pos_cves = self.functionality_labels.loc[self.functionality_labels[col_name] ==
                                                 col_value, "CVE ID"]
        pos_cves = set(pos_cves)
        neg_cves = all_cves - pos_cves
        select_cves = pos_cves if target == 1 else neg_cves
        select_cves = list(select_cves)
        if len(select_cves) > max_examples:
            select_cves = select_cves[:max_examples]
        # # Add CVE descriptions
        # Create a dataframe
        all_desc = []
        for cve in select_cves:
            cve_desc = self.cve_id_to_desc[cve]
            all_desc.append(cve_desc)
        df_5 = pd.DataFrame.from_dict({"CVE ID": select_cves, "description": all_desc})
        return df_5


    def run_functionality_method(self, cves, df_attack_labels, qa, neptune_run, compute_metrics,
                                                  logging_interval):
        """
        Run the functionality method
        """
        output = []
        # start clock
        start_time = time.time()
        df_func_combinations = self.create_df_functionality_combinations(cves)
        for i in range(len(df_func_combinations)):
            # print(f"Example {i} of in total {len(df_func_combinations)}")
            elapsed_time = time.time() - start_time
            # print(f"Elapsed time: {elapsed_time}")
            split_cve_id = df_func_combinations.loc[i, "CVE ID"]
            split_cve_desc = self.cve_id_to_desc[split_cve_id]
            func = df_func_combinations.loc[i, "Functionality"]
            desc_func = self.func_name_to_desc[func]
            base_func_instruction =  (
                "Determine whether or not an attacker gains access to "
                "the following "
                f"functionality: \n"
                "----\n"
                f"{func}: {desc_func}\n"
                "----\n"
                f"by exploiting this vulnerability:\n"
                f"---\n"
                f"{split_cve_desc}\n"
                "---\n"
            )
            # print(f"Functionality: {func}")
            if self.use_functionality_demos:
                try:
                    demo_cves = self.get_demo_examples(col_name="Functionality", col_value=func,
                                                       target=1)

                    instruction_component = [f"Examples of other CVEs that "
                                            f"give an "
                                   f"attacker access to this functionality include:", "----"]
                    for rows in demo_cves.itertuples():
                        instruction_component.append(f" - {rows.description}")
                    instruction_component.append("----")
                    instruction_component = "\n".join(instruction_component)
                    instruction_component = f"\n{instruction_component}\n"

                    negative_demos = self.get_demo_examples(col_name="Functionality",
                                                            col_value=func, target=0)
                    instruction_component_2 = [f"Examples of other CVEs that do not "
                                             f"give an "
                                             f"attacker access to this functionality include:",
                                             "----"]
                    for rows in negative_demos.itertuples():
                        instruction_component_2.append(f" - {rows.description}")
                    instruction_component_2.append("----")
                    instruction_component_2 = "\n".join(instruction_component_2)
                    instruction_component_2 = f"\n{instruction_component_2}\n"
                    instruction_component = (f"\n{instruction_component}\n"
                                             f"{instruction_component_2}\n")

                except IndexError:
                    instruction_component = ""
                    # print(f"No valid demo example for {func} functionality found")
                    # print("Prompting without example")
            else:
                instruction_component = ""
            specify_answer = "Answer with 'YES' or 'NO'."
            final_instruction = f"{base_func_instruction}{instruction_component}{specify_answer}"
            pred = qa(question=final_instruction)
            if pred.answer == 'YES':
                print(f"CVE ID: {split_cve_id}")
                print(f"Functionality: {func}")
            output.append(pred.answer)
            if compute_metrics:
                if i % logging_interval == 0:
                    df_pred_intermediate = create_df_pred_func(df_func_combinations, output)
                    df_pred_final = map_intermediate_labels(self.df_functionality_mapping,
                                                            df_pred_intermediate,
                                                            mapping_method="functionality")
                    compute_and_log_metrics(df_pred_final, df_attack_labels, self.attack_techniques,
                                            mapping_method="functionality", neptune_run=neptune_run)
            self.question_history.extend([str(split_cve_id), str(func), final_instruction])
            self.reasoning_history.extend([str(split_cve_id), str(func), pred.reasoning])
        df_pred_intermediate = create_df_pred_func(df_func_combinations, output)
        # compute final metrics
        return df_pred_intermediate


class Job:
    def __init__(self, args, neptune_run, data_handler):
        self.data_handler = data_handler
        self.vulnerability_mapping = self.data_handler.df_vulnerability_mapping
        self.df_functionality_mapping = self.data_handler.df_functionality_mapping
        self.qa = dspy.ChainOfThought("question -> answer")
        self.qa_2 = dspy.ChainOfThought('context, question -> answer')
        self.neptune_run = neptune_run
        self.mapping_method = data_handler.mapping_method
        self.compute_metrics = args.compute_metrics
        self.logging_interval = args.logging_interval
        self.cves_split = data_handler.cves_split
        self.df_attack_labels = data_handler.df_attack_labels
        self.attack_techniques = self.data_handler.attack_techniques
        self.df_vulnerability_mapping = self.data_handler.df_vulnerability_mapping
        self.include_intermediate_labels = args.include_intermediate_labels
        self.question_history = None
        self.reasoning_history = None
        self.df_pred_intermediate = None
        self.df_pred_final = None

    def run_experiment(self):
        if self.mapping_method == "vulnerability_type":
            df_mapping = self.df_vulnerability_mapping
            self.df_pred_intermediate = self.predict_vul_type(self.cves_split)

        elif self.mapping_method == "functionality":
            df_mapping = self.df_functionality_mapping
            self.df_pred_intermediate = self.predict_functionality(self.cves_split)

        else:
            RaiseNotImplementedError("Mapping method not implemented")
        self.df_pred_final = map_intermediate_labels(df_mapping, self.df_pred_intermediate,
                                                     self.mapping_method,
                                                     include_intermediate_labels=self.include_intermediate_labels)
        # Compute scores
        if self.compute_metrics:
            compute_and_log_metrics(self.df_pred_final, self.df_attack_labels,
                                    self.attack_techniques, mapping_method=self.mapping_method,
                                    neptune_run=self.neptune_run)


    def predict_vul_type(self, cves):
        """
        Predict CWEs and from these create a DataFrame with intermediate vul type predictions

        """
        df_cwe = self.data_handler.df_cwe
        df_vul_type_to_cwe = self.data_handler.df_vul_type_to_cwe
        vulnerability_type = VulnerabilityType(self.data_handler)
        output_intermediate = vulnerability_type.predict_cwe(cves, self.qa, self.qa_2)
        # Store prompt
        self.question_history = vulnerability_type.question_history
        self.reasoning_history = vulnerability_type.reasoning_history
        df_output = pd.DataFrame.from_dict(output_intermediate, orient='index',
                                           columns=['CWE Name', 'sublabel']).reset_index()
        df_output.rename(columns={'index': 'CVE ID'}, inplace=True)
        df_output = df_output.merge(df_cwe, how="left", on=["CWE Name"])
        df_output = df_output.merge(df_vul_type_to_cwe, how="left", on=["CWE ID"])
        df_pred = pd.DataFrame({'CVE ID': df_output['CVE ID'],
                                'pred_vul_type': df_output["Vulnerability Type"].to_list(),
                                'sublabel': df_output['sublabel']})

        return df_pred

    def predict_functionality(self, cves):
        functionality = Functionality(self.data_handler)
        df_pred = functionality.run_functionality_method(cves, self.df_attack_labels,
                                                         self.qa, self.neptune_run,
                                                         self.compute_metrics,
                                                         self.logging_interval)
        self.question_history = functionality.question_history
        self.reasoning_history = functionality.reasoning_history
        return df_pred


def create_df_pred_func(input_df, output):
    """
    Create a dataframe of intermediate functionality predictions from output.

    CVEs with no positive prediction will have an entry with N/A as functionality to ensure that
    all CVEs are included in the returned dataframe.
    Args:
        input_df: CVE and functionalities permutations
        output: Response from the LLM

    Returns:

    """
    output_bin = [1 if line.strip() == 'YES' else 0 for line in output]
    if len(output_bin) != len(input_df):
        input_df = input_df.iloc[:len(output_bin), :]
        input_df.reset_index(drop=True, inplace=True)
    df_pred_intermediate = pd.DataFrame({
        'CVE ID': input_df['CVE ID'],
        'Functionality': input_df['Functionality'],
        'pred': output_bin
    })
    # Handle CVEs without a positive functionality. We need to include these CVEs for correct
    #   evaluation later on
    df_pred_pos = df_pred_intermediate.loc[df_pred_intermediate['pred'] != 0]
    cves_pos = set(df_pred_pos["CVE ID"])
    df_pred_intermediate_neg = df_pred_intermediate.loc[~df_pred_intermediate['CVE ID'].isin(cves_pos)]
    # We add N/A as functionality to CVEs with only negatives
    df_pred_intermediate_neg.loc[:, 'Functionality'] = 'N/A'
    df_pred_intermediate_neg = df_pred_intermediate_neg.drop_duplicates()
    df_pred_intermediate_neg.loc[:, 'pred'] = 0
    df_pred_intermediate_pos = df_pred_intermediate.loc[df_pred_intermediate["pred"] == 1]
    # Create a new dataframe where negative predictions are kept only for CVEs when no positive
    #   exists
    df_pred_intermediate_2 = pd.concat([df_pred_intermediate_pos, df_pred_intermediate_neg])

    df_pred_intermediate_2 = df_pred_intermediate_2.drop(columns=['pred'])
    df_pred_intermediate_2.reset_index(drop=True, inplace=True)
    return df_pred_intermediate_2


def create_combinations(df_func_labels, df_func_mapping):
    """
    Create all possible combinations of CVE and functionality
    """
    # Get all CVEs
    cves = df_func_labels['CVE ID'].unique()
    # Get all functionalities
    functionalities = df_func_mapping['Functionality'].unique()
    # Create all combinations
    all_combinations = list(itertools.product(cves, functionalities))
    return all_combinations


def get_tactic_prompt(tactic_descriptions, cve_desc, tactic=None, binary_classification=False):
    if binary_classification:
        tactic_prompt_header = (f"CVE is a dictionary of common names for publicly known"
                                f"cybersecurity vulnerabilities. A tactic is the "
                                f"adversary's tactical goal: the reason for performing an "
                                f"action. For example, an adversary may want to achieve credential access."
                                f"The task is to map from CVE to tactic"
                                f" Does the following CVE with description: {cve_desc}"
                                f" map to the tactic: '{tactic}', with description: "
                                f"'{tactic_descriptions[tactic]}'? Please answer with YES or NO.")
    else:
        tactic_prompt_header = (f"CVE is a dictionary of common names for publicly known"
                                f"cybersecurity vulnerabilities. A tactic is the "
                                f"adversary's tactical goal: the reason for performing an "
                                f"action. For example, an adversary may want to achieve credential access."
                                f"The task is to map from CVE to tactic"
                                f" Please map a CVE with the following description: {cve_desc}"
                                f" To a tactic among the following tactics with corresponding "
                                f"descriptions: {tactic_descriptions}.")
    return tactic_prompt_header


def retrieve_exp_mapping(exp_data, q_short, option=None, use_parent_techniques=True,
                         return_all=False):
    """
    Retrieve mapped attack techniques from exploit mapping. For questions without a subquestion,
    the technique is found based on the shortform of the "parent" question. For questions with
    subquestions, an answer option is required.

    Args:
        exp_data (dict): The exploit mapping data containing questions and their mappings.
        q_short (str): The short form identifier for the target question.
        option (str, optional): The answer option for subquestions, if applicable. Defaults to None.
        use_parent_techniques (bool, optional): Whether to map techniques to their parent forms by
          removing sub-technique digits. Defaults to True.
        return_all (bool, optional): If True, returns a tuple
        (techniques, question_full, sub_question); otherwise, returns only techniques.

    Returns:
        list or tuple: If return_all is False, returns a list of mapped techniques (or None if not found).
                       If return_all is True, returns a tuple (techniques, question_full, sub_question).
    """
    techniques = None
    question_full = None
    sub_question = None
    for question in exp_data['questions']:
        if question['question_short'] == q_short:
            question_full = question['question']
            try:
                techniques = question['techniques']
            except KeyError:  # When techniques are not found at the top level, check answers to
                # subquestion.
                sub_question = question['sub_question']
                for answer in question['answers']:
                    if answer['option'] == option:
                        techniques = answer['techniques']
    # If use parent techniques, remove sub-technique digits
    if use_parent_techniques:
        if techniques is not None:
            techniques = [remove_digits(technique) for technique in techniques]
    if return_all:
        return techniques, question_full, sub_question
    else:
        return techniques


def get_instruction_exp_root(cve_desc, question):
    """
    Generate an instruction prompt for the root exploitation question based on a CVE description.

    Args:
        cve_desc (str): The description of the vulnerability (CVE).
        question (dict): A dictionary containing the root question with a "question" key.

    Returns:
        str: An instruction string prompting the user to answer YES or NO to the given question.
    """
    exp_root_header =  (f"We ask a series of questions to determine what steps are necessary to "
                        f"exploit a vulnerability with the following description: {cve_desc}"
                        f" Considering the given vulnerability description, please answer with YES or "
                        f"NO to the following:")
    instruction = exp_root_header + " " +  question["question"]
    return instruction


def get_instruction_sub(question):
    """
    Generate an instruction prompt for a sub-question in the exploitation technique method, listing all
    possible answer options.

    Args:
        question (dict): A dictionary containing a "sub_question" key and an "answers" list,
                         where each answer has an "option" key.

    Returns:
        str: An instruction string prompting the user to answer with one of the provided alternatives.
    """
    all_options = [answer["option"] for answer in question["answers"]]
    instruction = f"{question['sub_question']} Answer with one of these alternatives: {all_options}"
    return instruction


def get_sub_qs_w_ans(df, sub_q=None):
    """
    Get sub-questions of exploitation technique method with their answers from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing sub-questions.
        sub_q (list, optional): List of sub-question columns to check.

    Returns:
        dict: A dictionary mapping sub-question columns to their answers.
    """
    if sub_q is None:
        sub_q = ["File come from?", "Link come from?"]
    sub_q_dict = {}
    for col in sub_q:
        if col in df.columns:
            sub_q_dict[col] = df[col].values[0]
    return sub_q_dict
