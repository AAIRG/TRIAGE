import json
import dspy
import pandas as pd

class Job:
    def __init__(self, data_handler):
        self.cve_id_to_desc = data_handler.cve_id_to_desc
        self.affected_object_mapping = data_handler.affected_object_mapping
        self.df_cve_split = data_handler.cves_split
        self.cve_ids = list(self.df_cve_split["CVE ID"])
        self.dspy_module = "chain_of_thought"
        self.qa = self.get_dspy_module()
        self.question_history = []
        self.reasoning_history = []



    def get_dspy_module(self):
        if self.dspy_module == "chain_of_thought":
            qa = dspy.ChainOfThought('question -> answer')
        elif self.dspy_module == "predict":
            qa = dspy.Predict('question -> answer')
        else:
            raise NotImplementedError
        return qa

    def get_background(self):
        prompt = "An exploitation technique is the method (technique) used to exploit the vulnerability. The type of objects that is affected by an exploitation technique includes software, hardware, firmware, product, application, or code. "
        return prompt

    def get_goal(self):
        prompt = ("Your goal is to predict the affected object relevant to the exploitation of the vulnerability. "
                  "It should only be one affected object. In the remainder, you will first receive the vulnerability to label, then an overview of affected objects to use as labels. ")
        return prompt

    def get_cve_description(self, cve):
        # CVE ID to desc
        prompt = (
            f"The vulnerability to be labelled has the following description:\n"
            f"----\n"
            f"{self.cve_id_to_desc[cve]}\n"
            "----"
        )
        return prompt

    def get_affected_object_types(self):
        prompt = (f"\nThe JSON string below describes the affected objects to choose from:\n"
                 f"----")
        # Extract the desired keys into a new list of dictionaries
        selected_data = [
            {
                "affected_object": item.get("affected_object"),
                "description": item.get("description"),
                "examples": item.get("examples")
            }
            for item in self.affected_object_mapping.get("items", [])
        ]
        # Convert the new data to a JSON string
        selected_data_2 = json.dumps(selected_data, indent=2)
        output = f"{prompt}\n{selected_data_2}\n----\n"
        return output

    def specify_output(self):
        prompt = (f"Based on the given information, what is the affected object? Select one of the "
                  f"following:\n"
                  f"----")
        affected_objects = []
        for item in self.affected_object_mapping["items"]:
            affected_object = item.get("affected_object")
            affected_objects.append(affected_object)
        output = f"{prompt}\n{affected_objects}"
        return output

    def run_experiment(self):
        predictions = []
        count = 0
        for cve in self.cve_ids:
            print(f"Evaluating CVE: {cve}")
            background = self.get_background()
            goal = self.get_goal()
            cve_desc = self.get_cve_description(cve)
            affected_object_types = self.get_affected_object_types()
            specify_output = self.specify_output()
            question = f"{background}{goal}{cve_desc}{affected_object_types}{specify_output}"

            pred = self.qa(question=question)
            predictions.append(pred.answer)
            print(f"pred.answer: {pred.answer}")
            count += 1
            if count == 1:
                self.question_history.append(question)
            self.reasoning_history.append({"CVE ID": cve,
                                           "pred": pred.answer,
                                           "reasoning": pred.reasoning})
        data = {"CVE ID": self.cve_ids, "pred": predictions}
        df = pd.DataFrame(data)
        return df
