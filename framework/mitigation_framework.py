import os


class MitigationFramework:
    """ """
    def __init__(self, use_case_name, output_folder=None):
        # Init use case name
        self.use_case_name = use_case_name
        # Init output folder
        if output_folder is not None:
            self.output_folder = output_folder
        else:
            self.output_folder = os.path.join("..", "outputs")

        self._init_outputs_folders(use_case_name, output_folder)
        return

    def _init_outputs_folders(self, use_case_name, output_folder):

        return

    def run_explainer(self, explainer_method="integrated-gradients"):


        if explainer_method == "integrated-gradients":
            print("Integrated Gradients Explainer")
        elif explainer_method == "shap":
            print("SHAP Explainer")
        else:
            print("Unknown Explainer method")

        return

    def run_identifier(self, identifier_method="chatgpt"):

        if identifier_method == "chatgpt":
            print("ChatGPT Identifier")
        elif identifier_method == "mturk":
            print("MTurk Identifier")
        else:
            print("Unknown Identifier method")
        return

    def run_moderator(self):
        return