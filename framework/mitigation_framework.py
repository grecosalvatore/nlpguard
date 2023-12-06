from explainer.explainer import IntegratedGradientsExplainer, ShapExplainer
import os
import time
import datetime


class MitigationFramework:
    """ """
    def __init__(self):
        self.use_case_name = None
        self.output_folder = None
        return


    def initialize_mitigation_framework(self, use_case_name=None, output_folder=None):
        self.output_folder, self.use_case_name = self._init_output_folders(use_case_name, output_folder)
        return

    def run_full_mitigation_pipeline(self, use_case_name=None, output_folder=None):
        return


    def run_identifier(self):
        return

    def run_moderator(self):
        return

    @staticmethod
    def _init_output_folders(use_case_name, output_folder):
        """ Initializes the output folders.
        Args:
            use_case_name (str):
            output_folder (str):
        """
        # Init output folder. If output folder is not specified, the dafault is "outputs".
        if output_folder is None:
            output_folder = ("outputs")

        # Create the output folder if it does not exist.
        if not os.path.exists(output_folder):
            print("Mitigation Framework: Warning, output folder does not exist.")
            os.mkdir(output_folder)
            print("Mitigation Framework: Output folder created.")

        # Init use case folder. If use_case_name is not specified, the dafault is the current timestamp.
        if use_case_name is None:
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
            use_case_name = timestamp
        else:
            if os.path.exists(os.path.join(output_folder, use_case_name)):
                print("Mitigation Framework: Warning, specified use_case_name already exists in output folder.")
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
                use_case_name = use_case_name + "_" + timestamp

        use_case_folder_path = os.path.join(output_folder, use_case_name)
        os.mkdir(use_case_folder_path)

        # Create the explainer output folders
        os.mkdir(os.path.join(use_case_folder_path, "explainer_outputs"))
        os.mkdir(os.path.join(use_case_folder_path, "explainer_outputs", "local_explanations"))
        os.mkdir(os.path.join(use_case_folder_path, "explainer_outputs", "global_explanations"))

        # Create the identifier output folders
        os.mkdir(os.path.join(use_case_folder_path, "identifier_outputs"))

        # Create the moderator output folders
        os.mkdir(os.path.join(use_case_folder_path, "moderator_outputs"))

        print(f"Mitigation Framework: Initialized output folder - {use_case_folder_path}.")
        return output_folder, use_case_name

    def run_explainer(self, model, tokenizer, texts, explainer_method="integrated-gradients", batch_size=128):
        """
        Args:
            model:
            tokenizer:
            texts (List[str]):
            explainer_method
        """

        if explainer_method == "integrated-gradients":
            print("Mitigation Framework: Instantiated Integrated Gradients Explainer")
            exp = IntegratedGradientsExplainer(model, tokenizer)
        elif explainer_method == "shap":
            print("Mitigation Framework: Instantiated SHAP Explainer")
            exp = ShapExplainer(model, tokenizer)
        else:
            print("Mitigation Framework: Unknown Explainer method. Please select ....")

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