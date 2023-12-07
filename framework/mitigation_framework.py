from framework.explainer.explainer import IntegratedGradientsExplainer, ShapExplainer
from framework.identifier.identifier import ChatGPTIdentifier
from framework.moderator.moderator import PandasDataFrameModerator
from transformers.pipelines import pipeline
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd

class MitigationFramework:
    """ """
    def __init__(self):
        self.use_case_name = None
        self.output_folder = None
        self.explainer_output_folder = None
        self.identifier_output_folder = None
        self.moderator_output_folder = None
        self.id2label = None
        self.label2id = None
        return


    def initialize_mitigation_framework(self, id2label, use_case_name=None, output_folder=None):
        """ Initializes the output folders.
            Args:
                use_case_name (str):
                output_folder (str):
                labels2id (dict):
        """
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.output_folder, self.use_case_name, self.explainer_output_folder, self.identifier_output_folder, self.moderator_output_folder = self._init_output_folders(use_case_name, output_folder)
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
        explainer_output_folder = os.path.join(use_case_folder_path, "explainer_outputs")
        os.mkdir(explainer_output_folder)
        os.mkdir(os.path.join(explainer_output_folder, "local_explanations"))
        os.mkdir(os.path.join(explainer_output_folder, "global_explanations"))

        # Create the identifier output folders
        identifier_output_folder = os.path.join(use_case_folder_path, "identifier_outputs")
        os.mkdir(identifier_output_folder)

        # Create the moderator output folders
        moderator_output_folder = os.path.join(use_case_folder_path, "moderator_outputs")
        os.mkdir(moderator_output_folder)

        print(f"Mitigation Framework: Initialized output folder - {use_case_folder_path}.")
        return output_folder, use_case_name, explainer_output_folder, identifier_output_folder, moderator_output_folder

    def run_full_mitigation_pipeline(self, use_case_name=None, output_folder=None):
        return

    def run_explainer(self, model, tokenizer, texts, label_ids_to_explain,  explainer_method="integrated-gradients", batch_size=128, device="cpu"):
        """
        Args:
            model (:obj:`transformers.AutoModelForSequenceClassification`):
            tokenizer (:obj:`transformers.AutoTokenizer`):
            texts (:obj:List[str]):
            label_ids_to_explain (List[int]):
            explainer_method (str):
            device (str, optional):
        """

        if explainer_method == "integrated-gradients":
            print("Mitigation Framework: Instantiated Integrated Gradients Explainer")
            explainer = IntegratedGradientsExplainer(model, tokenizer, device)
        elif explainer_method == "shap":
            print("Mitigation Framework: Instantiated SHAP Explainer")
            explainer = ShapExplainer(model, tokenizer, device)
        else:
            print("Mitigation Framework: Unknown Explainer method. Please select ....")

        # Predict labels from texts
        print("Mitigation Framework: Running Predictions")
        df_predictions = self.run_predictions(model, tokenizer, texts, batch_size, device)

        print("Mitigation Framework: Running Local Explanations")
        local_explanations_folder = os.path.join(self.explainer_output_folder, "local_explanations")
        explainer.local_explanations(df_predictions, local_explanations_folder, label_ids_to_explain, self.id2label, batch_size)

        print("Mitigation Framework: Running Global Explanations")
        output_dict = explainer.global_explanations(label_ids_to_explain, self.id2label, self.explainer_output_folder)

        return output_dict

    def run_identifier(self, output_dict, identifier_method="chatgpt"):
        """ Runs the Identifier Component to determine which of the most important words are protected attributes.
        Args:
            output_dict (:obj:dict):
            identifier_method (str, optinal):
        """
        if identifier_method == "chatgpt":
            # Extracting distinct words from most important words for each label
            distinct_words = list(set(word for words_list in output_dict.values() for word in words_list))

            print("Mitigation Framework: Running ChatGPT Identification of Protected Attributes")
            # Instantiating the ChatGPT implementation of the Identifier
            identifier = ChatGPTIdentifier()

            # Annotating the protected attributes with ChatGPT
            identifier.annotate_protected_attributes(distinct_words)
        elif identifier_method == "mturk":
            print("MTurk Identifier")
        else:
            print("Unknown Identifier method")
        return

    def run_moderator(self, df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, mitigate_each_label_separately=False, batch_size=128):
        """ Runs the Moderator Component to produce a new mitigated training dataset based on the identified protected attributes.
        Args:
            df_train (:obj:`pandas.DataFrame`):
            tokenizer (:obj:`transformers.AutoTokenizer`):
            protected_attributes_per_label_dict (:obj:dict):
            text_column_name (str):
            label_column_name (str):
            mitigate_each_label_separately (bool, optional):
            batch_size (int, optional):
        """
        moderator = PandasDataFrameModerator()
        df_train_mitigated = moderator.words_removal_mitigation_strategy(df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, mitigate_each_label_separately, batch_size)
        return df_train_mitigated

    def run_predictions(self, model, tokenizer, texts, batch_size, device="cpu"):
        """ Runs predictions on the corpus of texts on which compute the explanations.
        Args:
            model (:obj:`transformers.AutoModelForSequenceClassification`):
            tokenizer (:obj:`transformers.AutoTokenizer`):
            texts (:obj:List[str]):
            batch_size (int):
            device (str, optional):
        """
        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 128}
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

        preds = []
        iterations = len(texts) // batch_size
        remainder = len(texts) % batch_size
        for i in tqdm(range(iterations)):
            batch_preds = pipe(texts[i:i + batch_size], **tokenizer_kwargs)
            preds.extend(batch_preds)

        batch_preds = pipe(texts[-remainder:], **tokenizer_kwargs)
        preds.extend(batch_preds)


        pred_label_names = [d["label"] for d in preds]
        pred_scores = [d["score"] for d in preds]

        pred_dict = {}
        pred_dict["text"] = texts
        pred_dict["pred_label_name"] = pred_label_names
        pred_dict["pred_label_id"] = [self.label2id[l] for l in pred_label_names]
        pred_dict["pred_score"] = pred_scores
        pred_dict["pred_probabilities"] = preds

        df = pd.DataFrame(pred_dict)
        df.to_csv(os.path.join(self.output_folder, self.use_case_name, "predictions.csv"))
        return df