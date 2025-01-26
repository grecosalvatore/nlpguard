from nlpguard.explainer.explainer import IntegratedGradientsExplainer, ShapExplainer
from nlpguard.identifier.identifier import ChatGPTIdentifier, LLamaIdentifier
from nlpguard.moderator.moderator import PandasDataFrameModerator
from transformers.pipelines import pipeline
import os
import time
import datetime
from tqdm import tqdm
import pandas as pd
import torch

class NLPGuard:
    """ NLPGuard Framework class. It is used to run the mitigation framework.
    It is composed of three main components: Explainer, Identifier and Moderator.
    The Explainer is used to extract the most important words from a corpus of texts.
    The Identifier is used to identify the protected attributes from the most important words extracted by the explainer.
    The Moderator is used to mitigate the protected attributes identified by the identifier in a corpus of texts.

    Attributes:
        use_case_name (:obj:`str`): Name of the use case. It is used to create a new folder in the output folder.
        output_folder (:obj:`str`): Path to the output folder.
        explainer_output_folder (:obj:`str`): Path to the explainer output folder.
        identifier_output_folder (:obj:`str`): Path to the identifier output folder.
        moderator_output_folder (:obj:`str`): Path to the moderator output folder.
        id2label (:obj:`dict`): Dictionary mapping label ids to label names.
        label2id (:obj:`dict`): Dictionary mapping label names to label ids.

    """
    def __init__(self):
        self.use_case_name = None
        self.output_folder = None
        self.explainer_output_folder = None
        self.identifier_output_folder = None
        self.moderator_output_folder = None
        self.id2label = None
        self.label2id = None
        return

    def initialize_mitigation_framework(self, id2label, use_case_name=None, output_folder=None) -> None:
        """ Initializes the output folders.

            Args:
                use_case_name (:obj:`str`): Name of the use case. It is used to create a new folder in the output folder.
                output_folder (:obj:`str`): Path to the output folder.
                label2id (:obj:`dict`): Dictionary mapping label names to label ids.

            Returns:
                None
        """
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.output_folder, self.use_case_name, self.explainer_output_folder, self.identifier_output_folder, self.moderator_output_folder = self._init_output_folders(use_case_name, output_folder)
        return

    @staticmethod
    def _init_output_folders(use_case_name, output_folder) -> tuple:
        """ Initializes the output folders.

        Args:
            use_case_name (:obj:`str`): Name of the use case. It is used to create a new folder in the output folder.
            output_folder (:obj:`str`): Path to the output folder.

        Returns:
            :obj:`tuple(str)`: Tuple containing the output folder, the use case folder, the explainer output folder, the identifier output folder and the moderator output folder.
        """
        # Init output folder. If output folder is not specified, the dafault is "outputs".
        if output_folder is None:
            output_folder = ("outputs")

        # Create the output folder if it does not exist.
        if not os.path.exists(output_folder):
            print("NLPGuard: Warning, output folder does not exist.")
            os.mkdir(output_folder)
            print("NLPGuard: Output folder created.")

        # Init use case folder. If use_case_name is not specified, the dafault is the current timestamp.
        if use_case_name is None:
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
            use_case_name = timestamp
        else:
            if os.path.exists(os.path.join(output_folder, use_case_name)):
                print("NLPGuard: Warning, specified use_case_name already exists in output folder.")
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

        print(f"NLPGuard: Initialized output folder - {use_case_folder_path}.")
        return output_folder, use_case_name, explainer_output_folder, identifier_output_folder, moderator_output_folder

    def run_full_mitigation_pipeline(self, use_case_name=None, output_folder=None):
        # TODO: implement full mitigation pipeline in one function (explainer + identifier + moderator)
        return

    def run_explainer(self, model, tokenizer, texts, label_ids_to_explain,  explainer_method="integrated-gradients", batch_size=128, device="cpu") -> dict:
        """ Runs the explainer on the texts.

        Args:
            model (:obj:`transformers.AutoModelForSequenceClassification`): The model to explain.
            tokenizer (:obj:`transformers.AutoTokenizer`): The tokenizer to use to preprocess the inputs.
            texts (:obj:`list[str]`): The list of texts to explain.
            label_ids_to_explain (:obj:`list(str)`): The list of label ids to explain.
            explainer_method (:obj:`str`): The explainer method to use. The default is Integrated Gradients.
            device (:obj:`str`, `optional`): The device to run the explainer on. The default is "cpu".

        Returns:
            :obj:`dict`: Dictionary containing the local explanations for each text.
        """

        if explainer_method == "integrated-gradients":
            print("NLPGuard (Explainer): Instantiated Integrated Gradients Explainer")
            explainer = IntegratedGradientsExplainer(model, tokenizer, device)
        elif explainer_method == "shap":
            print("NLPGuard (Explainer): Instantiated SHAP Explainer")
            explainer = ShapExplainer(model, tokenizer, device)
        else:
            print("NLPGuard (Explainer): Unknown Explainer method. Please select ....")

        # Predict labels from texts
        print("NLPGuard (Explainer): Running Predictions")
        df_predictions = self.run_predictions(model, tokenizer, texts, batch_size, device)

        print("NLPGuard (Explainer): Running Local Explanations")
        local_explanations_folder = os.path.join(self.explainer_output_folder, "local_explanations")
        explainer.local_explanations(df_predictions, local_explanations_folder, label_ids_to_explain, self.id2label, batch_size)

        print("NLPGuard (Explainer): Running Global Explanations")
        output_dict = explainer.global_explanations(label_ids_to_explain, self.id2label, self.explainer_output_folder)

        return output_dict

    def run_identifier(self, output_dict, identifier_method="chatgpt", number_most_important_words=400,
                       hf_token="", hf_endpoint="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
        """ Runs the Identifier Component to determine which of the most important words are protected attributes.

        Args:
            output_dict (:obj:dict): Dictionary containing the most important words for each label.
            identifier_method (str, optional): Method to use for identifying protected attributes. Defaults to "chatgpt".
            number_most_important_words (int, optional): Number of most important words to use for identifying protected attributes. Defaults to 400.
        """
        if identifier_method == "chatgpt":
            # Extracting distinct words from most important words for each label
            distinct_words = list(set(word for words_list in output_dict.values() for word in words_list[:number_most_important_words]))

            print("NLPGuard (Identifier): Running ChatGPT Identification of Protected Attributes")
            # Instantiating the ChatGPT implementation of the Identifier
            identifier = ChatGPTIdentifier()

            # Annotating the protected attributes with ChatGPT
            df_annotated, protected_attributes = identifier.annotate_protected_attributes(distinct_words)
        elif identifier_method == "llama":
            # Extracting distinct words from most important words for each label
            distinct_words = list(set(word for words_list in output_dict.values() for word in words_list[:number_most_important_words]))

            print("NLPGuard (Identifier): Running LLaMa Identification of Protected Attributes")
            # Instantiating the ChatGPT implementation of the Identifier
            identifier = LLamaIdentifier(hf_endpoint=hf_endpoint, hf_token=hf_token)

            # Annotating the protected attributes with ChatGPT
            df_annotated, protected_attributes = identifier.annotate_protected_attributes(distinct_words)
        elif identifier_method == "mturk":
            print("NLPGuard (Identifier): Running MTurk Identification of Protected Attributes")
        else:
            print("Unknown Identifier method")

        protected_attributes_dict = {}
        # Printing the protected attributes for each label
        print("NLPGuard (Identifier): Protected Attributes per Label:")
        for output_dict_key in output_dict.keys():
            protected_attributes_current_label = [word for word in output_dict[output_dict_key] if word in protected_attributes]
            print(f"Label {output_dict_key} - Protected Attributes: {protected_attributes_current_label}")
            protected_attributes_dict[self.label2id[output_dict_key]] = protected_attributes_current_label
        return df_annotated, protected_attributes, protected_attributes_dict

    def run_moderator(self, df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, mitigation_strategy="word_removal", mitigate_each_label_separately=False, batch_size=128, n_synonyms=5,
                                                                                 keep_original_sentence=True,):
        """ Runs the Moderator Component to produce a new mitigated training dataset based on the identified protected attributes.
        Args:
            df_train (:obj:`pandas.DataFrame`): the training dataset to mitigate.
            tokenizer (:obj:`transformers.AutoTokenizer`): the tokenizer to use for tokenizing the text.
            protected_attributes_per_label_dict (:obj:dict): a dictionary containing the protected attributes for each label.
            text_column_name (str): the name of the column containing the text.
            label_column_name (str): the name of the column containing the labels.
            mitigation_strategy (str, optional): the mitigation strategy to use. Defaults to "word_removal".
            mitigate_each_label_separately (bool, optional): whether to mitigate each label separately.
            batch_size (int, optional): the batch size to use for the mitigation strategy. Defaults to 128.
        """
        moderator = PandasDataFrameModerator()

        if mitigation_strategy == "word_removal":
            print("NLPGuard (Moderator): Running Word Removal Mitigation Strategy")
            df_train_mitigated = moderator.words_removal_mitigation_strategy(df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, self.id2label, mitigate_each_label_separately, batch_size)
        elif mitigation_strategy == "sentence_removal":
            print("NLPGuard (Moderator): Running Sentence Removal Mitigation Strategy")
            df_train_mitigated = moderator.sentences_removal_mitigation_strategy(df_train,
                                                                                 tokenizer,
                                                                                 protected_attributes_per_label_dict,
                                                                                 text_column_name,
                                                                                 label_column_name,
                                                                                 self.id2label,
                                                                                 mitigate_each_label_separately, batch_size)

        elif mitigation_strategy == "word_replacement_with_synonym":
            print("NLPGuard (Moderator): Running Word Replacement with Synonyms Mitigation Strategy")
            df_train_mitigated = moderator.word_replacement_with_synonyms_mitigation_strategy(df_train,
                                                                                 tokenizer,
                                                                                 protected_attributes_per_label_dict,
                                                                                 text_column_name,
                                                                                 label_column_name,
                                                                                 self.id2label,
                                                                                 n_synonyms,
                                                                                 keep_original_sentence,
                                                                                 mitigate_each_label_separately, batch_size)
        elif mitigation_strategy == "word_replacement_with_hypernym":
            print("NLPGuard (Moderator): Running Word Replacement with Hypernyms Mitigation Strategy (TODO)")
        else:
            print("NLPGuard (Moderator): Unknown Mitigation Strategy. Please select ....")
        return df_train_mitigated

    def run_predictions(self, model, tokenizer, texts, batch_size, device="cpu"):
        """ Runs predictions on the corpus of texts on which compute the explanations.
        Args:
            model (:obj:`transformers.AutoModelForSequenceClassification`): the model to use for predictions.
            tokenizer (:obj:`transformers.AutoTokenizer`): the tokenizer to use for tokenizing the text.
            texts (:obj:List[str]): the list of texts to predict.
            batch_size (int): the batch size to use for predictions.
            device (str, optional): the device to use for predictions. Defaults to "cpu".
        """
        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 128}
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

        # TODO - Improve this code using yeld
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

        pred_dict = {"text": texts, "pred_label_name": pred_label_names,
                     "pred_label_id": [self.label2id[l] for l in pred_label_names], "pred_score": pred_scores,
                     "pred_probabilities": preds}

        df = pd.DataFrame(pred_dict)
        df.to_csv(os.path.join(self.output_folder, self.use_case_name, "predictions.csv"))
        return df