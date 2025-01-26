from ferret import Benchmark
from ferret import IntegratedGradientExplainer
from tqdm import tqdm
from abc import ABC, abstractmethod
import pandas as pd
import os
import torch


class Explainer(ABC):
    """ Abstract Explainer Class. """
    def __init__(self):
        return

    @abstractmethod
    def local_explanations(self, **kwargs):
        """ Abstract method to compute local explanations. """
        return

    def global_explanations(self, label_ids_to_explain, id2label, explainer_output_folder):
        """ Computes global explanations for a set of class labels.

        Args:
            label_ids_to_explain (list): List of class label ids to explain.
            id2label (dict): Dictionary mapping class label ids to class names.
            explainer_output_folder (str): Path to the folder containing the local explanations.

        Returns:
            dict: Dictionary containing the global explanations for each class label.
        """

        output_dict = {}

        for label_id in label_ids_to_explain:
            label_name = id2label[label_id]

            # Check if the directory contains at least one CSV file
            csv_files = [file for file in os.listdir(os.path.join(explainer_output_folder, "local_explanations", label_name)) if file.endswith(".csv")]
            if not csv_files:
                print(f"\nGlobal Explanations for label: Local explanations not found for label {label_name}. Skipping...")
                output_dict[label_name] = []
            else:
                df_current = self._load_local_explanations(os.path.join(explainer_output_folder, "local_explanations", label_name))
                df_current_global = self._compute_global_scores(df_current)
                df_current_global.to_csv(os.path.join(explainer_output_folder, "global_explanations", f"global_scores_{label_name}.csv"))
                print(f"\nGlobal Explanations for label: {label_name}")
                print(df_current_global)
                print("\n")

                output_dict[label_name] = df_current_global["tokens"].tolist()

        return output_dict

    @staticmethod
    def _load_local_explanations(dir_path) -> pd.DataFrame:
        """ Loads local explanations from a directory.

        Args:
            dir_path :obj:`str`: Path to the directory containing the local explanations.

        Returns:
            :obj:`pandas.core.frame.DataFrame`: Dataframe containing the local explanations.
        """

        df_list = []

        for path in os.listdir(dir_path):
            current_path = os.path.join(dir_path, path)
            if current_path.endswith(".csv"):
                df_tmp = pd.read_csv(current_path)
                df_list.append(df_tmp)

        if not df_list:
            raise ValueError(f"No CSV files found in directory: {dir_path}")

        if len(df_list) == 1:
            df = df_list[0]
        else:
            df = pd.concat(df_list, ignore_index=True)

        df["tokens"] = df["tokens"].astype(str)
        df["tokens"] = df["tokens"].apply(lambda x: x.replace(" ", ""))
        df["tokens"] = df["tokens"].apply(lambda x: x.lower())

        return df

    @staticmethod
    def _compute_global_scores(df, minimum_frequency=2, subtoken_separator="##") -> pd.DataFrame:
        """ Computes overall importance scores (global explanations) for each token by aggregating their importance withing individual predictions (local explanations).

        Args:
            df (pd.DataFrame): Dataframe containing local explanations.
            minimum_frequency (int, optional): Minimum frequency of a token to be considered in the global explanations. Defaults to 2.
            subtoken_separator (str, optional): Subtoken separator used by the tokenizer. Defaults to "##".

        Returns:
            pd.DataFrame: Dataframe containing the global explanations.
        """

        df_grp = df.groupby(["tokens"]).sum().reset_index()

        df_freq = df.groupby(["tokens"]).size().reset_index(name='freq')

        neg_freq_dict = {}

        for token, freq in zip(df_freq.tokens.tolist(), df_freq.freq.tolist()):
            neg_freq_dict[token] = freq

        df_global = df_grp.copy()

        df_global = df_global.join(df_freq.set_index('tokens'), on='tokens')

        df_global["score_norm"] = df_global["explanation_scores"] / df_global["freq"]
        df_global["score_norm_weighted"] = df_global["explanation_scores_weighted"] / df_global["freq"]

        df_global = df_global.drop(columns=["Unnamed: 0"])

        df_global = df_global.sort_values(by='score_norm', ascending=False)

        df_global = df_global.loc[df_global.freq >= minimum_frequency]

        df_global = df_global.loc[~df_global.tokens.str.startswith(subtoken_separator)]

        df_global = df_global.loc[df_global.tokens.str.len() > 2]

        df_global = df_global.loc[~df_global.tokens.str.isdigit()]

        df_global = df_global.loc[df_global.score_norm > 0]

        return df_global


class IntegratedGradientsExplainer(Explainer):
    """ Integrated Gradients implementation of the Explainer abstract class. """

    def __init__(self, model, tokenizer, device="cpu"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model.to(device)

        # Create IG explainer Ferret library
        self.igx = IntegratedGradientExplainer(model, tokenizer)
        self.bench = Benchmark(model, tokenizer, explainers=[self.igx])
        return

    @staticmethod
    def batch_iterator(list1, list2, batch_size):
        """ Iterates two list by batch_size.

        Args:
            list1 (list): First list.
            list2 (list): Second list.
            batch_size (int): Batch size.
        Yields:
            tuple: Tuple containing the current batch of list1, list2, start_index and end_index.
        """

        # Ensure both input lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Input lists must have the same length")

        # Iterate over the lists by batch_size
        for i in range(0, len(list1), batch_size):
            start_index = i
            end_index = min(i + batch_size, len(list1))
            # Return the current batch
            yield list1[start_index:end_index], list2[start_index:end_index], start_index, end_index

    def local_explanations(self, df_predictions, local_explanations_folder, label_ids_to_explain, id2label, batch_size=128):
        """ Computes and saves local explanations for a set of predictions in the output folder.

        Args:
            df_predictions (pd.DataFrame): Dataframe containing the predictions to explain.
            local_explanations_folder (str): Path to the folder where the local explanations will be saved.
            label_ids_to_explain (list): List of label ids to explain.
            id2label (dict): Dictionary mapping label ids to label names.
            batch_size (int, optional): Batch size. Defaults to 128.

        """

        log_errors = []
        log_filepath = os.path.join(local_explanations_folder, "log_file.txt")

        start_index = 0

        label_ids_to_explain
        # Iterate over labels to explain
        for label_id in label_ids_to_explain:
            label_name = id2label[label_id]

            # Create folder for current label
            current_path = os.path.join(local_explanations_folder, label_name)
            if not os.path.exists(current_path):
                os.mkdir(current_path)

            # Select texts predicted with the current label to explain
            df_predictions_current_label = df_predictions.loc[df_predictions.pred_label_name == label_name]

            print(f"Explainer: Explaining label: {label_name} - Number of texts: {len(df_predictions_current_label)}")

            # Iterate over batches of texts
            for batch_texts, batch_predicted_prob_explained_class, start_index, end_index in self.batch_iterator(df_predictions_current_label["text"].tolist(), df_predictions_current_label["pred_score"].tolist(), batch_size):
                try:
                    print(f"\t Explaining Batch: {start_index} -> {end_index}")

                    # Compute local explanation on current batch
                    batch_tokens, batch_scores, batch_scores_weighted = self._compute_batch_local_explanations(label_id, batch_texts, batch_predicted_prob_explained_class)

                    # Save local explanations of current batch on a csv file
                    df = pd.DataFrame({"tokens": batch_tokens, "explanation_scores": batch_scores, "explanation_scores_weighted": batch_scores_weighted})
                    df.to_csv(os.path.join(current_path, f"scores_{start_index}_{end_index}.csv"))

                except Exception as e:
                    print(f"An exception occurred in {start_index} -> {end_index}")
                    print(e)
                    log_errors.append(f"An exception occurred in {start_index} -> {end_index} in label {label_name}")
                    with open(log_filepath, "a") as file_object:
                        file_object.write(f"\nAn exception occurred in {start_index} -> {end_index} in label {label_name}")

        return

    def _compute_batch_local_explanations(self, label_id, batch_texts, batch_pred_scores) -> tuple[list[str], list[float], list[float]]:
        """ Computes local explanations for a batch of texts.

        Args:
            label_id (:obj:`int`): Label id to explain.
            batch_texts (:obj:`list[str]`): List of texts to explain.
            batch_pred_scores (:obj:`list[float]`): List of predicted probabilities for the target label.

        Returns:
            :obj:`tuple(list(str), list(float), list(float))`: Tuple containing the tokens, feature importance scores, and weighted feature importance scores.
        """

        batch_tokens = []
        batch_scores = []
        batch_scores_weighted = []

        for text, prob in tqdm(zip(batch_texts, batch_pred_scores)):
            text = str(text)

            #tokenized_text = self.tokenizer.tokenize(text)
            #if len(tokenized) > 320:
            #    text = self.tokenizer.convert_tokens_to_string(tokenized[:320])

            explanations = self.bench.explain(text, target=label_id, show_progress=False, normalize_scores=True)

            # Remove [CLS] and [SEP] tokens
            tokens = explanations[0].tokens[1:-1]
            scores_raw = explanations[0].scores[1:-1].tolist()

            # Produce importance score weighted by predicted probability
            scores_weighted = [s * prob for s in scores_raw]

            # Store tokens, feature importance, and weighted feature importance
            batch_tokens.extend(tokens)
            batch_scores.extend(scores_raw)
            batch_scores_weighted.extend(scores_weighted)
        return batch_tokens, batch_scores, batch_scores_weighted


class ShapExplainer(Explainer):
    """ Shap implementation of the Explainer abstract class. """

    def local_explanations(self):
        return

    def global_explanations(self):
        return

