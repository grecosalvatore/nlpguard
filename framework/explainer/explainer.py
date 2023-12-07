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
        return


    def global_explanations(self, label_ids_to_explain, id2label, explainer_output_folder):

        for label_id in label_ids_to_explain:
            label_name = id2label[label_id]
            df_current = self._load_local_explanations(os.path.join(explainer_output_folder, "local_explanations", label_name))
            df_current_global = self._compute_global_scores(df_current)
            df_current_global.to_csv(os.path.join(explainer_output_folder, "global_explanations", f"global_explanations_{label_name}.csv"))
            print(df_current_global)
            print("\n\n")


        return

    @staticmethod
    def _load_local_explanations(dir_path):
        df_list = []

        for path in os.listdir(dir_path):
            current_path = dir_path + "/" + path
            if current_path.endswith(".csv"):
                df_tmp = pd.read_csv(dir_path + "/" + path)
                df_list.append(df_tmp)

        df = pd.concat(df_list)
        df["tokens"] = df["tokens"].astype(str)
        df["tokens"] = df["tokens"].apply(lambda x: x.replace(" ", ""))
        df["tokens"] = df["tokens"].apply(lambda x: x.lower())
        return df

    @staticmethod
    def _compute_global_scores(df):
        df_grp = df.groupby(["tokens"]).sum().reset_index()

        df_freq = df.groupby(["tokens"]).size().reset_index(name='freq')

        neg_freq_dict = {}

        for token, freq in zip(df_freq.tokens.tolist(), df_freq.freq.tolist()):
            neg_freq_dict[token] = freq

        df_global = df_grp.copy()

        df_global = df_global.join(df_freq.set_index('tokens'), on='tokens')

        df_global["score_norm"] = df_global["explanation_scores"] / df_global["freq"]

        df_global = df_global.drop(columns=["Unnamed: 0"])

        df_global = df_global.sort_values(by='score_norm', ascending=False)

        df_global = df_global.loc[df_global.freq > 1]

        df_global = df_global.loc[~df_global.tokens.str.startswith("##")]

        df_global = df_global.loc[df_global.tokens.str.len() > 2]

        df_global = df_global.loc[~df_global.tokens.str.isdigit()]

        df_global = df_global.loc[df_global.score_norm > 0]

        return df_global


class IntegratedGradientsExplainer(Explainer):
    """ Integrated Gradients implementation of the Explainer abstract class. """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Create IG explainer Ferret library
        self.igx = IntegratedGradientExplainer(model, tokenizer)
        self.bench = Benchmark(model, tokenizer, explainers=[self.igx])
        return

    @staticmethod
    def batch_iterator(list1, list2, batch_size):
        # Ensure both input lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Input lists must have the same length")

        for i in range(0, len(list1), batch_size):
            start_index = i
            end_index = min(i + batch_size, len(list1))
            yield list1[start_index:end_index], list2[start_index:end_index], start_index, end_index


    def local_explanations(self, df_predictions, local_explanations_folder, label_ids_to_explain, id2label, batch_size=128, device=None):
        """ ."""
        log_errors = []
        log_filepath = os.path.join(local_explanations_folder, "log_file.txt")

        start_index = 0

        label_ids_to_explain
        for label_id in label_ids_to_explain:
            label_name = id2label[label_id]

            current_path = os.path.join(local_explanations_folder, label_name)
            if not os.path.exists(current_path):
                os.mkdir(current_path)

            # Select texts predicted with the current label to explain
            df_predictions_current_label = df_predictions.loc[df_predictions.pred_label_name == label_name]

            print(f"Explainer: Explaining label: {label_name} - Number of texts: {len(df_predictions_current_label)}")

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

    def _compute_batch_local_explanations(self, label_id, batch_texts, batch_pred_scores):
        """ ."""
        batch_tokens = []
        batch_scores = []
        batch_scores_weighted = []

        for text, prob in tqdm(zip(batch_texts,batch_pred_scores)):
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

    def local_explanations(self):
        return

    def global_explanations(self):
        return

