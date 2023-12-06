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
    def local_explanations(self):
        return

    @abstractmethod
    def global_explanations(self):
        return


class IntegratedGradientsExplainer(Explainer):
    """ """
    def __init__(self, model, tokenizer):
        #
        self.model = model
        self.tokenizer = tokenizer

        # Create IG explainer Ferret library
        self.igx = IntegratedGradientExplainer(model, tokenizer)
        self.bench = Benchmark(model, tokenizer, explainers=[self.igx])
        return


    def local_explanations(self, df_predictions, local_explanations_folder, labels2explain=None,  batch_size=128):
        log_errors = []
        log_filepath = os.path.join(local_explanations_folder, "log_file.txt")


        if labels2explain is None:
            labels2explain = labels2explain

        start_index = 0
        for label_id, l in enumerate(labels2explain):

            current_path = os.path.join(local_explanations_folder, l)
            if not os.path.exists(current_path):
                os.mkdir(current_path)

            df_predictions_current_label = df_predictions.loc[df_predictions.pred_labels == l]
            print(f"Explainer: Explaining label: {l} - Number of texts: {len(df_predictions_current_label)}")

            iterations = len(df_predictions_current_label["text"].tolist()[start_index:]) // batch_size
            remainder = len(df_predictions_current_label["text"].tolist()[start_index:]) % batch_size

            for i in range(iterations):
                try:
                    index_a = start_index + i * batch_size
                    index_b = start_index + i * batch_size + batch_size
                    print(f"\t Explaining Batch: {index_a} -> {index_b}")

                    batch_texts = df_predictions_current_label["text"].tolist()[index_a:index_b]
                    batch_predicted_prob_explained_class = df_predictions_current_label["pred_probs"].tolist()[index_a:index_b]

                    batch_tokens, batch_scores, batch_scores_weighted = self._compute_batch_local_explanations(label_id, batch_texts, batch_predicted_prob_explained_class)

                    df = pd.DataFrame({"tokens": batch_tokens, "explanation_scores": batch_scores, "explanation_scores_weighted": batch_scores_weighted})
                    df.to_csv(os.path.join(current_path, f"scores_{index_a}_{index_b}.csv"))

                except Exception as e:
                    print(f"An exception occurred in {index_a} -> {index_b}")
                    print(e)
                    log_errors.append(f"An exception occurred in {index_a} -> {index_b} in label {l}")
                    with open(log_filepath, "a") as file_object:
                        file_object.write(f"\nAn exception occurred in {index_a} -> {index_b} in label {l}")

                print(f"{len(df_test_selected[text_column_name].tolist()) - remainder} -> {len(df_test_selected[text_column_name].tolist())}")
                for text, prob in zip(df_test_selected[text_column_name].tolist()[-remainder:],
                                      df_test_selected["pred_probs"].tolist()[-remainder:]):

                    batch_texts = df_predictions_current_label["text"].tolist()[-remainder:]
                    batch_predicted_prob_explained_class = df_predictions_current_label["pred_probs"].tolist()[-remainder:]


                    batch_tokens, batch_scores, batch_scores_weighted = self._compute_batch_local_explanations(label_id, batch_texts, batch_predicted_prob_explained_class)

                    df.to_csv(os.path.join(current_path,
                                           f"scores_{len(df_test_selected[text_column_name].tolist()) - remainder}_end.csv"))

        return

    def _compute_batch_local_explanations(self, label_id, batch_texts, batch_pred_scores):
        batch_tokens = []
        batch_scores = []
        batch_scores_weighted = []

        for text, prob in tqdm(zip(batch_texts,batch_pred_scores)):
            text = str(text)

            #tokenized_text = self.tokenizer.tokenize(text)
            #if len(tokenized) > 320:
            #    text = self.tokenizer.convert_tokens_to_string(tokenized[:320])

            explanations = self.bench.explain(text, target=label_id, show_progress=False, normalize_scores=True)

            tokens = explanations[0].tokens[1:-1]

            scores_raw = explanations[0].scores[1:-1].tolist()

            scores_weighted = [s * prob for s in scores_raw]

            batch_tokens.extend(tokens)
            batch_scores.extend(scores_raw)
            batch_scores_weighted.extend(scores_weighted)
        return batch_tokens, batch_scores, batch_scores_weighted


    def global_explanations(self):
        return


class ShapExplainer(Explainer):

    def local_explanations(self):
        return

    def global_explanations(self):
        return

