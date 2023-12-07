from abc import ABC, abstractmethod
import pandas as pd

class Moderator(ABC):
    """ Abstract Moderator Class. """
    def __init__(self):
        return

    @abstractmethod
    def words_removal_mitigation_strategy(self, **kwargs):
        return

    @abstractmethod
    def sentences_removal_mitigation_strategy(self, **kwargs):
        return



class PandasDataFrameModerator(Moderator):
    def __init__(self):
        return

    def words_removal_mitigation_strategy(self, df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, mitigate_each_label_separately=False, batch_size=128):
        if mitigate_each_label_separately == False:
            # Remove words related to protected attributes from all labels
            distinct_protected_attributes = list(set(word for words_list in protected_attributes_per_label_dict.values() for word in words_list))

            # Apply the mitigation strategy in batches to all class labels
            df_train['mitigated_text'] = pd.concat([pd.Series(self._batch_words_removal(df_train[text_column_name][i:i + batch_size], tokenizer, distinct_protected_attributes)) for i in range(0, df_train.shape[0], batch_size)], ignore_index=True)
        return df_train


    @staticmethod
    def _batch_words_removal(texts, tokenizer, words_to_remove):
        # Tokenize the texts in batch
        encoded_inputs = tokenizer(texts.tolist(), add_special_tokens=False, return_tensors='pt', padding=True)

        # Process each text
        clean_texts = []
        for i in range(encoded_inputs.input_ids.size(0)):
            # Ensure input_ids[i] is 1-dimensional
            input_ids = encoded_inputs.input_ids[i]
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)

            # Convert token IDs to tokens
            tokens = [tokenizer.convert_ids_to_tokens(id.item()) for id in input_ids]

            # Remove unwanted tokens and reconstruct the text
            tokens = [t for t in tokens if t not in words_to_remove]
            clean_texts.append(tokenizer.convert_tokens_to_string(tokens))

        return clean_texts

    def sentences_removal_mitigation_strategy(self, **kwargs):
        return




class HuggingFaceDatasetModerator(Moderator):
    def __init__(self):
        return