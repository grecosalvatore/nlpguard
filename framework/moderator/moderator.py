from abc import ABC, abstractmethod
import pandas as pd
import nltk
from nltk.corpus import wordnet
import gensim.downloader as api

class Moderator(ABC):
    """ Abstract Moderator Class. """
    def __init__(self):
        return

    @abstractmethod
    def words_removal_mitigation_strategy(self, **kwargs):
        """ Abstract method to perform the word removal mitigation strategy. """
        return

    @abstractmethod
    def sentences_removal_mitigation_strategy(self, **kwargs):
        """ Abstract method to perform the sentence removal mitigation strategy. """
        return

    @abstractmethod
    def word_replacement_with_synonyms_mitigation_strategy(self, **kwargs):
        """ Abstract method to perform the word replacement with synonyms mitigation strategy. """
        return

    @abstractmethod
    def word_replacement_with_hypernym_mitigation_strategy(self, **kwargs):
        """ Abstract method to perform the word replacement with hypernym mitigation strategy. """
        return


class PandasDataFrameModerator(Moderator):
    """ Moderator Implementation for Pandas DataFrames. """
    def __init__(self):
        super().__init__()
        return

    def words_removal_mitigation_strategy(self, df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, id2label, mitigate_each_label_separately=False, batch_size=128):
        """ Performs the word removal mitigation strategy.
        Args:
            df_train (:obj:`pandas.DataFrame`): Training dataset.
            tokenizer (:obj:`transformers.AutoTokenizer`): Tokenizer.
            protected_attributes_per_label_dict (:obj:dict): Dictionary of protected attributes per class label.
            text_column_name (str): Name of the column containing the text.
            label_column_name (str): Name of the column containing the class label.
            id2label (:obj:dict): Dictionary mapping class label ids to class label names.
            mitigate_each_label_separately (bool, optional): Whether to mitigate each class label separately.
            batch_size (int, optional): Batch size.
        """
        if mitigate_each_label_separately == False:
            # Remove words related to protected attributes from all labels
            distinct_protected_attributes = list(set(word for words_list in protected_attributes_per_label_dict.values() for word in words_list))

            # Apply the mitigation strategy in batches to all class labels
            df_train['mitigated_text'] = pd.concat([pd.Series(self._batch_words_removal(df_train[text_column_name][i:i + batch_size], tokenizer, distinct_protected_attributes)) for i in range(0, df_train.shape[0], batch_size)], ignore_index=True)
        else:
            # Apply the mitigation strategy in batches to each class label separately
            for label_id in protected_attributes_per_label_dict.keys():
                # Remove words related to protected attributes from current label
                distinct_protected_attributes = list(set(word for words_list in protected_attributes_per_label_dict[label_id] for word in words_list))

                label_name = id2label[label_id]

                # Apply the mitigation strategy in batches to current class label
                df_train.loc[df_train[label_column_name] == label_name, 'mitigated_text'] = pd.concat([pd.Series(self._batch_words_removal(df_train.loc[df_train[label_column_name] == label_name, text_column_name][i:i + batch_size], tokenizer, distinct_protected_attributes)) for i in range(0, df_train.loc[df_train[label_column_name] == label_name].shape[0], batch_size)], ignore_index=True)
        return df_train

    @staticmethod
    def _batch_words_removal(texts, tokenizer, protected_attributes):
        """ Removes words from texts in batch.
        Args:
            texts (:obj:`list` of :obj:`str`): List of texts.
            tokenizer (:obj:`transformers.AutoTokenizer`): Tokenizer.
            protected_attributes (:obj:`list` of :obj:`str`): List of protected attributes to remove.
        """
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
            tokens = [t for t in tokens if t not in protected_attributes]
            clean_texts.append(tokenizer.convert_tokens_to_string(tokens))

        return clean_texts

    @staticmethod
    def _batch_sentences_removal(texts, tokenizer, protected_attributes):
        """ Removes sentences from texts in batch.
        Args:
            texts (:obj:`list` of :obj:`str`): List of texts.
            tokenizer (:obj:`transformers.AutoTokenizer`): Tokenizer.
            protected_attributes (:obj:`list` of :obj:`str`): List of protected attributes to remove.
        """
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

            contains_protected_attribute = False
            for word in protected_attributes:
                if word in tokens:
                    contains_protected_attribute = True
                    break

            if contains_protected_attribute:
                clean_texts.append(None)
            else:
                clean_texts.append(texts.tolist()[i])

        return clean_texts

    def sentences_removal_mitigation_strategy(self, df_train, tokenizer, protected_attributes_per_label_dict, text_column_name, label_column_name, id2label, mitigate_each_label_separately=False, batch_size=128):
        """ Performs the sentence removal mitigation strategy.
        Args:
            df_train (:obj:`pandas.DataFrame`): Training dataset.
            tokenizer (:obj:`transformers.AutoTokenizer`): Tokenizer.
            protected_attributes_per_label_dict (:obj:dict): Dictionary of protected attributes per class label.
            text_column_name (str): Name of the column containing the text.
            label_column_name (str): Name of the column containing the class label.
            mitigate_each_label_separately (bool, optional): Whether to mitigate each class label separately.
        """
        if mitigate_each_label_separately == False:
            # Remove sentences related to protected attributes from all labels
            distinct_protected_attributes = list(set(word for words_list in protected_attributes_per_label_dict.values() for word in words_list))

            # Apply the mitigation strategy in batches to all class labels
            df_train['mitigated_text'] = pd.concat([pd.Series(self._batch_sentences_removal(df_train[text_column_name][i:i + batch_size], tokenizer, distinct_protected_attributes)) for i in range(0, df_train.shape[0], batch_size)], ignore_index=True)
        else:
            # Apply the mitigation strategy in batches to each class label separately
            for label_id in protected_attributes_per_label_dict.keys():
                # Remove words related to protected attributes from current label
                distinct_protected_attributes = list(set(word for words_list in protected_attributes_per_label_dict[label_id] for word in words_list))

                label_name = id2label[label_id]

                # Apply the mitigation strategy in batches to current class label
                df_train.loc[df_train[label_column_name] == label_name, 'mitigated_text'] = pd.concat([pd.Series(self._batch_sentences_removal(df_train.loc[df_train[label_column_name] == label_name, text_column_name][i:i + batch_size], tokenizer, distinct_protected_attributes)) for i in range(0, df_train.loc[df_train[label_column_name] == label_name].shape[0], batch_size)], ignore_index=True)
        return df_train

    # TODO - Implement the following mitigation strategies for Pandas DataFrames
    def word_replacement_with_synonyms_mitigation_strategy(self, **kwargs):
        # TODO
        print("Loading word vectors...")
        glove_word_embedding = self.load_embedding_model()
        print("Word vectors loaded.")
        self._get_synonyms()

        return

    def word_replacement_with_hypernym_mitigation_strategy(self, **kwargs):
        # TODO
        # Download the WordNet corpus
        nltk.download('wordnet')

        self._get_hypernyms()
        return

    @staticmethod
    def _get_synonyms(word_list, glove_word_embedding, k=5):
        """" Returns the synonyms of the words in the given list."""
        synonyms_dict = {}
        for word in word_list:
            if word in glove_word_embedding:
                k_words_raw = glove_word_embedding.most_similar(word)
                k_words = [word[0] for word in k_words_raw[:k]]
                synonyms_dict[word] = k_words
        return synonyms_dict

    @staticmethod
    def load_GloVe_embedding_model(model_name="glove-wiki-gigaword-300"):
        """ Load GloVe Vectors
            Return:
                wv_from_bin: All 400000 embeddings, each lengh 200
        """
        wv_from_bin = api.load(model_name)
        return wv_from_bin

    @staticmethod
    def _get_hypernyms(word_list):
        """" Returns the hypernyms of the words in the given list."""
        hypernyms_dict = {}
        for word in word_list:
            print(word)
            my_syns = [wordnet.synsets(word)[0]]
            if len(my_syns) > 0:
                ss_list = []
                for ss in my_syns:
                    ss_list.extend(ss.hypernyms())
                if len(ss_list) > 0:
                    hypernyms_dict[word] = ss_list[0].name()

        for w, h in hypernonyms_dict.items():
            tmp_h = h.split(".", 1)[0]
            tmp_h = tmp_h.replace("_", " ")
            hypernyms_dict[w] = tmp_h
        return hypernyms_dict


class HuggingFaceDatasetModerator(Moderator):
    # TODO - Implement the HUGGINGFACE Dataset Moderator
    def __init__(self):
        super().__init__()
        return

    def words_removal_mitigation_strategy(self, **kwargs):
        return

    def sentences_removal_mitigation_strategy(self, **kwargs):
        return

    def word_replacement_with_synonyms_mitigation_strategy(self, **kwargs):
        return

    def word_replacement_with_hypernym_mitigation_strategy(self, **kwargs):
        return