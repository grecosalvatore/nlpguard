from abc import ABC, abstractmethod


class Moderator(ABC):
    """ Abstract Moderator Class. """
    def __init__(self):
        return

    @abstractmethod
    def words_removal_mitigation_strategy(self):
        return

    @abstractmethod
    def sentences_removal_mitigation_strategy(self):
        return



class PandasDataFrameModerator(Moderator):
    def __init__(self):
        return

    def words_removal_mitigation_strategy(self, df, protected_attributes, tokenizer):
        return


class HuggingFaceDatasetModerator(Moderator):
    def __init__(self):
        return