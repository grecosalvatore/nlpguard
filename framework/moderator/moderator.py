from abc import ABC, abstractmethod


class Moderator(ABC):
    """ Abstract Moderator Class. """
    def __init__(self):
        return

    def remove_words_mitigation(self):
        return

    def remove_sentences_mitigation(self):
        return



class PandasDataFrameModerator(Moderator):
    def __init__(self):
        return


class HFDatasetModerator(Moderator):
    def __init__(self):
        return