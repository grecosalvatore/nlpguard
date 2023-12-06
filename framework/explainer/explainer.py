from ferret import Benchmark
from ferret import IntegratedGradientExplainer

from abc import ABC, abstractmethod


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


    def local_explanations(self, batch_size=128):
        return

    def global_explanations(self):
        return


class ShapExplainer(Explainer):
    def local_explanations(self):
        return

    def global_explanations(self):
        return