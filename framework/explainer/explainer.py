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

    def __init__(self, model, tokenizer):
        # Create IG explainer Ferret library
        igx = IntegratedGradientExplainer(model, tokenizer)
        bench = Benchmark(model, tokenizer, explainers=[igx])
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