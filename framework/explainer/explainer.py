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
    def local_explanations(self):
        return

    def global_explanations(self):
        return


class SHAPExplainer(Explainer):
    def local_explanations(self):
        return

    def global_explanations(self):
        return