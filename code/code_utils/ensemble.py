import numpy as np

class WeightedEnsembleClassifier:
    def __init__(self, models, weights):
        """
        models: list of trained classifiers (must have predict_proba method)
        weights: list of weights (must sum to 1)
        """
        assert len(models) == len(weights), "Each model must have a corresponding weight"
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        probas = [model.predict_proba(X)[:, 1] for model in self.models]
        weighted = sum(float(w) * p for w, p in zip(self.weights, probas))
        return np.vstack([1 - weighted, weighted]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] > threshold).astype(int)
    
class WeihtedEnsembleRegressor:
    def __init__(self, models, weights):
        """
        models: list of trained regressors (must have predict method)
        weights: list of weights (must sum to 1)
        """
        assert len(models) == len(weights), "Each model must have a corresponding weight"
        self.models = models
        self.weights = weights

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        weighted = sum(float(w) * p for w, p in zip(self.weights, predictions))
        return weighted 