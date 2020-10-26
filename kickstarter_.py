from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ImputerDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.cols = []
        
    def fit(self, X, y=None):
        self.imputer.fit(X)
        self.cols = list(X.columns)
        return self
    
    def transform(self, X):
        X_t = self.imputer.transform(X)
        return pd.DataFrame(X_t, columns=self.cols)