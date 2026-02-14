import numpy as np


class LogisticRegression():
    
    def __init__(self, learning_rate = 0.1, max_iter = 1000):
        
        self.bias = None  ## bias will be different for each model
        self.weights = None  ## weight will be different for each model
        self.lr = learning_rate
        self.iter = max_iter


    def _sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))


    def fit(self, X, y):  ## X = X_train, y = y_train
        
        m, n = np.shape(X)
        
        ## Step 1 - initialize 

        self.bias = 0
        self.weights = np.zeros(n)

        for i in range(self.iter):   

            ## Step 2 - calc z

            z = self.bias + np.dot(X, self.weights) 
            y_pred = self._sigmoid(z)

            ## Step 3 - gradient

            gb = (1/m) * np.sum(y_pred - y)
            gw = (1/m) * np.dot(X.T, (y_pred - y))

            ## Step 4 - convergence

            self.bias -= self.lr * gb
            self.weights -= self.lr * gw


    def get_probabilities(self, X):  ## calculates the probabilities for each value
        
        z = self.bias + np.dot(X, self.weights) 
        return self._sigmoid(z)


    def predict(self, X, threshold=0.5):  ## X is X_test

        probabilities = self.get_probabilities(X) 
        y_pred = probabilities >= threshold  ## classifies the probability into False or True

        return y_pred.astype(int)  ## converts into 0 or 1
