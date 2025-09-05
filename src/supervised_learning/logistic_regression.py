import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, add_intercept=True):
        """
        Parameters:
        - learning_rate (float): Tingkat pembelajaran untuk gradient descent.
        - n_iters (int): Jumlah iterasi maksimum.
        - add_intercept (bool): Apakah akan menambahkan bias/intercept term.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.add_intercept = add_intercept
        self.weights = None
        self.bias = None 

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Parameters:
        - X (array-like): Data training.
        - y (array-like): Label target.
        """
        X_fit = np.array(X).astype(float) 
        y_fit = np.array(y)
        
        if self.add_intercept:
            X_fit = self._add_intercept(X_fit)
        
        n_samples, n_features = X_fit.shape
        self.weights = np.zeros(n_features)

        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X_fit, self.weights)
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X_fit.T, (y_predicted - y_fit))
            self.weights -= self.lr * dw

    def predict_proba(self, X):
        X_pred = np.array(X).astype(float)
        if self.add_intercept:
            X_pred = self._add_intercept(X_pred)
            
        linear_model = np.dot(X_pred, self.weights)
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Parameters:
        - X (array-like): Data yang ingin diprediksi.
        - threshold (float): Batas untuk klasifikasi biner.
        """
        probas = self.predict_proba(X)
        return [1 if i > threshold else 0 for i in probas]