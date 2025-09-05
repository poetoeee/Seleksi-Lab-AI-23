import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        """
        Parameters:
        - X (array-like): Data training.
        - y (array-like): Label target.
        """
        X_fit = np.array(X).astype(float)
        y_fit = np.array(y)
        n_samples, n_features = X_fit.shape
        self._classes = np.unique(y_fit)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X_fit[y_fit == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Parameters:
        - X (array-like): Data yang ingin diprediksi.
        """
        X_pred = np.array(X).astype(float)
        y_pred = [self._predict_single(x) for x in X_pred]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        var = var + 1e-9 
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator