'''
CS 6364.002

Homework 5
----------
Name:       Kevin Chen
NetID:      nkc160130
Instructor: Professor Chen
Due:        09/28/2020

Collection of Regression models
'''
import numpy as np
import pandas as pd
from secrets import SystemRandom
from math import exp


# Linear Regression with Gradient Descent
class LinearReg_GD:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.num_iterations = 2000

    def __calc_grad(self, X, Y):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Iterate through all rows of x
        for i in range(0, np.shape(X)[0]):
            Xi = X[i].reshape((np.shape(X)[1], 1))
            sum_error += np.asscalar((self.betas.T.dot(Xi) - Y[i])) * Xi

        return sum_error / np.shape(X)[0]

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Calculate gradient for each beta
            new_betas -= self.learning_rate * self.__calc_grad(X, Y)

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        return self.betas.T.dot(X.T).T

    def mean_squared_error(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        # Calculate mean squared error
        return np.average((Y - predictions) ** 2)


# Linear Regression with Stochastic Gradient Descent
class LinearReg_SGD:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.num_iterations = 10000

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += np.asscalar((self.betas.T.dot(Xi) - Y[i])) * Xi

        return sum_error / np.shape(X)[0]

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            # Calculate gradient for each beta
            new_betas -= self.learning_rate * self.__calc_grad(X, Y, randI)

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        return self.betas.T.dot(X.T).T

    def mean_squared_error(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        # Calculate mean squared error
        return np.average((Y - predictions) ** 2)


# Linear Regression with Stochastic Gradient Descent with Momentum
class LinearReg_SGD_Momentum:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.velocity = np.empty(shape=(0, 0))
        self.eta = 0.9
        self.num_iterations = 10000

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += np.asscalar((self.betas.T.dot(Xi) - Y[i])) * Xi

        return sum_error / np.shape(X)[0]

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values and velocity values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))
        self.velocity = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            self.velocity = (self.eta * self.velocity) - \
                (self.learning_rate * self.__calc_grad(X, Y, randI))

            # Calculate gradient for each beta
            new_betas += self.velocity

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        return self.betas.T.dot(X.T).T

    def mean_squared_error(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        # Calculate mean squared error
        return np.average((Y - predictions) ** 2)


# Linear Regression with Stochastic Gradient Descent with Nesterov Momentum
class LinearReg_SGD_Nesterov:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.velocity = np.empty(shape=(0, 0))
        self.eta = 0.9
        self.num_iterations = 10000

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += np.asscalar(((self.betas +
                                   (self.eta * self.velocity)).T.dot(Xi) - Y[i])) * Xi

        return sum_error / np.shape(X)[0]

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values and velocity values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))
        self.velocity = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            self.velocity = (self.eta * self.velocity) - \
                (self.learning_rate * self.__calc_grad(X, Y, randI))

            # Calculate gradient for each beta
            new_betas += self.velocity

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        return self.betas.T.dot(X.T).T

    def mean_squared_error(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        # Calculate mean squared error
        return np.average((Y - predictions) ** 2)


# Linear Regression with AdaGrad
class LinearReg_AdaGrad:
    def __init__(self):
        self.learning_rate = 1
        self.betas = np.empty(shape=(0, 0))
        self.velocity = np.empty(shape=(0, 0))
        self.r = np.empty(shape=(0, 0))
        self.delta = np.empty(shape=(0, 0))
        self.num_iterations = 10000

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += np.asscalar(self.betas.T.dot(Xi) - Y[i]) * Xi

        return sum_error / np.shape(X)[0]

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values and velocity values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))
        self.velocity = np.zeros(shape=(np.shape(X)[1], 1))
        self.r = np.zeros(shape=(np.shape(X)[1], 1))
        self.delta = np.full(shape=(np.shape(X)[1], 1), fill_value=10e-8)

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            # Calculate gradient
            g = self.__calc_grad(X, Y, randI)

            # Calculate r
            self.r += g * g

            # Calculate velocity
            self.velocity = (
                (self.learning_rate / np.sqrt(self.delta + self.r)) * g) * -1

            # Calculate gradient for each beta
            new_betas += self.velocity

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        return self.betas.T.dot(X.T).T

    def mean_squared_error(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        # Calculate mean squared error
        return np.average((Y - predictions) ** 2)


# Logistic Regression with Gradient Descent
class LogisticReg_GD:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.num_iterations = 2000

    def sigmoid(self, z):
        return 1.0 / (exp(z * -1) + 1)

    def __calc_grad(self, X, Y):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Iterate through all rows of x
        for i in range(0, np.shape(X)[0]):
            Xi = X[i].reshape((np.shape(X)[1], 1))
            sum_error += (Y[i] -
                          self.sigmoid(np.asscalar(self.betas.T.dot(Xi)))) * Xi

        return sum_error / np.shape(X)[0] * -1

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Calculate gradient for each beta
            new_betas -= self.learning_rate * self.__calc_grad(X, Y)

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        predictions = self.betas.T.dot(X.T).T

        predictions_sigmoid = np.empty(shape=np.shape(predictions))

        for i in range(0, len(predictions)):
            predictions_sigmoid[i] = np.round(self.sigmoid(predictions[i]))

        return predictions_sigmoid

    def accuracy_score(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        true_positives = 0
        true_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 0:
                true_negatives += 1

        # Calculate accuracy
        return (true_positives + true_negatives) / np.shape(Y)[0]

    def classification_report(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        metrics = {"precision": [], "recall": [], "f1_score": []}

        # Metrics for category 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 0 and predictions[i] == 0:
                true_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        # Metrics for category 1
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        return pd.DataFrame(metrics)


# Logistic Regression with Stochastic Gradient Descent
class LogisticReg_SGD:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.num_iterations = 10000

    def sigmoid(self, z):
        return 1.0 / (exp(z * -1) + 1)

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += (Y[i] -
                      self.sigmoid(np.asscalar(self.betas.T.dot(Xi)))) * Xi

        return sum_error / np.shape(X)[0] * -1

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            # Calculate gradient for each beta
            new_betas -= self.learning_rate * self.__calc_grad(X, Y, randI)

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        predictions = self.betas.T.dot(X.T).T

        predictions_sigmoid = np.empty(shape=np.shape(predictions))

        for i in range(0, len(predictions)):
            predictions_sigmoid[i] = np.round(self.sigmoid(predictions[i]))

        return predictions_sigmoid

    def accuracy_score(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        true_positives = 0
        true_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 0:
                true_negatives += 1

        # Calculate accuracy
        return (true_positives + true_negatives) / np.shape(Y)[0]

    def classification_report(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        metrics = {"precision": [], "recall": [], "f1_score": []}

        # Metrics for category 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 0 and predictions[i] == 0:
                true_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        # Metrics for category 1
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        return pd.DataFrame(metrics)


# Logistic Regression with Stochastic Gradient Descent with Momentum
class LogisticReg_SGD_Momentum:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.velocity = np.empty(shape=(0, 0))
        self.eta = 0.9
        self.num_iterations = 10000

    def sigmoid(self, z):
        return 1.0 / (exp(z * -1) + 1)

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += (Y[i] -
                      self.sigmoid(np.asscalar(self.betas.T.dot(Xi)))) * Xi

        return sum_error / np.shape(X)[0] * -1

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values and velocity values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))
        self.velocity = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            self.velocity = (self.eta * self.velocity) - \
                (self.learning_rate * self.__calc_grad(X, Y, randI))

            # Calculate gradient for each beta
            new_betas += self.velocity

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        predictions = self.betas.T.dot(X.T).T

        predictions_sigmoid = np.empty(shape=np.shape(predictions))

        for i in range(0, len(predictions)):
            predictions_sigmoid[i] = np.round(self.sigmoid(predictions[i]))

        return predictions_sigmoid

    def accuracy_score(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        true_positives = 0
        true_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 0:
                true_negatives += 1

        # Calculate accuracy
        return (true_positives + true_negatives) / np.shape(Y)[0]

    def classification_report(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        metrics = {"precision": [], "recall": [], "f1_score": []}

        # Metrics for category 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 0 and predictions[i] == 0:
                true_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        # Metrics for category 1
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        return pd.DataFrame(metrics)


# Logistic Regression with Stochastic Gradient Descent with Nesterov Momentum
class LogisticReg_SGD_Nesterov:
    def __init__(self):
        self.learning_rate = 0.001
        self.betas = np.empty(shape=(0, 0))
        self.velocity = np.empty(shape=(0, 0))
        self.eta = 0.9
        self.num_iterations = 10000

    def sigmoid(self, z):
        return 1.0 / (exp(z * -1) + 1)

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += (Y[i] -
                      self.sigmoid(np.asscalar(((self.betas + (self.eta * self.velocity)).T.dot(Xi))))) * Xi

        return sum_error / np.shape(X)[0] * -1

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values and velocity values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))
        self.velocity = np.zeros(shape=(np.shape(X)[1], 1))

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            self.velocity = (self.eta * self.velocity) - \
                (self.learning_rate * self.__calc_grad(X, Y, randI))

            # Calculate gradient for each beta
            new_betas += self.velocity

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        predictions = self.betas.T.dot(X.T).T

        predictions_sigmoid = np.empty(shape=np.shape(predictions))

        for i in range(0, len(predictions)):
            predictions_sigmoid[i] = np.round(self.sigmoid(predictions[i]))

        return predictions_sigmoid

    def accuracy_score(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        true_positives = 0
        true_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 0:
                true_negatives += 1

        # Calculate accuracy
        return (true_positives + true_negatives) / np.shape(Y)[0]

    def classification_report(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        metrics = {"precision": [], "recall": [], "f1_score": []}

        # Metrics for category 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 0 and predictions[i] == 0:
                true_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        # Metrics for category 1
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        return pd.DataFrame(metrics)


# Logistic Regression with AdaGrad
class LogisticReg_AdaGrad:
    def __init__(self):
        self.learning_rate = 1
        self.betas = np.empty(shape=(0, 0))
        self.velocity = np.empty(shape=(0, 0))
        self.r = np.empty(shape=(0, 0))
        self.delta = np.empty(shape=(0, 0))
        self.num_iterations = 10000

    def sigmoid(self, z):
        return 1.0 / (exp(z * -1) + 1)

    def __calc_grad(self, X, Y, i):
        sum_error = np.zeros(shape=(np.shape(X)[1], 1))

        # Get row i
        Xi = X[i].reshape((np.shape(X)[1], 1))
        sum_error += (Y[i] -
                      self.sigmoid(np.asscalar(self.betas.T.dot(Xi)))) * Xi

        return sum_error / np.shape(X)[0] * -1

    def fit(self, X_train, Y_train):
        # Initialize x and y
        X = np.empty(shape=(0, 0))
        Y = np.empty(shape=(0, 0))

        if isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.core.series.Series):
            X = pd.DataFrame.to_numpy(X_train)
            Y = pd.DataFrame.to_numpy(Y_train)
        elif isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            X = X_train
            Y = Y_train

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError(
                "There is a NaN value in either your X or Y array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Initialize beta values and velocity values
        self.betas = np.zeros(shape=(np.shape(X)[1], 1))
        self.velocity = np.zeros(shape=(np.shape(X)[1], 1))
        self.r = np.zeros(shape=(np.shape(X)[1], 1))
        self.delta = np.full(shape=(np.shape(X)[1], 1), fill_value=10e-8)

        # Go through num_iterations epochs
        for epoch in range(0, self.num_iterations):
            new_betas = self.betas.copy()

            # Get random row of X and Y
            randI = SystemRandom().randint(0, np.shape(X)[0] - 1)

            # Calculate gradient
            g = self.__calc_grad(X, Y, randI)

            # Calculate r
            self.r += g * g

            # Calculate velocity
            self.velocity = (
                (self.learning_rate / np.sqrt(self.delta + self.r)) * g) * -1

            # Calculate gradient for each beta
            new_betas += self.velocity

            # Make these the updated values for beta matrix
            self.betas = new_betas.copy()

    def predict(self, X_test):
        # Initialize X
        X = np.empty(shape=(0, 0))
        if isinstance(X_test, pd.DataFrame):
            X = pd.DataFrame.to_numpy(X_test)
        elif isinstance(X_test, np.ndarray):
            X = X_test

        # Check for NaN in numpy arrays
        if np.any(np.isnan(X)):
            raise ValueError("There is a NaN value in your X array.")

        # Add bias to X (column of 1's)
        X = np.insert(X, 0, [1] * np.shape(X)[0], axis=1)

        # Calculate prediction
        predictions = self.betas.T.dot(X.T).T

        predictions_sigmoid = np.empty(shape=np.shape(predictions))

        for i in range(0, len(predictions)):
            predictions_sigmoid[i] = np.round(self.sigmoid(predictions[i]))

        return predictions_sigmoid

    def accuracy_score(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        true_positives = 0
        true_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 0:
                true_negatives += 1

        # Calculate accuracy
        return (true_positives + true_negatives) / np.shape(Y)[0]

    def classification_report(self, Y_test, predictions):
        # Initialize Y
        Y = np.empty(shape=(0, 0))
        if isinstance(Y_test, pd.core.series.Series):
            Y = pd.DataFrame.to_numpy(Y_test)
        elif isinstance(Y_test, np.ndarray):
            Y = Y_test

        Y = Y.reshape((np.shape(Y)[0], 1))

        # Check for NaN in numpy arrays
        if np.any(np.isnan(Y)):
            raise ValueError("There is a NaN value in your Y array.")

        metrics = {"precision": [], "recall": [], "f1_score": []}

        # Metrics for category 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 0 and predictions[i] == 0:
                true_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        # Metrics for category 1
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(0, np.shape(Y)[0]):
            if Y[i] == 1 and predictions[i] == 1:
                true_positives += 1
            if Y[i] == 0 and predictions[i] == 1:
                false_positives += 1
            if Y[i] == 1 and predictions[i] == 0:
                false_negatives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        metrics["precision"].append(precision)

        # Calculate recall
        recall = true_positives / (true_positives + false_negatives)
        metrics["recall"].append(recall)

        # Calculate F-1 score
        f1_score = (2 * precision * recall) / (precision + recall)
        metrics["f1_score"].append(f1_score)

        return pd.DataFrame(metrics)
