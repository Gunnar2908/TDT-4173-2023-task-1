import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.bias = 0
        self.w_0 = 0
        self.w_1 = 0
        self.w_2 = 0
        self.w_3 = 0
        self.weights = np.array([self.w_0, self.w_1, self.w_2, self.w_3])
        self.learning_rate = 0.01
        self.loss_history = []

        
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        
        print(x.shape[1])
        print(self.weights.shape[0])
        x = self.prepare_data(x)
        for epoch in range(10000):
            prediction = self.predict(x)
            print(y.shape)
            print(prediction.shape)
            difference: np.array = y-prediction
            self.weights = self.weights + self.learning_rate * np.dot(difference.T, x)

            self.bias = self.bias + self.learning_rate * (np.sum(y) - np.sum(prediction))

            loss = binary_cross_entropy(y, prediction)
            self.loss_history.append((loss, epoch))



    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        if x.shape[1] != self.weights.shape[0]:
            x = self.prepare_data(x)
        predictions: np.array = np.dot(x, self.weights.T)
        predictions += self.bias
        predictions = sigmoid(predictions)
        return predictions
        
    @staticmethod      
    def plot_xy_pairs(xy_pairs):
        """
        Plots x,y values from a list of (x,y) value pairs.

        Parameters:
        - xy_pairs: List of (x,y) tuples
        """

        # Unzip the x and y values
        x_values, y_values = zip(*xy_pairs)

        plt.figure(figsize=(10,6))
        plt.scatter(x_values, y_values, color='blue', marker='o', label='Data Points')
        plt.xlabel('Iterations')
        plt.ylabel('Loss Values')
        plt.title('Scatter Plot of X,Y Value Pairs')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def prepare_data(self, x: np.ndarray):
        new_np: np.ndarray = x**2
        total_array = np.concatenate((x, new_np), axis=1)
        return total_array
        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    # assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))




if __name__ == "__main__":
    data = pd.read_csv("TDT-4173-2023-task-1\logistic_regression\data_1.csv")
    print(data)
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', hue='y', data=data)
    x = data[['x0', 'x1']]
    y = data[['y']]
    model = LogisticRegression()
    model.fit(x,y)
    print(model.predict(x))