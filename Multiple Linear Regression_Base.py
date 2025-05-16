import numpy as np


def estimate_coefficients(X, y):
    """Estimates the coefficients for a multiple linear regression model using pseudo-inverse.

    Args:
        X (numpy.ndarray): The feature matrix (independent variables).
        y (numpy.ndarray): The target vector (dependent variable).

    Returns:
        numpy.ndarray: The coefficient vector.
    """
    # Use pseudo-inverse to handle singular or ill-conditioned matrices
    coefficients = np.linalg.pinv(X) @ y
    return coefficients


def predict(X, coefficients):
    """Predicts the target variable using the regression coefficients.

    Args:
        X (numpy.ndarray): The feature matrix.
        coefficients (numpy.ndarray): The coefficient vector.

    Returns:
        numpy.ndarray: The predicted target values.
    """
    return X @ coefficients


def calculate_r_squared(y_true, y_predicted):
    """
    Calculates the R-squared value to evaluate the model's fit.

    Args:
        y_true (numpy.ndarray): The true target values.
        y_predicted (numpy.ndarray): The predicted target values.

    Returns:
        float: The R-squared value.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_predicted) ** 2)
    return 1 - (ss_residual / ss_total)


# Example Usage
# Sample data (replace with your actual data)
X = np.array([[1, 2, 3],
              [1, 4, 5],
              [1, 6, 7],
              [1, 8, 9]])  # Feature matrix with a column of ones for the intercept
y = np.array([4, 8, 12, 16])  # Target vector

# Estimate coefficients
coefficients = estimate_coefficients(X, y)
print("Coefficients:", coefficients)

# Make predictions
predictions = predict(X, coefficients)
print("Predictions:", predictions)

# Evaluate the model
r_squared = calculate_r_squared(y, predictions)
print("R-squared:", r_squared)
