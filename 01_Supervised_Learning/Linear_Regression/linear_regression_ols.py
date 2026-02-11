import numpy as np

class LinearRegressionOLS():

    def __init__(self):
        self.bias = None
        self.weights = None

    def fit(self, X, y):

        m, n = np.shape(X)

        ## Step 1 - add bias column

        X_bias = np.c_[np.ones((m,1)), X]   

        ## Step 2 - normal equation

        theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

        ## Step 3 - separate bias and weights

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):

        y_pred = self.bias + np.dot(X, self.weights)
        return y_pred

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = fetch_california_housing()

X = data.data[:2000]   ## smaller subset
y = data.target[:2000]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (use same scaling as GD for fair comparison)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ols_model = LinearRegressionOLS()
ols_model.fit(X_train, y_train)

y_pred_ols = ols_model.predict(X_test)

print("OLS Results")
print("MSE:", mean_squared_error(y_test, y_pred_ols))
print("R2 Score:", r2_score(y_test, y_pred_ols))

plt.scatter(y_test, y_pred_ols)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (OLS)")
plt.show()
