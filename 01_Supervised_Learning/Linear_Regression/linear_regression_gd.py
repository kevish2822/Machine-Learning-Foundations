import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression():
    
    def __init__(self, learning_rate = 0.01, max_iter = 500):
        
        self.bias = None  ## bias will be different for each model
        self.weights = None  ## weight will be different for each model
        self.lr = learning_rate
        self.iter = max_iter
        self.cost_history = []  ## to store cost values


    def fit(self, X, y):  ## X = X_train, y = y_train
        
        m, n = np.shape(X)
        
        ## Step 1 - initialize 

        self.bias = 0
        self.weights = np.zeros(n)

        for i in range(self.iter):   

            ## Step 2 - calc y_pred

            y_pred = self.bias + np.dot(X, self.weights) 
            
            ## Step 3 - gradient

            gb = (1/m) * np.sum(y_pred - y)
            gw = (1/m) * np.dot(X.T, (y_pred - y))

            ## Step 4 - convergence

            self.bias -= self.lr * gb
            self.weights -= self.lr * gw

            ## Step 5 - cost tracking

            cost = (1/(2*m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)


    def predict(self, X):  ## X is X_test

        y_pred = self.bias + np.dot(X, self.weights)
        return y_pred

# Load dataset
data = fetch_california_housing()

X = data.data[:2000]   ## taking smaller subset
y = data.target[:2000]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for GD)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression(learning_rate=0.01, max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.plot(model.cost_history)
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
