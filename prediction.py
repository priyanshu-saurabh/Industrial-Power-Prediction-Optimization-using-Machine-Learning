import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    required_columns = ["Temp", "machineUsages", "rate", "humidity", "power"]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

    data[required_columns] = data[required_columns].apply(pd.to_numeric, errors="coerce")
    data.fillna(data.mean(), inplace=True)

    X = data[["Temp", "machineUsages", "rate", "humidity"]].values
    y = data["power"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def predict(X, weights, bias):
    return np.dot(X, weights) + bias

def compute_cost(X, y, weights, bias):
    predictions = predict(X, weights, bias)
    return np.mean((predictions - y) ** 2) / 2

def gradient_descent(X, y, weights, bias, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = predict(X, weights, bias)
        dw = (1 / m) * np.dot(X.T, predictions - y)
        db = (1 / m) * np.sum(predictions - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        cost = compute_cost(X, y, weights, bias)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return weights, bias, cost_history

def optimize_features(desired_power, weights, bias, scaler):
    def objective(features):
        norm_features = scaler.transform([features])
        return np.abs(desired_power - predict(norm_features, weights, bias))

    bounds = [(-3, 3)] * len(weights)
    result = minimize(objective, [0]*len(weights), bounds=bounds)

    if not result.success:
        raise RuntimeError("Optimization failed.")

    return scaler.inverse_transform([result.x])[0]

def evaluate_model(y_true, y_pred):
    error_percent = np.mean(np.abs(y_pred - y_true) / y_true) * 100
    return 100 - error_percent

def main():
    file_path = "data_new.csv"
    X, y, scaler = load_data(file_path)

    weights = np.zeros(X.shape[1])
    bias = 0.0

    weights, bias, _ = gradient_descent(X, y, weights, bias)

    predictions = predict(X, weights, bias)
    efficiency = evaluate_model(y, predictions)

    print(f"Final Model Efficiency: {efficiency:.2f}%")

    new_samples = np.array([
        [45.0, 84, 120.0, 60.0],
        [35.0, 65, 100.0, 55.0]
    ])
    new_scaled = scaler.transform(new_samples)
    new_predictions = predict(new_scaled, weights, bias)

    for i, pred in enumerate(new_predictions):
        print(f"Prediction {i+1}: {pred:.2f} units of power consumption")

    desired_power = 150.0
    optimized = optimize_features(desired_power, weights, bias, scaler)
    print(f"Optimized input for {desired_power} units:")
    print(f"Temperature: {optimized[0]:.2f}, Utilization: {optimized[1]:.2f}, Rate: {optimized[2]:.2f}, Humidity: {optimized[3]:.2f}")

if __name__ == "__main__":
    main()
