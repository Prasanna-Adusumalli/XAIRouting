import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def train_model_and_explainer(X, y):
    """
    Train a model and create a SHAP explainer.

    Parameters:
    - X: Features for training
    - y: Target variable for training

    Returns:
    - model: Trained model
    - explainer: SHAP explainer
    """
    # Train a RandomForest model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Create a SHAP explainer
    explainer = shap.Explainer(model, X)
    return model, explainer

def test_shap_fidelity(explainer, model, instance):
    """
    Test the fidelity of SHAP explanations by comparing the SHAP sum with the model prediction.

    Parameters:
    - explainer: SHAP explainer object
    - model: Trained model
    - instance: Instance to explain

    Returns:
    - fidelity_score: Fidelity score (absolute difference between model prediction and SHAP prediction)
    """
    # Calculate SHAP values for the instance
    shap_values = explainer(instance)

    # Model prediction for the instance
    model_prediction = model.predict(instance)[0]

    # SHAP prediction (SHAP values sum + base value)
    shap_prediction = shap_values.base_values[0] + np.sum(shap_values.values[0])

    # Fidelity score (should be close to zero if high fidelity)
    fidelity_score = np.abs(model_prediction - shap_prediction)

    # Print fidelity score
    print(f"Model Prediction: {model_prediction:.3f}, SHAP Prediction: {shap_prediction:.3f}")
    print(f"Fidelity Score (Difference): {fidelity_score:.5f}")

    return fidelity_score

# Example usage
if __name__ == "__main__":
    # Create sample regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Scale features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model and create SHAP explainer
    model, explainer = train_model_and_explainer(X_scaled, y)

    # Select a sample instance to explain
    instance = X_scaled[0].reshape(1, -1)

    # Test SHAP fidelity
    fidelity_score = test_shap_fidelity(explainer, model, instance)