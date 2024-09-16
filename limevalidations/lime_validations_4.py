import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

def train_model_and_explainer(X, y):
    """
    Train a model and create a LIME explainer.

    Parameters:
    - X: Features for training
    - y: Target variable for training

    Returns:
    - model: Trained model
    - explainer: LIME explainer
    """
    # Train a RandomForest model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Create a LIME explainer
    explainer = LimeTabularExplainer(X, feature_names=[f'Feature_{i}' for i in range(X.shape[1])],
                                     mode='regression', verbose=True)
    return model, explainer

def generate_lime_explanation_with_fidelity(explainer, model, instance, num_features=5):
    """
    Generate LIME explanation for a given instance and return the fidelity score.

    Parameters:
    - explainer: LIME explainer object
    - model: Trained model
    - instance: Instance for which to generate explanation
    - num_features: Number of top features to display

    Returns:
    - explanation: LIME explanation object
    - fidelity_score: Fidelity score (R²) of the surrogate model
    """
    explanation = explainer.explain_instance(instance[0], model.predict, num_features=num_features)
    fidelity_score = explanation.score  # Extract R² score of the surrogate model
    return explanation, fidelity_score

def display_explanation_and_fidelity(explanation, fidelity_score):
    """
    Display the LIME explanation and print the fidelity score.

    Parameters:
    - explanation: LIME explanation object
    - fidelity_score: Fidelity score (R²) of the surrogate model
    """
    print(f"Fidelity Score (R²): {fidelity_score:.2f}")
    explanation.show_in_notebook()

# Example usage
if __name__ == "__main__":
    # Create sample regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Train model and create explainer
    model, explainer = train_model_and_explainer(X, y)

    # Select a sample instance to explain
    instance = X[0].reshape(1, -1)

    # Generate LIME explanation and get fidelity score
    explanation, fidelity_score = generate_lime_explanation_with_fidelity(explainer, model, instance)

    # Display explanation and fidelity score
    display_explanation_and_fidelity(explanation, fidelity_score)