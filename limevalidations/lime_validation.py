import numpy as np
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

def generate_lime_explanation(explainer, model, instance, num_features=5):
    """
    Generate LIME explanation for a given instance.

    Parameters:
    - explainer: LIME explainer object
    - model: Trained model
    - instance: Instance for which to generate explanation
    - num_features: Number of top features to display

    Returns:
    - explanation: LIME explanation object
    """
    explanation = explainer.explain_instance(instance[0], model.predict, num_features=num_features)
    explanation.show_in_notebook()
    return explanation

def test_lime_stability(explainer, model, instance, perturbation_scale=0.01, num_features=5):
    """
    Test the stability of LIME explanations by perturbing the input instance slightly.

    Parameters:
    - explainer: LIME explainer object
    - model: Trained model
    - instance: Original instance
    - perturbation_scale: Scale of the random noise to add for perturbation
    - num_features: Number of top features to display

    Returns:
    - None: Prints the comparison of original and perturbed explanations
    """
    # Generate explanation for the original instance
    explanation_original = generate_lime_explanation(explainer, model, instance, num_features)

    # Perturb the instance slightly
    perturbed_instance = instance + np.random.normal(0, perturbation_scale, instance.shape)

    # Generate explanation for the perturbed instance
    explanation_perturbed = generate_lime_explanation(explainer, model, perturbed_instance, num_features)

    # Compare top features between original and perturbed explanations
    top_features_original = [feature[0] for feature in explanation_original.as_list()]
    top_features_perturbed = [feature[0] for feature in explanation_perturbed.as_list()]

    print("Top Features (Original):", top_features_original)
    print("Top Features (Perturbed):", top_features_perturbed)

# Example usage (could be moved to a separate script or notebook)
if __name__ == "__main__":
    # Create sample regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Train model and create explainer
    model, explainer = train_model_and_explainer(X, y)

    # Select a sample instance to explain
    instance = X[0].reshape(1, -1)

    # Test LIME stability
    test_lime_stability(explainer, model, instance)