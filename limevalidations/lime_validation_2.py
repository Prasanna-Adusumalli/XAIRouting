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
    return explanation

def compute_jaccard_similarity(set1, set2):
    """
    Compute the Jaccard Similarity between two sets.

    Parameters:
    - set1: First set of features
    - set2: Second set of features

    Returns:
    - Jaccard Similarity score
    """
    intersection = len(set(set1) & set(set2))
    union = len(set(set1) | set(set2))
    return intersection / union

def test_lime_stability(explainer, model, instance, perturbation_scale=0.01, num_features=5, num_perturbations=10):
    """
    Test the stability of LIME explanations by perturbing the input instance multiple times.

    Parameters:
    - explainer: LIME explainer object
    - model: Trained model
    - instance: Original instance
    - perturbation_scale: Scale of the random noise to add for perturbation
    - num_features: Number of top features to display
    - num_perturbations: Number of perturbations to perform

    Returns:
    - stability_scores: List of Jaccard similarity scores for each perturbation
    """
    # Generate explanation for the original instance
    explanation_original = generate_lime_explanation(explainer, model, instance, num_features)
    top_features_original = [feature[0] for feature in explanation_original.as_list()]

    stability_scores = []

    for _ in range(num_perturbations):
        # Perturb the instance slightly
        perturbed_instance = instance + np.random.normal(0, perturbation_scale, instance.shape)

        # Generate explanation for the perturbed instance
        explanation_perturbed = generate_lime_explanation(explainer, model, perturbed_instance, num_features)
        top_features_perturbed = [feature[0] for feature in explanation_perturbed.as_list()]

        # Compute Jaccard similarity between original and perturbed features
        jaccard_similarity = compute_jaccard_similarity(top_features_original, top_features_perturbed)
        stability_scores.append(jaccard_similarity)

    return stability_scores

def plot_jaccard_similarity(stability_scores):
    """
    Plot a bar chart of Jaccard Similarity scores for each perturbation.

    Parameters:
    - stability_scores: List of Jaccard Similarity scores for each perturbation.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(stability_scores)), stability_scores, color='skyblue')
    plt.xlabel('Perturbation Number')
    plt.ylabel('Jaccard Similarity Score')
    plt.title('Jaccard Similarity Scores of LIME Explanations Across Perturbations')
    plt.ylim(0, 1)  # Jaccard Similarity ranges from 0 to 1
    plt.axhline(y=np.mean(stability_scores), color='r', linestyle='--', label=f'Average: {np.mean(stability_scores):.2f}')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Train model and create explainer
    model, explainer = train_model_and_explainer(X, y)

    # Select a sample instance to explain
    instance = X[0].reshape(1, -1)

    # Test LIME stability and get Jaccard similarity scores
    stability_scores = test_lime_stability(explainer, model, instance)

    # Plot the Jaccard Similarity Scores
    plot_jaccard_similarity(stability_scores)