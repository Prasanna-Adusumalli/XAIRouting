import shap
import numpy as np
import matplotlib.pyplot as plt
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

def get_top_features(shap_values, num_features=5):
    """
    Get the top features based on absolute SHAP values.

    Parameters:
    - shap_values: SHAP values object
    - num_features: Number of top features to return

    Returns:
    - top_features: Set of indices of the top features
    """
    # Get SHAP values for the first instance (assuming batch size of 1)
    values = np.abs(shap_values.values[0])  # Absolute SHAP values
    top_features = np.argsort(-values)[:num_features]  # Indices of top features
    return set(top_features)

def calculate_jaccard_similarity(set1, set2):
    """
    Calculate Jaccard Similarity between two sets.

    Parameters:
    - set1: First set of elements
    - set2: Second set of elements

    Returns:
    - jaccard_similarity: Jaccard Similarity Index
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_similarity = intersection / union
    return jaccard_similarity

def test_shap_stability(explainer, model, instance, perturbation_scale=0.01, num_features=5):
    """
    Test the stability of SHAP explanations by perturbing the input instance slightly.

    Parameters:
    - explainer: SHAP explainer object
    - model: Trained model
    - instance: Original instance
    - perturbation_scale: Scale of the random noise to add for perturbation
    - num_features: Number of top features to consider for similarity

    Returns:
    - jaccard_scores: List of Jaccard Similarity scores for each perturbation
    """
    # Generate SHAP values for the original instance
    shap_values_original = explainer(instance)

    # Get top features for the original instance
    original_top_features = get_top_features(shap_values_original, num_features)

    # List to store Jaccard similarity scores
    jaccard_scores = []

    # Perturb the instance slightly and calculate SHAP values
    for i in range(5):  # Do 5 perturbations
        perturbed_instance = instance + np.random.normal(0, perturbation_scale, instance.shape)
        shap_values_perturbed = explainer(perturbed_instance)

        # Get top features for the perturbed instance
        perturbed_top_features = get_top_features(shap_values_perturbed, num_features)

        # Calculate Jaccard similarity
        jaccard_score = calculate_jaccard_similarity(original_top_features, perturbed_top_features)
        jaccard_scores.append(jaccard_score)

        print(f"Perturbation {i+1} - Jaccard Similarity: {jaccard_score:.2f}")

    return jaccard_scores

def plot_jaccard_similarity(jaccard_scores):
    """
    Plot the Jaccard Similarity scores using a bar plot.

    Parameters:
    - jaccard_scores: List of Jaccard Similarity scores

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(jaccard_scores) + 1), jaccard_scores, color='skyblue')
    plt.xlabel('Perturbation Number', fontsize=14)
    plt.ylabel('Jaccard Similarity', fontsize=14)
    plt.title('Jaccard Similarity of SHAP Explanations Across Perturbations', fontsize=16)
    plt.xticks(range(1, len(jaccard_scores) + 1))
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

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

    # Test SHAP stability and calculate Jaccard Similarity
    jaccard_scores = test_shap_stability(explainer, model, instance)

    # Plot Jaccard Similarity scores
    plot_jaccard_similarity(jaccard_scores)

    # Print the average Jaccard Similarity score
    print(f"Average Jaccard Similarity: {np.mean(jaccard_scores):.2f}")