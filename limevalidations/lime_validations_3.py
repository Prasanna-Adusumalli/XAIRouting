import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

def compute_feature_importance_distribution(explainer, model, instance, perturbation_scale=0.01, num_features=5, num_perturbations=10):
    """
    Compute the distribution of feature importances across multiple perturbations.

    Parameters:
    - explainer: LIME explainer object
    - model: Trained model
    - instance: Original instance
    - perturbation_scale: Scale of the random noise to add for perturbation
    - num_features: Number of top features to display
    - num_perturbations: Number of perturbations to perform

    Returns:
    - feature_importances: Pandas DataFrame containing feature importances across perturbations
    """
    feature_importances = []

    for _ in range(num_perturbations):
        # Perturb the instance slightly
        perturbed_instance = instance + np.random.normal(0, perturbation_scale, instance.shape)

        # Generate explanation for the perturbed instance
        explanation_perturbed = generate_lime_explanation(explainer, model, perturbed_instance, num_features)

        # Extract feature importances and append to list
        feature_importances.append({feature[0]: feature[1] for feature in explanation_perturbed.as_list()})

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(feature_importances).melt(var_name='Feature', value_name='Importance')
    return df

def plot_feature_importance_distribution(feature_importances_df):
    """
    Plot the distribution of feature importances across multiple perturbations.

    Parameters:
    - feature_importances_df: Pandas DataFrame containing feature importances across perturbations
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Feature', y='Importance', data=feature_importances_df, palette="Set3")
    plt.title('Distribution of Feature Importances Across Perturbations')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    # Train model and create explainer
    model, explainer = train_model_and_explainer(X, y)

    # Select a sample instance to explain
    instance = X[0].reshape(1, -1)

    # Compute the distribution of feature importances across perturbations
    feature_importances_df = compute_feature_importance_distribution(explainer, model, instance)

    # Plot the distribution of feature importances
    plot_feature_importance_distribution(feature_importances_df)