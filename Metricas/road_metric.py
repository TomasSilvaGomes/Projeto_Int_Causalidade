"""
ROAD (Remove and Debias) Metric Implementation
Measures the faithfulness of attribution methods by iteratively removing features
and retraining the model, then comparing how much the attribution changes.
Higher ROAD scores indicate more faithful attributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
from Metodos_interpretabilidade.interpretability_methods import InterpretabilityMethods


class ROADEvaluator:
    def __init__(self, model_methods=None):
        """Initialize with interpretability methods"""
        if model_methods is None:
            self.interp = InterpretabilityMethods()
        else:
            self.interp = model_methods
        
        self._retrained_models = {}  # Cache for retrained models
    
    def create_masked_dataset(self, X_data, y_data, important_features, mask_fraction=0.1):
        """
        Create a dataset with most important features masked (set to 0)
        """
        X_masked = X_data.copy()
        n_features_to_mask = int(len(important_features) * mask_fraction)
        features_to_mask = important_features[:n_features_to_mask]
        
        # Set important features to 0 (baseline value)
        X_masked[:, features_to_mask] = 0.0
        
        return X_masked
    
    def retrain_model_with_masked_data(self, mask_fraction=0.1, important_features=None, 
                                     epochs=2, cache_key=None):
        """
        Retrain model with important features masked in training data
        """
        if cache_key and cache_key in self._retrained_models:
            return self._retrained_models[cache_key]
        
        # If no important features provided, use random features (baseline)
        if important_features is None:
            important_features = np.random.permutation(784)  # Random for MNIST
        
        # Create masked training data
        X_train_masked = self.create_masked_dataset(
            self.interp.X_train, self.interp.y_train, 
            important_features, mask_fraction
        )
        X_train_flat_masked = X_train_masked.reshape(-1, 784)
        
        # Create new model with same architecture
        retrained_model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(32, activation='sigmoid'),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        retrained_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
        
        # Train on masked data
        print(f"  Retraining model with {mask_fraction*100:.0f}% features masked...")
        # Use only subset of training data for speed (still representative for MNIST)
        n_train_subset = min(30000, len(self.interp.y_train))  # Use max 30k samples
        subset_indices = np.random.choice(len(self.interp.y_train), n_train_subset, replace=False)
        
        retrained_model.fit(X_train_flat_masked[subset_indices], self.interp.y_train[subset_indices], 
                          epochs=epochs, batch_size=128, verbose=0, validation_split=0.05)
        
        # Cache the model
        if cache_key:
            self._retrained_models[cache_key] = retrained_model
        
        return retrained_model
    
    def compute_global_feature_importance(self, n_samples=100):
        """
        Compute global feature importance across multiple samples
        """
        print("  Computing global feature importance...")
        
        # Get random sample of training data
        indices = np.random.choice(len(self.interp.x_train_flat), n_samples, replace=False)
        sample_inputs = self.interp.x_train_flat[indices]
        
        # Compute gradients for all samples
        all_gradients = self.interp.vanilla_gradients(sample_inputs)
        
        # Average absolute gradients across all samples
        global_importance = np.mean(np.abs(all_gradients), axis=0)
        
        # Get feature indices sorted by importance (most important first)
        important_features = np.argsort(global_importance)[::-1]
        
        return important_features, global_importance
    
    def compute_road_score(self, input_data, attribution_method, target_class=None,
                          mask_fractions=[0.1, 0.2], retrain_epochs=3):
        """
        Compute ROAD score for an attribution method
        """
        # Get global feature importance
        global_important_features, _ = self.compute_global_feature_importance()
        
        road_scores = {}
        
        for mask_fraction in mask_fractions:
            print(f"  Evaluating ROAD at {mask_fraction*100:.0f}% masking...")
            
            # Retrain model with masked data
            cache_key = f"masked_{mask_fraction}"
            retrained_model = self.retrain_model_with_masked_data(
                mask_fraction=mask_fraction,
                important_features=global_important_features,
                epochs=retrain_epochs,
                cache_key=cache_key
            )
            
            # Compute attributions for original model
            original_attributions = attribution_method(input_data, target_class)
            
            # Temporarily replace model for attribution computation
            original_model = self.interp.model
            self.interp.model = retrained_model
            
            try:
                # Compute attributions for retrained model
                retrained_attributions = attribution_method(input_data, target_class)
                
                # Calculate ROAD score as correlation between attributions
                batch_correlations = []
                for i in range(len(input_data)):
                    orig_attr = original_attributions[i].flatten()
                    retrained_attr = retrained_attributions[i].flatten()
                    
                    # Compute correlation coefficient
                    correlation = np.corrcoef(orig_attr, retrained_attr)[0, 1]
                    if not np.isnan(correlation):
                        batch_correlations.append(correlation)
                
                road_score = np.mean(batch_correlations) if batch_correlations else 0.0
                road_scores[mask_fraction] = road_score
                
            finally:
                # Restore original model
                self.interp.model = original_model
        
        return road_scores
    
    def evaluate_all_methods(self, n_samples=10, random_state=42):
        """
        Evaluate ROAD scores for all attribution methods
        """
        print("Getting sample data...")
        sample_data = self.interp.get_sample_data(n_samples=n_samples, random_state=random_state)
        
        # Define attribution methods
        attribution_methods = {
            'lime': self.interp.lime_method,
            'shap': self.interp.shap_method,
            'gradcam': self.interp.gradcam_method,
            'integrated_gradients': self.interp.integrated_gradients_method,
            'saliency_maps': self.interp.saliency_maps
        }
        
        results = {}
        
        print("\nEvaluating ROAD for each method:")
        print("Warning: This is computationally intensive...")
        
        for method_name, method_func in attribution_methods.items():
            try:
                print(f"\nEvaluating {method_name}...")
                road_scores = self.compute_road_score(
                    sample_data['images_flat'],
                    method_func,
                    target_class=None,
                    mask_fractions=[0.1, 0.2],
                    retrain_epochs=2  # Very reduced for speed
                )
                results[method_name] = road_scores
                
                # Print summary
                avg_road = np.mean(list(road_scores.values()))
                print(f"  Average ROAD score: {avg_road:.4f}")
                
            except Exception as e:
                print(f"  Error evaluating {method_name}: {str(e)}")
                continue
        
        return results
    
    def generate_report(self, results):
        """
        Generate a comprehensive ROAD evaluation report
        """
        print("="*60)
        print("ROAD (REMOVE AND DEBIAS) EVALUATION REPORT")
        print("="*60)
        
        print("\nROAD METRIC EXPLANATION:")
        print("ROAD measures faithfulness by retraining models with important")
        print("features masked and comparing attribution consistency.")
        print("Higher ROAD scores (closer to 1) = more faithful attributions")
        
        print("\nRESULTS SUMMARY:")
        print("-" * 40)
        
        # Calculate average ROAD scores for ranking
        method_avg_scores = []
        for method_name, scores in results.items():
            avg_score = np.mean(list(scores.values()))
            method_avg_scores.append((method_name, avg_score))
        
        method_avg_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<4} {'Method':<20} {'Avg ROAD Score':<15} {'Quality':<12}")
        print("-" * 60)
        
        for rank, (method, avg_score) in enumerate(method_avg_scores, 1):
            if avg_score > 0.5:
                quality = "Excellent"
            elif avg_score > 0.2:
                quality = "Good"
            elif avg_score > 0:
                quality = "Fair"
            else:
                quality = "Poor"
                
            print(f"{rank:<4} {method.replace('_', ' ').title():<20} {avg_score:.4f}{'':>11} {quality:<12}")


def main():
    """Main evaluation function"""
    print("Starting ROAD Evaluation...")
    print("WARNING: This is computationally intensive!")
    
    # Initialize evaluator
    evaluator = ROADEvaluator()
    
    # Run evaluation with minimal samples
    results = evaluator.evaluate_all_methods(n_samples=5)
    
    if not results:
        print("No results obtained.")
        return None
    
    # Generate report
    evaluator.generate_report(results)
    
    print("\nROAD evaluation complete!")
    return results


if __name__ == "__main__":
    results = main()
