"""
Selectivity Metric Implementation
Measures how much the model's output changes when the most important features are removed.
Higher selectivity indicates that the attribution method correctly identifies important features.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Metodos_interpretabilidade.interpretability_methods import InterpretabilityMethods


class SelectivityEvaluator:
    def __init__(self, model_methods=None):
        """Initialize with interpretability methods"""
        if model_methods is None:
            self.interp = InterpretabilityMethods()
        else:
            self.interp = model_methods
    
    def compute_selectivity(self, input_data, attribution, target_class=None, 
                          percentiles=[10, 20, 50, 80, 90, 95]):
        """
        Compute selectivity by progressively removing most important features
        
        Args:
            input_data: Original input samples [batch_size, features]
            attribution: Attribution scores [batch_size, features]  
            target_class: Target classes for evaluation
            percentiles: Percentages of features to remove
            
        Returns:
            selectivity_scores: Score differences for each percentile
        """
        batch_size = input_data.shape[0]
        selectivity_scores = {p: [] for p in percentiles}
        
        # Get original predictions (single batch call)
        original_preds = self.interp.model.predict(input_data, verbose=0)
        
        if target_class is None:
            target_class = np.argmax(original_preds, axis=1)
        
        # Get original confidence scores for target classes (vectorized)
        original_scores = original_preds[np.arange(batch_size), target_class]
        
        # Process all percentiles for all samples in batch
        for percentile in percentiles:
            # Create batch of modified inputs for this percentile
            modified_batch = input_data.copy()
            
            for sample_idx in range(batch_size):
                sample_attribution = attribution[sample_idx]
                
                # Get indices of features sorted by attribution importance (descending)
                importance_indices = np.argsort(np.abs(sample_attribution))[::-1]
                
                # Calculate number of features to remove
                n_features_to_remove = int(len(sample_attribution) * percentile / 100)
                features_to_remove = importance_indices[:n_features_to_remove]
                
                # Set most important features to baseline (0) in the batch
                modified_batch[sample_idx, features_to_remove] = 0.0
            
            # Single batch prediction for all modified samples
            modified_preds = self.interp.model.predict(modified_batch, verbose=0)
            modified_scores = modified_preds[np.arange(batch_size), target_class]
            
            # Calculate selectivity for all samples (vectorized)
            selectivity_batch = original_scores - modified_scores
            selectivity_scores[percentile] = selectivity_batch.tolist()
        
        # Average across all samples
        avg_selectivity = {p: np.mean(scores) for p, scores in selectivity_scores.items()}
        
        return avg_selectivity, selectivity_scores
    
    def evaluate_all_methods(self, n_samples=50, random_state=42):
        """
        Evaluate selectivity for all attribution methods
        """
        print("Getting sample data...")
        sample_data = self.interp.get_sample_data(n_samples=n_samples, random_state=random_state)
        
        print("Computing attributions for all methods...")
        attributions = self.interp.get_all_attributions(sample_data['images_flat'])
        
        results = {}
        
        print("\nEvaluating selectivity for each method:")
        for method_name, attribution in attributions.items():
            if attribution is not None:
                print(f"  Evaluating {method_name}...")
                avg_selectivity, individual_scores = self.compute_selectivity(
                    sample_data['images_flat'], 
                    attribution,
                    target_class=None  # Use predicted classes
                )
                results[method_name] = {
                    'avg_selectivity': avg_selectivity,
                    'individual_scores': individual_scores
                }
                
                # Print summary
                final_selectivity = avg_selectivity[90]  # At 90% removal
                print(f"    Final selectivity (90% removal): {final_selectivity:.4f}")
        
        return results
    
    def plot_selectivity_curves(self, results, figsize=(12, 8)):
        """
        Plot selectivity curves for all methods
        """
        plt.figure(figsize=figsize)
        
        percentiles = list(results[list(results.keys())[0]]['avg_selectivity'].keys())
        
        for method_name, result in results.items():
            selectivity_values = [result['avg_selectivity'][p] for p in percentiles]
            plt.plot(percentiles, selectivity_values, marker='o', label=method_name, linewidth=2)
        
        plt.xlabel('Percentage of Features Removed (%)')
        plt.ylabel('Average Selectivity (Confidence Drop)')
        plt.title('Selectivity Evaluation: Attribution Method Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        
        # Add interpretation text
        plt.text(0.02, 0.98, 'Higher curves = Better attribution methods\n(Removing important features causes larger drops in confidence)', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results):
        """
        Generate a comprehensive selectivity evaluation report
        """
        print("="*60)
        print("SELECTIVITY EVALUATION REPORT")
        print("="*60)
        
        print("\nSELECTIVITY METRIC EXPLANATION:")
        print("Selectivity measures how much the model's confidence drops when")
        print("the most important features (according to the attribution method) are removed.")
        print("Higher selectivity = better attribution method (identifies truly important features)")
        
        print("\nRESULTS SUMMARY:")
        print("-" * 40)
        
        # Rank methods by final selectivity
        method_scores = []
        for method_name, result in results.items():
            final_score = result['avg_selectivity'][90]
            method_scores.append((method_name, final_score))
        
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<4} {'Method':<20} {'Final Selectivity':<15} {'Quality':<10}")
        print("-" * 55)
        
        for rank, (method, score) in enumerate(method_scores, 1):
            if score > 0.3:
                quality = "Excellent"
            elif score > 0.2:
                quality = "Good"
            elif score > 0.1:
                quality = "Fair"
            else:
                quality = "Poor"
                
            print(f"{rank:<4} {method.replace('_', ' ').title():<20} {score:.4f}{'':>11} {quality:<10}")
        
        print(f"\nBest performing method: {method_scores[0][0].replace('_', ' ').title()}")
        print(f"Worst performing method: {method_scores[-1][0].replace('_', ' ').title()}")


def main():
    """Main evaluation function"""
    print("Starting Selectivity Evaluation...")
    
    # Initialize evaluator
    evaluator = SelectivityEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_all_methods(n_samples=50)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_selectivity_curves(results)
    
    # Generate report
    evaluator.generate_report(results)
    
    print("\nSelectivity evaluation complete!")
    return results


if __name__ == "__main__":
    results = main()
