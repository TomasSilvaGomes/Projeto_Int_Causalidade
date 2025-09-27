"""
Continuity Metric Implementation
Measures how stable attribution methods are to small perturbations in the input.
Higher continuity (lower variance) indicates more stable and reliable attributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from interpretability_methods import InterpretabilityMethods


class ContinuityEvaluator:
    def __init__(self, model_methods=None):
        """Initialize with interpretability methods"""
        if model_methods is None:
            self.interp = InterpretabilityMethods()
        else:
            self.interp = model_methods
    
    def compute_continuity(self, input_data, attribution_method, target_class=None,
                          noise_levels=[0.02, 0.05], n_perturbations=5):  # OPTIMIZED - fewer levels & perturbations
        """
        Compute continuity by measuring attribution stability under input perturbations
        
        Args:
            input_data: Original input samples [batch_size, features]
            attribution_method: Method function to compute attributions
            target_class: Target classes for evaluation
            noise_levels: Standard deviations of Gaussian noise to add
            n_perturbations: Number of perturbed versions per noise level
            
        Returns:
            continuity_scores: Variance of attributions across perturbations
        """
        batch_size = input_data.shape[0]
        continuity_results = {noise: [] for noise in noise_levels}
        
        for sample_idx in range(batch_size):
            sample_input = input_data[sample_idx:sample_idx+1]
            sample_target = target_class[sample_idx] if target_class is not None else None
            
            # Get original attribution
            original_attribution = attribution_method(sample_input, sample_target)
            
            for noise_level in noise_levels:
                attribution_variations = []
                
                # Generate perturbations and compute attributions
                for _ in range(n_perturbations):
                    # Add Gaussian noise
                    noise = np.random.normal(0, noise_level, sample_input.shape)
                    perturbed_input = sample_input + noise
                    
                    # Clip to valid range [0, 1]
                    perturbed_input = np.clip(perturbed_input, 0, 1)
                    
                    # Compute attribution for perturbed input
                    perturbed_attribution = attribution_method(perturbed_input, sample_target)
                    attribution_variations.append(perturbed_attribution[0])  # Remove batch dimension
                
                # Calculate variance across perturbations
                attribution_variations = np.array(attribution_variations)
                attribution_variance = np.var(attribution_variations, axis=0)
                
                # Use mean variance as continuity score (lower = better continuity)
                continuity_score = np.mean(attribution_variance)
                continuity_results[noise_level].append(continuity_score)
        
        # Average across all samples
        avg_continuity = {noise: np.mean(scores) for noise, scores in continuity_results.items()}
        
        return avg_continuity, continuity_results
    
    def evaluate_all_methods(self, n_samples=30, random_state=42):
        """
        Evaluate continuity for all attribution methods
        Note: Using fewer samples than selectivity as this is computationally intensive
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
        
        print("\nEvaluating continuity for each method:")
        for method_name, method_func in attribution_methods.items():
            try:
                print(f"  Evaluating {method_name}...")
                avg_continuity, individual_scores = self.compute_continuity(
                    sample_data['images_flat'],
                    method_func,
                    target_class=None,  # Use predicted classes
                    noise_levels=[0.01, 0.02, 0.05, 0.1],
                    n_perturbations=5  # Reduced for faster computation
                )
                results[method_name] = {
                    'avg_continuity': avg_continuity,
                    'individual_scores': individual_scores
                }
                
                # Print summary (lower is better for continuity)
                final_continuity = avg_continuity[0.1]  # At highest noise level
                print(f"    Continuity score (noise=0.1): {final_continuity:.6f}")
                
            except Exception as e:
                print(f"    Error evaluating {method_name}: {str(e)}")
                continue
        
        return results
    
    def plot_continuity_curves(self, results, figsize=(12, 8)):
        """
        Plot continuity curves for all methods (lower is better)
        """
        plt.figure(figsize=figsize)
        
        noise_levels = list(results[list(results.keys())[0]]['avg_continuity'].keys())
        
        for method_name, result in results.items():
            continuity_values = [result['avg_continuity'][noise] for noise in noise_levels]
            plt.plot(noise_levels, continuity_values, marker='o', label=method_name, linewidth=2)
        
        plt.xlabel('Noise Level (Standard Deviation)')
        plt.ylabel('Attribution Variance (Continuity Score)')
        plt.title('Continuity Evaluation: Attribution Method Stability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        # Add interpretation text
        plt.text(0.02, 0.98, 'Lower curves = Better attribution methods\n(More stable under input perturbations)', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_continuity_comparison(self, results, noise_level=0.05, figsize=(10, 6)):
        """
        Create bar plot comparing continuity scores at specific noise level
        """
        methods = []
        continuity_scores = []
        
        for method_name, result in results.items():
            methods.append(method_name.replace('_', ' ').title())
            continuity_scores.append(result['avg_continuity'][noise_level])
        
        plt.figure(figsize=figsize)
        bars = plt.bar(methods, continuity_scores, color='lightcoral', alpha=0.7, edgecolor='darkred')
        
        # Add value labels on bars
        for bar, score in zip(bars, continuity_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Attribution Method')
        plt.ylabel(f'Continuity Score (noise={noise_level})')
        plt.title(f'Continuity Comparison at Noise Level {noise_level}\n(Lower is Better)')
        plt.xticks(rotation=45, ha='right')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_continuity_heatmap(self, results, figsize=(10, 6)):
        """
        Create heatmap showing continuity across all methods and noise levels
        """
        methods = list(results.keys())
        noise_levels = list(results[methods[0]]['avg_continuity'].keys())
        
        # Create matrix for heatmap
        continuity_matrix = []
        for method in methods:
            row = [results[method]['avg_continuity'][noise] for noise in noise_levels]
            continuity_matrix.append(row)
        
        continuity_matrix = np.array(continuity_matrix)
        
        plt.figure(figsize=figsize)
        
        # Use log scale for better visualization
        log_matrix = np.log10(continuity_matrix + 1e-10)  # Add small value to avoid log(0)
        
        sns.heatmap(log_matrix, 
                   xticklabels=[f'{noise:.2f}' for noise in noise_levels],
                   yticklabels=[method.replace('_', ' ').title() for method in methods],
                   annot=True, fmt='.2f', cmap='RdYlBu_r', cbar_kws={'label': 'log10(Continuity Score)'})
        
        plt.xlabel('Noise Level')
        plt.ylabel('Attribution Method')
        plt.title('Continuity Heatmap (Log Scale)\nBlue = Better (Lower variance)')
        plt.tight_layout()
        plt.show()
    
    def analyze_noise_sensitivity(self, results, method_name='vanilla_gradients', figsize=(12, 5)):
        """
        Analyze how a specific method's continuity changes with noise level
        """
        if method_name not in results:
            print(f"Method {method_name} not found in results")
            return
        
        individual_scores = results[method_name]['individual_scores']
        avg_scores = results[method_name]['avg_continuity']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Distribution at different noise levels
        noise_levels = list(individual_scores.keys())
        for noise in noise_levels:
            scores = individual_scores[noise]
            axes[0].hist(scores, alpha=0.6, label=f'noise={noise}', bins=15)
        
        axes[0].set_xlabel('Continuity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Continuity Distribution - {method_name.replace("_", " ").title()}')
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Average continuity vs noise level
        noise_vals = list(avg_scores.keys())
        continuity_vals = list(avg_scores.values())
        
        axes[1].plot(noise_vals, continuity_vals, 'o-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('Average Continuity Score')
        axes[1].set_title(f'Noise Sensitivity - {method_name.replace("_", " ").title()}')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results):
        """
        Generate a comprehensive continuity evaluation report
        """
        print("="*60)
        print("CONTINUITY EVALUATION REPORT")
        print("="*60)
        
        print("\nCONTINUITY METRIC EXPLANATION:")
        print("Continuity measures how stable attribution methods are to small")
        print("perturbations in the input. Lower variance = better continuity.")
        print("More stable methods are more reliable for interpretation.")
        
        print("\nRESULTS SUMMARY:")
        print("-" * 40)
        
        # Rank methods by continuity at medium noise level (lower is better)
        noise_level = 0.05
        method_scores = []
        for method_name, result in results.items():
            continuity_score = result['avg_continuity'][noise_level]
            method_scores.append((method_name, continuity_score))
        
        method_scores.sort(key=lambda x: x[1])  # Sort ascending (lower is better)
        
        print(f"{'Rank':<4} {'Method':<20} {'Continuity Score':<15} {'Quality':<10}")
        print("-" * 55)
        
        for rank, (method, score) in enumerate(method_scores, 1):
            if score < 0.001:
                quality = "Excellent"
            elif score < 0.01:
                quality = "Good"
            elif score < 0.1:
                quality = "Fair"
            else:
                quality = "Poor"
                
            print(f"{rank:<4} {method.replace('_', ' ').title():<20} {score:.6f}{'':>9} {quality:<10}")
        
        print(f"\nMost stable method: {method_scores[0][0].replace('_', ' ').title()}")
        print(f"Least stable method: {method_scores[-1][0].replace('_', ' ').title()}")
        
        # Detailed analysis
        print("\nDETAILED ANALYSIS:")
        print("-" * 40)
        
        for method_name, result in results.items():
            print(f"\n{method_name.replace('_', ' ').title()}:")
            avg_cont = result['avg_continuity']
            print(f"  - Continuity at noise=0.01: {avg_cont[0.01]:.6f}")
            print(f"  - Continuity at noise=0.05: {avg_cont[0.05]:.6f}")
            print(f"  - Continuity at noise=0.10: {avg_cont[0.1]:.6f}")
            
            # Calculate noise sensitivity (how much continuity degrades with noise)
            sensitivity = avg_cont[0.1] / avg_cont[0.01]
            print(f"  - Noise sensitivity ratio: {sensitivity:.2f}x")
        
        print("\nINTERPRETATION GUIDELINES:")
        print("- Continuity < 0.001: Very stable, highly reliable")
        print("- Continuity 0.001-0.01: Stable, reliable for most uses")
        print("- Continuity 0.01-0.1: Moderately stable, use with caution")
        print("- Continuity > 0.1: Unstable, may not be reliable")


def main():
    """Main evaluation function"""
    print("Starting Continuity Evaluation...")
    print("Note: This evaluation is computationally intensive and may take a few minutes.")
    
    # Initialize evaluator
    evaluator = ContinuityEvaluator()
    
    # Run evaluation with smaller sample size for computational efficiency
    results = evaluator.evaluate_all_methods(n_samples=20)
    
    if not results:
        print("No results obtained. Please check for errors.")
        return None
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_continuity_curves(results)
    evaluator.plot_continuity_comparison(results, noise_level=0.05)
    evaluator.plot_continuity_heatmap(results)
    
    # Analyze most stable method in detail
    best_method = min(results.keys(), 
                     key=lambda x: results[x]['avg_continuity'][0.05])
    evaluator.analyze_noise_sensitivity(results, best_method)
    
    # Generate report
    evaluator.generate_report(results)
    
    print("\nContinuity evaluation complete!")
    return results


if __name__ == "__main__":
    results = main()
