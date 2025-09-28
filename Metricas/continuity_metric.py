"""
Continuity Metric Implementation
Measures how stable attribution methods are to small perturbations in the input.
Higher continuity (lower variance) indicates more stable and reliable attributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import hashlib
import os
from Metodos_interpretabilidade.interpretability_methods import InterpretabilityMethods


class ContinuityEvaluator:
    def __init__(self, model_methods=None):
        """Initialize with interpretability methods"""
        if model_methods is None:
            self.interp = InterpretabilityMethods()
        else:
            self.interp = model_methods
        
        # Caching system for continuity results
        self.cache_dir = './cache'
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_continuity_cache_key(self, method_name, input_data, noise_levels, n_perturbations):
        """Generate cache key for continuity evaluation"""
        data_hash = hashlib.md5(input_data.tobytes()).hexdigest()[:8]
        params_hash = hashlib.md5(f"{noise_levels}_{n_perturbations}".encode()).hexdigest()[:4]
        return f"continuity_{method_name}_{data_hash}_{params_hash}.pkl"
    
    def _save_continuity_cache(self, cache_key, data):
        """Save continuity results to cache"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save continuity to cache: {e}")
    
    def _load_continuity_cache(self, cache_key):
        """Load continuity results from cache"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load continuity from cache: {e}")
        return None
    
    def compute_continuity(self, input_data, attribution_method, target_class=None,
                          noise_levels=[0.02, 0.05], n_perturbations=3):
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
        
        # Get original attributions for all samples in one batch call
        print(f"    Computing original attributions for {batch_size} samples...")
        try:
            original_attributions = attribution_method(input_data, target_class)
        except Exception as e:
            print(f"    Error computing original attributions: {e}")
            return {}, {}
        
        for noise_level in noise_levels:
            print(f"    Processing noise level {noise_level}...")
            
            # Create all perturbed versions at once for efficient batch processing
            all_perturbed_inputs = []
            sample_indices = []  # Track which sample each perturbed input belongs to
            
            # Generate all perturbations for all samples
            for sample_idx in range(batch_size):
                sample_input = input_data[sample_idx:sample_idx+1]
                
                for perturbation_idx in range(n_perturbations):
                    # Add Gaussian noise
                    noise = np.random.normal(0, noise_level, sample_input.shape)
                    perturbed_input = sample_input + noise
                    
                    # Clip to valid range [0, 1]
                    perturbed_input = np.clip(perturbed_input, 0, 1)
                    
                    all_perturbed_inputs.append(perturbed_input[0])  # Remove batch dimension
                    sample_indices.append(sample_idx)
            
            # Convert to batch format for efficient processing
            all_perturbed_batch = np.array(all_perturbed_inputs)
            
            # Create target classes for all perturbed samples
            if target_class is not None:
                perturbed_target_classes = []
                for sample_idx in sample_indices:
                    perturbed_target_classes.append(target_class[sample_idx])
                perturbed_target_classes = np.array(perturbed_target_classes)
            else:
                perturbed_target_classes = None
            
            print(f"      Computing attributions for {len(all_perturbed_batch)} perturbed samples in batch...")
            
            # CRITICAL: Compute all perturbed attributions in ONE batch call
            all_perturbed_attributions = attribution_method(all_perturbed_batch, perturbed_target_classes)
            
            # Reorganize results by original sample
            attribution_idx = 0
            for sample_idx in range(batch_size):
                attribution_variations = []
                
                # Collect all perturbations for this sample
                for perturbation_idx in range(n_perturbations):
                    attribution_variations.append(all_perturbed_attributions[attribution_idx])
                    attribution_idx += 1
                
                # Calculate variance across perturbations for this sample
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
                # Check cache first
                cache_key = self._generate_continuity_cache_key(
                    method_name, 
                    sample_data['images_flat'], 
                    [0.02, 0.05],  # Default noise levels
                    5  # Default perturbations
                )
                cached_result = self._load_continuity_cache(cache_key)
                
                if cached_result is not None:
                    print(f"  ✓ {method_name} continuity loaded from cache")
                    results[method_name] = cached_result
                    continue
                
                print(f"  Evaluating {method_name}...")
                avg_continuity, individual_scores = self.compute_continuity(
                    sample_data['images_flat'],
                    method_func,
                    target_class=None,  # Use predicted classes
                    noise_levels=[0.05],  # Single noise level for speed
                    n_perturbations=3  # Minimal perturbations for MNIST
                )
                result_data = {
                    'avg_continuity': avg_continuity,
                    'individual_scores': individual_scores
                }
                results[method_name] = result_data
                
                # Save to cache
                self._save_continuity_cache(cache_key, result_data)
                
                # Print summary (lower is better for continuity)
                final_continuity = avg_continuity[0.05]  # At highest noise level
                print(f"    Continuity score (noise=0.05): {final_continuity:.6f}")
                
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
