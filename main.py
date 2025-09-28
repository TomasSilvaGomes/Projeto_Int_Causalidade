"""
Comprehensive Interpretability Evaluation Suite
Runs all three metrics (Selectivity, Continuity, ROAD) and provides comparative analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Metodos_interpretabilidade.interpretability_methods import InterpretabilityMethods
from Metricas.selectivity_metric import SelectivityEvaluator
from Metricas.continuity_metric import ContinuityEvaluator
from Metricas.road_metric import ROADEvaluator
import time


class ComprehensiveEvaluator:
    def __init__(self):
        """Initialize with shared interpretability methods"""
        print("Initializing Comprehensive Interpretability Evaluator...")
        self.interp = InterpretabilityMethods()
        
        # Initialize metric evaluators with shared methods
        self.selectivity_eval = SelectivityEvaluator(self.interp)
        self.continuity_eval = ContinuityEvaluator(self.interp)
        self.road_eval = ROADEvaluator(self.interp)
    
    def run_all_evaluations(self, selectivity_samples=100, continuity_samples=30, road_samples=10):
        """
        Run all three metric evaluations
        
        Args:
            selectivity_samples: Number of samples for selectivity evaluation
            continuity_samples: Number of samples for continuity evaluation  
            road_samples: Number of samples for ROAD evaluation
        """
        results = {}
        
        print("="*70)
        print("COMPREHENSIVE INTERPRETABILITY EVALUATION")
        print("="*70)
        
        # 1. Selectivity Evaluation
        print("\n1. SELECTIVITY EVALUATION")
        print("-" * 30)
        start_time = time.time()
        
        try:
            selectivity_results = self.selectivity_eval.evaluate_all_methods(n_samples=selectivity_samples)
            results['selectivity'] = selectivity_results
            print("✓ Selectivity evaluation completed")
        except Exception as e:
            print(f"✗ Selectivity evaluation failed: {str(e)}")
            results['selectivity'] = None
        
        # 2. Continuity Evaluation
        print("\n2. CONTINUITY EVALUATION")
        print("-" * 30)
        start_time = time.time()
        
        try:
            continuity_results = self.continuity_eval.evaluate_all_methods(n_samples=continuity_samples)
            results['continuity'] = continuity_results
            print("✓ Continuity evaluation completed")
        except Exception as e:
            print(f"✗ Continuity evaluation failed: {str(e)}")
            results['continuity'] = None
        
        # 3. ROAD Evaluation
        print("\n3. ROAD EVALUATION")
        print("-" * 30)
        start_time = time.time()
        
        try:
            road_results = self.road_eval.evaluate_all_methods(n_samples=road_samples)
            results['road'] = road_results
            print("✓ ROAD evaluation completed")
        except Exception as e:
            print(f"✗ ROAD evaluation failed: {str(e)}")
            results['road'] = None
        
        return results
    
    def create_comprehensive_comparison(self, results, figsize=(15, 10)):
        """
        Create comprehensive comparison visualization across all metrics
        """
        # Extract method names (common across all metrics)
        if results['selectivity']:
            methods = list(results['selectivity'].keys())
        elif results['continuity']:
            methods = list(results['continuity'].keys())
        elif results['road']:
            methods = list(results['road'].keys())
        else:
            print("No valid results to visualize")
            return
        
        # Prepare data for comparison
        comparison_data = []
        
        for method in methods:
            method_data = {'Method': method.replace('_', ' ').title()}
            
            # Selectivity (higher is better)
            if results['selectivity'] and method in results['selectivity']:
                selectivity_score = results['selectivity'][method]['avg_selectivity'][90]
                method_data['Selectivity'] = selectivity_score
            else:
                method_data['Selectivity'] = np.nan
            
            # Continuity (lower is better, so we invert for visualization)
            if results['continuity'] and method in results['continuity']:
                continuity_score = results['continuity'][method]['avg_continuity'][0.05]
                # Invert and normalize for comparison (smaller values become larger)
                method_data['Continuity'] = 1 / (1 + continuity_score)
            else:
                method_data['Continuity'] = np.nan
            
            # ROAD (higher is better)
            if results['road'] and method in results['road']:
                road_scores = list(results['road'][method].values())
                road_score = np.mean(road_scores)
                method_data['ROAD'] = road_score
            else:
                method_data['ROAD'] = np.nan
            
            comparison_data.append(method_data)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        df = df.set_index('Method')
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Radar chart (normalized scores)
        ax1 = axes[0, 0]
        df_normalized = df.copy()
        for col in df_normalized.columns:
            if not df_normalized[col].isna().all():
                col_min, col_max = df_normalized[col].min(), df_normalized[col].max()
                if col_max > col_min:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        
        angles = np.linspace(0, 2*np.pi, len(df_normalized.columns), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        for idx, method in enumerate(df_normalized.index):
            values = df_normalized.loc[method].values
            if not np.isnan(values).all():
                values = np.concatenate((values, [values[0]]))
                ax1.plot(angles, values, 'o-', linewidth=2, label=method)
                ax1.fill(angles, values, alpha=0.1)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(df_normalized.columns)
        ax1.set_title('Normalized Performance Radar')
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 2. Heatmap
        ax2 = axes[0, 1]
        sns.heatmap(df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax2, cbar_kws={'label': 'Performance Score'})
        ax2.set_title('Performance Heatmap')
        ax2.set_xlabel('Attribution Method')
        
        # 3. Bar chart comparison
        ax3 = axes[1, 0]
        df.plot(kind='bar', ax=ax3, alpha=0.7)
        ax3.set_title('Metric Comparison by Method')
        ax3.set_ylabel('Score')
        ax3.legend(title='Metrics')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Overall ranking
        ax4 = axes[1, 1]
        
        # Calculate overall score (normalized average)
        overall_scores = df_normalized.mean(axis=1, skipna=True).sort_values(ascending=False)
        
        bars = ax4.bar(range(len(overall_scores)), overall_scores.values, 
                      color='lightblue', alpha=0.7, edgecolor='navy')
        ax4.set_xticks(range(len(overall_scores)))
        ax4.set_xticklabels(overall_scores.index, rotation=45, ha='right')
        ax4.set_title('Overall Ranking (Normalized Average)')
        ax4.set_ylabel('Overall Score')
        
        # Add value labels on bars
        for bar, score in zip(bars, overall_scores.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return df, overall_scores
    
    def create_results_table(self, results):
        """
        Create a clean results table showing methods vs metrics
        """
        # Extract method names
        if results['selectivity']:
            methods = list(results['selectivity'].keys())
        elif results['continuity']:
            methods = list(results['continuity'].keys())
        elif results['road']:
            methods = list(results['road'].keys())
        else:
            return None
        
        # Create table data
        table_data = []
        
        for method in methods:
            row = {'Method': method.replace('_', ' ').title()}
            
            # Selectivity (higher is better)
            if results['selectivity'] and method in results['selectivity']:
                selectivity_score = results['selectivity'][method]['avg_selectivity'][90]
                row['Selectivity'] = f"{selectivity_score:.4f}"
            else:
                row['Selectivity'] = "N/A"
            
            # Continuity (lower is better)
            if results['continuity'] and method in results['continuity']:
                continuity_score = results['continuity'][method]['avg_continuity'][0.05]
                row['Continuity'] = f"{continuity_score:.6f}"
            else:
                row['Continuity'] = "N/A"
            
            # ROAD (higher is better)
            if results['road'] and method in results['road']:
                road_scores = list(results['road'][method].values())
                road_avg = np.mean(road_scores)
                row['ROAD'] = f"{road_avg:.4f}"
            else:
                row['ROAD'] = "N/A"
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Print formatted table
        print("\n" + "="*80)
        print("RESULTS TABLE - INTERPRETABILITY METHODS vs METRICS")
        print("="*80)
        print("\nNote: Selectivity & ROAD - Higher is Better | Continuity - Lower is Better")
        print("-" * 80)
        print(df.to_string(index=False, justify='center'))
        print("-" * 80)
        
        # Save table as CSV for easy use in reports
        df.to_csv('interpretability_results_table.csv', index=False)
        print("📋 Results table saved as: interpretability_results_table.csv")
        
        return df

    def generate_comprehensive_report(self, results):
        """
        Generate a comprehensive report combining all three metrics
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE INTERPRETABILITY EVALUATION REPORT")
        print("="*80)
        
        print("\nEVALUATION SUMMARY:")
        print("-" * 50)
        
        # Check which evaluations completed successfully
        completed_metrics = []
        if results['selectivity']:
            completed_metrics.append("Selectivity")
        if results['continuity']:
            completed_metrics.append("Continuity")
        if results['road']:
            completed_metrics.append("ROAD")
        
        print(f"Completed evaluations: {', '.join(completed_metrics)}")
        
        if not completed_metrics:
            print("No evaluations completed successfully.")
            return
        
        # Extract common methods
        all_methods = set()
        for metric_name, metric_results in results.items():
            if metric_results:
                all_methods.update(metric_results.keys())
        
        print(f"Evaluated methods: {', '.join([m.replace('_', ' ').title() for m in all_methods])}")
        
        # Method-by-method analysis
        print("\nMETHOD ANALYSIS:")
        print("-" * 50)
        
        for method in all_methods:
            print(f"\n{method.replace('_', ' ').title()}:")
            
            # Selectivity
            if results['selectivity'] and method in results['selectivity']:
                sel_score = results['selectivity'][method]['avg_selectivity'][90]
                sel_quality = "Excellent" if sel_score > 0.3 else "Good" if sel_score > 0.2 else "Fair" if sel_score > 0.1 else "Poor"
                print(f"  • Selectivity: {sel_score:.4f} ({sel_quality})")
            
            # Continuity
            if results['continuity'] and method in results['continuity']:
                cont_score = results['continuity'][method]['avg_continuity'][0.05]
                cont_quality = "Excellent" if cont_score < 0.001 else "Good" if cont_score < 0.01 else "Fair" if cont_score < 0.1 else "Poor"
                print(f"  • Continuity: {cont_score:.6f} ({cont_quality})")
            
            # ROAD
            if results['road'] and method in results['road']:
                road_scores = list(results['road'][method].values())
                road_avg = np.mean(road_scores)
                road_quality = "Excellent" if road_avg > 0.5 else "Good" if road_avg > 0.2 else "Fair" if road_avg > 0 else "Poor"
                print(f"  • ROAD: {road_avg:.4f} ({road_quality})")
        
        # Overall recommendations
        print("\nOVERALL RECOMMENDATIONS:")
        print("-" * 50)
        
        print("Based on the multi-metric evaluation:")
        
        # Find best performer in each category
        if results['selectivity']:
            best_selectivity = max(results['selectivity'].keys(), 
                                 key=lambda x: results['selectivity'][x]['avg_selectivity'][90])
            print(f"• Best for feature importance: {best_selectivity.replace('_', ' ').title()}")
        
        if results['continuity']:
            best_continuity = min(results['continuity'].keys(), 
                                key=lambda x: results['continuity'][x]['avg_continuity'][0.05])
            print(f"• Most stable method: {best_continuity.replace('_', ' ').title()}")
        
        if results['road']:
            best_road = max(results['road'].keys(), 
                          key=lambda x: np.mean(list(results['road'][x].values())))
            print(f"• Most faithful method: {best_road.replace('_', ' ').title()}")
        
        print("\nUSE CASE RECOMMENDATIONS:")
        print("• For critical decisions: Use methods with high scores in all metrics")
        print("• For exploration: Selectivity is most important")
        print("• For production systems: Continuity and ROAD are crucial")
        print("• Avoid methods with consistently poor performance across metrics")
    
    def save_results(self, results, filename='interpretability_evaluation_results.npz'):
        """Save evaluation results for later analysis"""
        np.savez(filename, **results)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename='interpretability_evaluation_results.npz'):
        """Load previously saved evaluation results"""
        loaded = np.load(filename, allow_pickle=True)
        results = {}
        for key in loaded.files:
            results[key] = loaded[key].item() if loaded[key].shape == () else loaded[key]
        print(f"Results loaded from {filename}")
        return results


def main():
    """Main comprehensive evaluation function"""
    print("Starting Interpretability Evaluation...")
    selectivity_samples, continuity_samples, road_samples = 100, 50, 25
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Run all evaluations
    results = evaluator.run_all_evaluations(
        selectivity_samples=selectivity_samples,
        continuity_samples=continuity_samples,
        road_samples=road_samples
    )
    
    # Generate comprehensive comparison
    print("\nGenerating comprehensive comparison...")
    if any(results.values()):
        # Create and display results table
        results_table = evaluator.create_results_table(results)
        
        # Create comparison visualizations
        comparison_df, rankings = evaluator.create_comprehensive_comparison(results)
        
        # Generate comprehensive report
        evaluator.generate_comprehensive_report(results)
        
        # Save results
        evaluator.save_results(results)
        
        print("\nTop 3 Attribution Methods (Overall):")
        for i, (method, score) in enumerate(rankings.head(3).items(), 1):
            print(f"{i}. {method}: {score:.3f}")
    
    else:
        print("No evaluations completed successfully. Please check for errors.")
    
    print("\nComprehensive evaluation complete!")
    return results


if __name__ == "__main__":
    results = main()
