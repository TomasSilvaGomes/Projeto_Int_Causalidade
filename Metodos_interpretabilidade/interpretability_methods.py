"""
Interpretability Methods Implementation
Contains implementations of various attribution methods (GradCAM, Integrated Gradients, SHAP, etc.)
and provides the trained model and data for metric evaluation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import struct
import os
import pickle
import hashlib
import time
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge


class InterpretabilityMethods:
    def __init__(self, model_path='Rede/mnist_model.keras', weights_path='Rede/mnist_model.weights.h5'):
        """Initialize with trained model and data"""
        self.model_path = model_path
        self.weights_path = weights_path
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.x_train_flat = None
        self.x_test_flat = None
        
        # Caching system
        self.cache_dir = './cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load data and model
        self._load_data()
        self._load_model()
    
    def _load_data(self):
        """Load MNIST data from local files"""
        def load_images(filename):
            with open(filename, 'rb') as f:
                magic, num_images, rows, cols = struct.unpack('>4I', f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8)
                images = images.reshape(num_images, rows, cols)
            return images
        
        def load_labels(filename):
            with open(filename, 'rb') as f:
                magic, num_labels = struct.unpack('>2I', f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        
        path = './sample_data/MNIST/raw'
        
        # Load training data
        self.X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte'))
        self.y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte'))
        
        # Load test data
        self.X_test = load_images(os.path.join(path, 't10k-images-idx3-ubyte'))
        self.y_test = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte'))
        
        # Normalize pixel values to [0, 1]
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        # Flatten for dense network
        self.x_train_flat = self.X_train.reshape((len(self.X_train), 28 * 28))
        self.x_test_flat = self.X_test.reshape((len(self.X_test), 28 * 28))
        
        print(f"Data loaded: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
    
    def _generate_cache_key(self, method_name, input_data, target_class=None, **kwargs):
        """Generate unique cache key for attribution results"""
        # Create hash from input data and parameters
        data_hash = hashlib.md5(input_data.tobytes()).hexdigest()[:8]
        target_hash = hashlib.md5(str(target_class).encode()).hexdigest()[:4] if target_class is not None else "none"
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:4]
        
        return f"{method_name}_{data_hash}_{target_hash}_{params_hash}.pkl"
    
    def _save_to_cache(self, cache_key, data):
        """Save attribution results to cache"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def _load_from_cache(self, cache_key):
        """Load attribution results from cache"""
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load from cache: {e}")
        return None

    def _load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
            
            # Quick evaluation
            test_loss, test_acc = self.model.evaluate(self.x_test_flat, self.y_test, verbose=0)
            print(f"Model accuracy: {test_acc:.4f}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please run Rede.py first to train the model.")
    
    def get_sample_data(self, n_samples=100, random_state=42):
        """Get a sample of test data for evaluation"""
        np.random.seed(random_state)
        indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        
        return {
            'images': self.X_test[indices],
            'images_flat': self.x_test_flat[indices],
            'labels': self.y_test[indices],
            'indices': indices
        }
    

    
    def lime_method(self, input_data, target_class=None, n_samples=500):
        """
        LIME (Local Interpretable Model-agnostic Explanations)
        Uses local linear approximation to explain predictions
        """
        # Check cache first
        cache_key = self._generate_cache_key('lime', input_data, target_class, n_samples=n_samples)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        attributions = []
        
        for sample_idx in range(len(input_data)):
            sample = input_data[sample_idx:sample_idx+1]
            
            # Get original prediction
            original_pred = self.model.predict(sample, verbose=0)[0]
            if target_class is None:
                target_idx = np.argmax(original_pred)
            else:
                target_idx = target_class[sample_idx] if hasattr(target_class, '__len__') else target_class
            
            # Generate perturbed samples around the original
            sample_flat = sample.flatten()
            perturbed_samples = []
            perturbed_preds = []
            
            for _ in range(n_samples):
                # Create random mask (binary features on/off)
                mask = np.random.binomial(1, 0.5, size=sample_flat.shape)
                perturbed = sample_flat * mask
                
                perturbed_samples.append(perturbed)
                pred = self.model.predict(perturbed.reshape(1, -1), verbose=0)[0]
                perturbed_preds.append(pred[target_idx])
            
            perturbed_samples = np.array(perturbed_samples)
            perturbed_preds = np.array(perturbed_preds)
            
            # Fit linear model to explain local behavior
            ridge = Ridge(alpha=1.0)
            ridge.fit(perturbed_samples, perturbed_preds)
            
            # Use coefficients as feature importance
            lime_attribution = ridge.coef_
            attributions.append(lime_attribution)
        
        result = np.array(attributions)
        # Save to cache
        self._save_to_cache(cache_key, result)
        return result
    
    def shap_method(self, input_data, target_class=None, n_samples=200):
        """
        SHAP (SHapley Additive exPlanations)
        Uses more systematic sampling to approximate Shapley values
        """
        # Check cache first
        cache_key = self._generate_cache_key('shap', input_data, target_class, n_samples=n_samples)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        attributions = []
        
        for sample_idx in range(len(input_data)):
            sample = input_data[sample_idx]
            
            # Get target class
            if target_class is None:
                pred = self.model.predict(sample.reshape(1, -1), verbose=0)[0]
                target_idx = np.argmax(pred)
            else:
                target_idx = target_class[sample_idx] if hasattr(target_class, '__len__') else target_class
            
            # Use baseline of zeros (black image)
            baseline = np.zeros_like(sample)
            
            # Get baseline and sample predictions
            baseline_pred = self.model.predict(baseline.reshape(1, -1), verbose=0)[0][target_idx]
            sample_pred = self.model.predict(sample.reshape(1, -1), verbose=0)[0][target_idx]
            
            n_features = len(sample)
            shap_values = np.zeros(n_features)
            
            # For each feature, estimate its Shapley value
            for feature_idx in range(n_features):
                if sample[feature_idx] == 0:  # Skip if feature is already zero
                    continue
                    
                marginal_contributions = []
                
                # Sample different coalition sizes
                for coalition_size in range(0, min(30, n_features), 8):  # Larger steps, fewer sizes
                    for _ in range(max(1, n_samples // 30)):  # Fewer samples per coalition
                        # Create random coalition of given size (excluding current feature)
                        other_features = np.arange(n_features)
                        other_features = other_features[other_features != feature_idx]
                        
                        if coalition_size > 0 and len(other_features) > 0:
                            coalition_features = np.random.choice(
                                other_features, 
                                min(coalition_size, len(other_features)), 
                                replace=False
                            )
                        else:
                            coalition_features = []
                        
                        # Create samples with and without current feature
                        sample_with = baseline.copy()
                        sample_without = baseline.copy()
                        
                        # Add coalition features to both
                        for feat_idx in coalition_features:
                            sample_with[feat_idx] = sample[feat_idx]
                            sample_without[feat_idx] = sample[feat_idx]
                        
                        # Add current feature only to 'with' sample
                        sample_with[feature_idx] = sample[feature_idx]
                        
                        # Get predictions
                        pred_with = self.model.predict(sample_with.reshape(1, -1), verbose=0)[0][target_idx]
                        pred_without = self.model.predict(sample_without.reshape(1, -1), verbose=0)[0][target_idx]
                        
                        # Marginal contribution
                        marginal_contributions.append(pred_with - pred_without)
                
                # Average marginal contributions for this feature
                if marginal_contributions:
                    shap_values[feature_idx] = np.mean(marginal_contributions)
            
            attributions.append(shap_values)
        
        result = np.array(attributions)
        # Save to cache
        self._save_to_cache(cache_key, result)
        return result
    
    def gradcam_method(self, input_data, target_class=None):
        """
        GradCAM - Gradient-weighted Class Activation Mapping
        Simplified implementation for dense networks using layer gradients
        """
        # Check cache first
        cache_key = self._generate_cache_key('gradcam', input_data, target_class)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            print("  ✓ GradCAM loaded from cache")
            return cached_result
        
        attributions = []
        
        for sample_idx in range(len(input_data)):
            sample = input_data[sample_idx:sample_idx+1]
            sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
            
            # We'll get gradients w.r.t. intermediate layer by building sub-models
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(sample_tensor)
                
                # Forward pass through layers to get intermediate activations
                x = sample_tensor
                activations = []
                
                for i, layer in enumerate(self.model.layers):
                    x = layer(x)
                    activations.append(x)
                
                predictions = x
                
                if target_class is None:
                    target_idx = tf.argmax(predictions[0])
                else:
                    target_idx = target_class[sample_idx] if hasattr(target_class, '__len__') else target_class
                
                target_idx = tf.cast(target_idx, tf.int32)
                class_output = predictions[0][target_idx]
            
            # Get gradients w.r.t. penultimate layer activations
            penultimate_activations = activations[-2]  # Second to last layer
            grads = tape.gradient(class_output, penultimate_activations)
            
            if grads is not None:
                # Calculate importance weights by averaging gradients
                importance_weights = tf.reduce_mean(tf.abs(grads), axis=0)
                
                # Weight the activations
                weighted_activations = penultimate_activations[0] * importance_weights
                
                # Sum across activation dimensions to get scalar importance per activation
                activation_importance = tf.reduce_sum(tf.abs(weighted_activations), axis=-1)
                
                # Map back to input space
                # For dense layers, we create a spatial mapping based on neuron importance
                n_activations = tf.size(activation_importance)
                
                if n_activations < 784:
                    # Upscale activation importance to input size
                    scale_factor = 784 // n_activations
                    remainder = 784 % n_activations
                    
                    # Repeat each activation importance
                    upscaled = tf.repeat(activation_importance, scale_factor)
                    
                    # Handle remainder
                    if remainder > 0:
                        extra = activation_importance[:remainder]
                        upscaled = tf.concat([upscaled, extra], axis=0)
                    
                    gradcam_attribution = upscaled
                else:
                    # Downscale if needed
                    gradcam_attribution = activation_importance[:784]
                
                # Apply ReLU and normalize
                gradcam_attribution = tf.nn.relu(gradcam_attribution)
                
                # Normalize to [0,1]
                max_val = tf.reduce_max(gradcam_attribution)
                if max_val > 0:
                    gradcam_attribution = gradcam_attribution / max_val
                
                # Combine with input magnitude for better visualization
                input_magnitude = tf.abs(sample_tensor[0])
                final_attribution = gradcam_attribution * (1 + input_magnitude)
                
                attributions.append(final_attribution.numpy())
            else:
                # Fallback: use guided gradients
                input_grads = tape.gradient(class_output, sample_tensor)
                if input_grads is not None:
                    guided_grads = tf.nn.relu(input_grads[0])
                    attributions.append(guided_grads.numpy())
                else:
                    # Last resort: input magnitude
                    attributions.append(np.abs(input_data[sample_idx]))
            
            del tape  # Clean up persistent tape
        
        result = np.array(attributions)
        # Save to cache
        self._save_to_cache(cache_key, result)
        return result
    
    def integrated_gradients_method(self, input_data, target_class=None, m_steps=50):
        """
        Integrated Gradients
        """
        # Check cache first
        cache_key = self._generate_cache_key('integrated_gradients', input_data, target_class, m_steps=m_steps)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            print("  ✓ Integrated Gradients loaded from cache")
            return cached_result
        
        attributions = []
        
        for sample_idx in range(len(input_data)):
            sample = input_data[sample_idx]
            
            # Get target class prediction
            if target_class is None:
                sample_pred = self.model.predict(sample.reshape(1, -1), verbose=0)[0]
                target_idx = np.argmax(sample_pred)
            else:
                target_idx = target_class[sample_idx] if hasattr(target_class, '__len__') else target_class
            
            # Use baseline of zeros
            baseline = np.zeros_like(sample)
            
            # Compute gradients along the path
            path_gradients = []
            alphas = np.linspace(0, 1, m_steps)
            
            for alpha in alphas:
                # Interpolate between baseline and input
                interpolated = baseline + alpha * (sample - baseline)
                interpolated_tensor = tf.convert_to_tensor(interpolated.reshape(1, -1), dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    tape.watch(interpolated_tensor)
                    predictions = self.model(interpolated_tensor)
                    
                    # Convert target_idx to tensor if needed
                    if isinstance(target_idx, np.ndarray):
                        target_idx = target_idx.item()
                    
                    target_score = predictions[0][target_idx]
                
                # Compute gradients
                grads = tape.gradient(target_score, interpolated_tensor)
                if grads is not None:
                    path_gradients.append(grads.numpy().flatten())
                else:
                    path_gradients.append(np.zeros_like(sample))
            
            # Convert to numpy array
            path_gradients = np.array(path_gradients)
            
            # Compute integrated gradients using trapezoidal rule
            # This is more accurate than simple averaging
            integrated_grads = np.trapz(path_gradients, axis=0, dx=1.0/m_steps)
            
            # Scale by the path difference
            integrated_grads = integrated_grads * (sample - baseline)
            
            # Enhance contrast and reduce noise
            integrated_grads_abs = np.abs(integrated_grads)
            
            # Focus on most important features
            threshold = np.percentile(integrated_grads_abs, 75)  # Keep top 25%
            mask = integrated_grads_abs >= threshold
            
            # Apply mask and preserve sign
            final_attribution = np.where(mask, integrated_grads, integrated_grads * 0.1)
            
            # Slight smoothing only on the most important regions
            attribution_2d = final_attribution.reshape(28, 28)
            
            # Apply minimal smoothing
            from scipy import ndimage
            smoothed_2d = ndimage.gaussian_filter(attribution_2d, sigma=0.8)
            
            attributions.append(smoothed_2d.flatten())
        
        result = np.array(attributions)
        # Save to cache
        self._save_to_cache(cache_key, result)
        return result
    
    def saliency_maps(self, input_data, target_class=None):
        """
        Saliency Maps - Gradient-based attribution
        """
        # Check cache first
        cache_key = self._generate_cache_key('saliency_maps', input_data, target_class)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            print("  ✓ Saliency Maps loaded from cache")
            return cached_result
        
        attributions = []
        
        for sample_idx in range(len(input_data)):
            sample = input_data[sample_idx:sample_idx+1]
            
            # Get target class
            if target_class is None:
                sample_pred = self.model.predict(sample, verbose=0)[0]
                target_idx = np.argmax(sample_pred)
            else:
                target_idx = target_class[sample_idx] if hasattr(target_class, '__len__') else target_class
            
            sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(sample_tensor)
                predictions = self.model(sample_tensor)
                
                # Ensure target_idx is correct type
                if isinstance(target_idx, np.ndarray):
                    target_idx = target_idx.item()
                
                target_score = predictions[0][target_idx]
            
            # Get gradients
            gradients = tape.gradient(target_score, sample_tensor)
            
            if gradients is not None:
                grads_numpy = gradients[0].numpy()
                
                # Use squared gradients for better contrast
                saliency = np.square(grads_numpy)
                
                # Focus on high-magnitude gradients
                threshold = np.percentile(saliency, 85)  # Keep top 15%
                
                # Create high-contrast map
                high_contrast_saliency = np.where(saliency >= threshold, 
                                                saliency, 
                                                saliency * 0.05)  # Dim low-importance areas
                
                # Reshape to 2D for processing
                saliency_2d = high_contrast_saliency.reshape(28, 28)
                
                # Apply edge-preserving smoothing
                from scipy import ndimage
                
                # First pass: light gaussian for noise reduction
                smooth_light = ndimage.gaussian_filter(saliency_2d, sigma=0.5)
                
                # Second pass: preserve edges by combining with original
                final_2d = 0.8 * saliency_2d + 0.2 * smooth_light
                
                # Enhance contrast further
                # Apply histogram stretching
                p2, p98 = np.percentile(final_2d, (2, 98))
                if p98 > p2:
                    final_2d = np.clip((final_2d - p2) / (p98 - p2), 0, 1)
                
                # Power law transformation for better visibility
                final_2d = np.power(final_2d, 0.6)
                
                # Multiply by input magnitude to focus on relevant pixels
                input_2d = sample[0].reshape(28, 28)
                input_magnitude = np.abs(input_2d)
                
                # Combine saliency with input information
                combined_saliency = final_2d * (1 + 2 * input_magnitude)
                
                # Final normalization
                if combined_saliency.max() > 0:
                    combined_saliency = combined_saliency / combined_saliency.max()
                
                attributions.append(combined_saliency.flatten())
            else:
                # Fallback: enhanced input magnitude
                input_mag = np.abs(input_data[sample_idx])
                enhanced_mag = np.power(input_mag, 0.7)
                attributions.append(enhanced_mag)
        
        result = np.array(attributions)
        # Save to cache
        self._save_to_cache(cache_key, result)
        return result

    def get_all_attributions(self, input_data, target_class=None):
        """
        Get attributions from all 5 implemented methods for comparison
        """
        methods = {
            'lime': self.lime_method,
            'shap': self.shap_method, 
            'gradcam': self.gradcam_method,
            'integrated_gradients': self.integrated_gradients_method,
            'saliency_maps': self.saliency_maps
        }
        
        attributions = {}
        
        for method_name, method_func in methods.items():
            try:
                if method_name == 'random_baseline':
                    attribution = method_func(input_data)
                else:
                    attribution = method_func(input_data, target_class)
                
                attributions[method_name] = attribution
                print(f"✓ {method_name} computed successfully")
                
            except Exception as e:
                print(f"✗ Error computing {method_name}: {str(e)}")
                attributions[method_name] = None
        
        return attributions
    
    def visualize_attributions(self, input_data, attributions, sample_idx=0, figsize=(15, 10), 
                             save_image=True, output_dir='./Imagens'):
        """
        Visualize original image and all attribution methods and save to file
        """
        # Create output directory if it doesn't exist
        if save_image:
            os.makedirs(output_dir, exist_ok=True)
        n_methods = len([attr for attr in attributions.values() if attr is not None])
        n_cols = min(4, n_methods + 1)  # +1 for original image
        n_rows = (n_methods + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Show original image
        axes[0].imshow(input_data[sample_idx].reshape(28, 28), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show attributions
        plot_idx = 1
        for method_name, attribution in attributions.items():
            if attribution is not None and plot_idx < len(axes):
                # Reshape to 28x28 for visualization
                attr_reshaped = attribution[sample_idx].reshape(28, 28)
                
                # Normalize for better visualization
                attr_normalized = (attr_reshaped - attr_reshaped.min()) / (attr_reshaped.max() - attr_reshaped.min() + 1e-8)
                
                im = axes[plot_idx].imshow(attr_normalized, cmap='hot', alpha=0.8)
                axes[plot_idx].set_title(method_name.replace('_', ' ').title())
                axes[plot_idx].axis('off')
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save image with descriptive name
        if save_image:
            # Get the predicted class for the sample (use flattened data for model)
            input_flat = input_data[sample_idx:sample_idx+1].reshape(1, -1)
            sample_pred = self.model.predict(input_flat, verbose=0)[0]
            predicted_class = np.argmax(sample_pred)
            confidence = sample_pred[predicted_class]
            
            # Create descriptive filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            methods_list = "_".join([name for name, attr in attributions.items() if attr is not None])
            filename = f"tipos_de_metodos_sample{sample_idx}_class{predicted_class}_conf{confidence:.3f}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✓ Visualization saved as: {filepath}")
        
        plt.show()

    def save_multiple_visualizations(self, n_samples=5, output_dir='./Imagens'):
        """
        Generate and save visualizations for multiple samples
        """
        print(f"Generating visualizations for {n_samples} samples...")
        
        # Get sample data
        sample_data = self.get_sample_data(n_samples=n_samples)
        
        # Get attributions for all samples
        attributions = self.get_all_attributions(sample_data['images_flat'])
        
        # Create visualizations for each sample
        for i in range(n_samples):
            print(f"Creating visualization {i+1}/{n_samples}...")
            self.visualize_attributions(
                sample_data['images'], 
                attributions, 
                sample_idx=i,
                save_image=True,
                output_dir=output_dir
            )
            plt.close()  # Close figure to save memory
        
        print(f"✅ All visualizations saved in {output_dir}/")

    def create_methods_comparison_grid(self, n_samples=3, output_dir='./Imagens'):
        """
        Create a comprehensive grid showing all methods for multiple samples
        """
        print(f"Creating comprehensive comparison grid for {n_samples} samples...")
        
        # Get sample data
        sample_data = self.get_sample_data(n_samples=n_samples)
        attributions = self.get_all_attributions(sample_data['images_flat'])
        
        # Filter out None attributions
        valid_methods = {k: v for k, v in attributions.items() if v is not None}
        method_names = list(valid_methods.keys())
        
        # Create large grid: n_samples rows x (n_methods + 1) columns
        n_methods = len(method_names)
        fig, axes = plt.subplots(n_samples, n_methods + 1, figsize=(20, 5*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for sample_idx in range(n_samples):
            # Original image in first column
            axes[sample_idx, 0].imshow(sample_data['images'][sample_idx], cmap='gray')
            axes[sample_idx, 0].set_title(f'Original\nSample {sample_idx+1}')
            axes[sample_idx, 0].axis('off')
            
            # Attribution methods in remaining columns
            for method_idx, (method_name, attribution) in enumerate(valid_methods.items()):
                col_idx = method_idx + 1
                
                # Reshape and normalize attribution
                attr_reshaped = attribution[sample_idx].reshape(28, 28)
                attr_normalized = (attr_reshaped - attr_reshaped.min()) / (attr_reshaped.max() - attr_reshaped.min() + 1e-8)
                
                im = axes[sample_idx, col_idx].imshow(attr_normalized, cmap='hot', alpha=0.8)
                axes[sample_idx, col_idx].set_title(f'{method_name.replace("_", " ").title()}\nSample {sample_idx+1}')
                axes[sample_idx, col_idx].axis('off')
        
        plt.tight_layout()
        
        # Save comprehensive grid
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tipos_de_metodos_comprehensive_grid_{n_samples}samples_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Comprehensive grid saved as: {filepath}")
        
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Test the implementation and generate saved visualizations
    interp = InterpretabilityMethods()
    
    print("🎨 Generating and saving interpretability visualizations...")
    
    # Option 1: Save individual visualizations for multiple samples
    interp.save_multiple_visualizations(n_samples=3)
    
    # Option 2: Create comprehensive comparison grid
    interp.create_methods_comparison_grid(n_samples=2)
    
    print("\n✅ All visualizations saved! Check the './Imagens' folder.")
    print("🎯 Interpretability methods ready for metric evaluation!")
