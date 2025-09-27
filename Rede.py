import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import struct
import os

def load_mnist_from_local(path='./sample_data/MNIST/raw'):
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
    
    # Load training data
    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte'))
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte'))
    
    # Load test data
    X_test = load_images(os.path.join(path, 't10k-images-idx3-ubyte'))
    y_test = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte'))
    
    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = load_mnist_from_local()

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


x_train_flat = X_train.reshape((len(X_train), (28 * 28)))
x_test_flat = X_test.reshape((len(X_test), (28 * 28)))


model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='sigmoid'),
    keras.layers.Dense(32, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Check if model weights already exist
model_weights_path = 'mnist_model.weights.h5'
model_path = 'mnist_model.keras'

if os.path.exists(model_weights_path):
    print("Loading existing model weights...")
    model.load_weights(model_weights_path)
    
    # Evaluate the loaded model
    test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=2)
    print(f'\nLoaded model test accuracy: {test_acc:.4f}')
else:
    print("Training new model...")
    # Train the model and evaluate with train and validation data
    history = model.fit(x_train_flat, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    # Save the trained model weights and full model
    model.save_weights(model_weights_path)
    model.save(model_path)
    print(f"Model weights saved to {model_weights_path}")
    print(f"Full model saved to {model_path}")
    
    # Evaluate the model with test data
    test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=2)
    print(f'\nTrained model test accuracy: {test_acc:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

print("Model ready for interpretability analysis!")
print(f"Model weights available at: {model_weights_path}")
print(f"Full model available at: {model_path}")