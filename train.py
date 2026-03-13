"""
Training script for the Deep Neural Network
Run this to train the model with your own hyperparameters!
"""

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time
import argparse
from model import DeepNeuralNetwork

# Parse command line arguments
# This lets us easily experiment with different hyperparameters!
parser = argparse.ArgumentParser(description='Train Neural Network from Scratch!')
parser.add_argument('--activation', 
                    action='store', 
                    dest='activation', 
                    required=False, 
                    default='sigmoid', 
                    help='Activation function: sigmoid or relu')
parser.add_argument('--batch_size', 
                    action='store', 
                    dest='batch_size', 
                    required=False, 
                    default=128,
                    help='Batch size for training')
parser.add_argument('--optimizer', 
                    action='store', 
                    dest='optimizer', 
                    required=False, 
                    default='momentum', 
                    help='Optimizer: sgd or momentum')
parser.add_argument('--l_rate', 
                    action='store', 
                    dest='l_rate', 
                    required=False, 
                    default=1e-3, 
                    help='Learning rate (how big our update steps are)')
parser.add_argument('--beta', 
                    action='store', 
                    dest='beta', 
                    required=False, 
                    default=.9, 
                    help='Beta for momentum optimizer (memory factor)')
args = parser.parse_args()

# Helper function to visualize images
def show_images(image, num_row=2, num_col=5):
    """
    Display a grid of images (useful for looking at MNIST samples)
    Reshapes flattened images back to 28x28 format for viewing
    """
    # Reshape flat arrays back to square images
    image_size = int(np.sqrt(image.shape[-1]))
    image = np.reshape(image, (image.shape[0], image_size, image_size))
    
    # Create a subplot grid
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    
    # Plot each image
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    
def one_hot(x, k, dtype=np.float32):
    """
    Convert class labels to one-hot encoding.
    Example: label 3 with 10 classes -> [0,0,0,1,0,0,0,0,0,0]
    This is what our neural network outputs!
    """
    # Convert pandas Series to numpy array if needed (fixes compatibility)
    x = np.asarray(x)
    return np.array(x[:, None] == np.arange(k), dtype)


def main():
    """
    Main training routine.
    Loads MNIST, preprocesses it, and trains the neural network.
    """
    
    # Step 1: Load the MNIST dataset
    # This downloads the data from OpenML (might take a minute on first run)
    print(" Loading MNIST dataset...")
    mnist_data = fetch_openml("mnist_784")
    x = mnist_data["data"]
    y = mnist_data["target"]

    # Convert pandas DataFrames/Series to numpy arrays for compatibility
    x = np.asarray(x)
    y = np.asarray(y)

    # Step 2: Preprocess the data
    print(" Preprocessing data...")
    
    # Normalize pixel values from [0,255] to [0,1]
    # This helps the network learn better (numerical stability!)
    # Convert to float first to avoid casting issues
    x = x.astype(np.float32)
    x /= 255.0

    # Convert labels to one-hot encoding
    # Example: label 3 becomes [0,0,0,1,0,0,0,0,0,0]
    num_labels = 10
    examples = y.shape[0]
    y_new = one_hot(y.astype('int32'), num_labels)

    # Split into train and test sets
    # 60,000 for training, 10,000 for testing
    train_size = 60000
    test_size = x.shape[0] - train_size
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y_new[:train_size], y_new[train_size:]
    
    # Shuffle training data (random order helps learning)
    shuffle_index = np.random.permutation(train_size)
    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
    
    print(f" Training data: {x_train.shape}")
    print(f" Test data: {x_test.shape}")

    # Step 3: Train the network!
    print("\n Starting training...")
    dnn = DeepNeuralNetwork(sizes=[784, 64, 10], activation=args.activation)
    dnn.train(x_train, y_train, x_test, y_test, 
              batch_size=int(args.batch_size), 
              optimizer=args.optimizer, 
              l_rate=float(args.l_rate), 
              beta=float(args.beta))
    
    print("\n Training complete!")

    

if __name__ == '__main__':
    main()