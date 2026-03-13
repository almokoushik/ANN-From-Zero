"""
Deep Neural Network Implementation from Scratch
A simple 3-layer neural network built with NumPy only.
No fancy frameworks - just pure linear algebra!
"""

import numpy as np
import time


class DeepNeuralNetwork():
    def __init__(self, sizes, activation='sigmoid'):
        """
        Initialize the neural network with given architecture.
        
        Args:
            sizes: List of layer sizes, e.g., [784, 64, 10] for input, hidden, output
            activation: Activation function to use ('sigmoid' or 'relu')
        """
        self.sizes = sizes
        
        # Pick which activation function to use
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Use 'relu' or 'sigmoid' as activation function, mate!")
        
        # Initialize all the weights and biases
        self.params = self.initialize()
        
        # Store intermediate values during forward pass (we'll need them for backprop)
        self.cache = {}
        
    def relu(self, x, derivative=False):
        """
        ReLU activation function - Returns max(0, x)
        Really simple: takes negative values to 0, keeps positive ones as-is.
        This helps the network learn non-linear patterns!
        """
        if derivative:
            # For backprop: gradient is 0 if x<0, and 1 if x>=0
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        
        # Forward pass: just clamp negatives to 0
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        """
        Sigmoid activation function - squashes values between 0 and 1
        Useful because it gives us probability-like outputs.
        The S-shaped curve means big changes in input = small changes in output near extremes
        """
        if derivative:
            # Gradient for backprop: σ'(x) = σ(x) * (1 - σ(x))
            # This is derived mathematically but trust me, it works!
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        
        # Forward: the classic sigmoid formula
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Softmax function - converts raw scores into probability distribution
        Makes sure all outputs are between 0 and 1 and sum to 1.
        Perfect for multi-class classification!
        
        Note: We subtract max for numerical stability (prevents overflow with large exponents)
        """
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        """
        Set up initial weights and biases.
        We use He initialization: scale weights by sqrt(1/n) where n is input size.
        This prevents the vanishing/exploding gradient problem!
        """
        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        output_layer = self.sizes[2]
        
        # Random weights scaled properly, biases start at zero
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1. / input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1. / input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1. / hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1. / hidden_layer)
        }
        return params
    
    def initialize_momemtum_optimizer(self):
        """
        Set up momentum variables - these remember where we came from!
        Used to smooth out the updates and escape local minima.
        """
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
        }
        return momemtum_opt

    def feed_forward(self, x):
        """
        Run data through the network (forward pass).
        We compute: 
        - Z = W*X + b (linear transformation)
        - A = activation(Z) (add non-linearity)
        
        We save intermediate values in self.cache because we'll need them for backprop!
        """
        # Input layer - just save it
        self.cache["X"] = x
        
        # First hidden layer
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        
        # Output layer
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        
        # Use softmax for final output (gives us probabilities for each class)
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        
        return self.cache["A2"]
    
    def back_propagate(self, y, output):
        """
        Run backpropagation to compute gradients.
        This tells us how much to adjust each weight to reduce the error.
        
        We work backwards from the output layer, computing:
        - How wrong are we? (dZ2)
        - How do we fix the weights? (dW2, db2)
        - What did the previous layer do wrong? (dA1)
        - Repeat for previous layers...
        """
        current_batch_size = y.shape[0]
        
        # Output layer: how far off are our predictions?
        dZ2 = output - y.T
        
        # How much should we adjust W2 and b2?
        dW2 = (1. / current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1. / current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        # Backprop to the hidden layer
        dA1 = np.matmul(self.params["W2"].T, dZ2)
        
        # Account for the activation function's gradient
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        
        # Gradients for first layer weights
        dW1 = (1. / current_batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1. / current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        # Collect all gradients
        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads
    
    def cross_entropy_loss(self, y, output):
        """
        Measure how wrong our predictions are using cross-entropy loss.
        Lower loss = better predictions. This is what we're trying to minimize!
        
        Formula: L = -mean(y * log(output))
        """
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1. / m) * l_sum
        return l
                
    def optimize(self, l_rate=0.1, beta=.9):
        """
        Update the weights using the computed gradients.
        Two strategies available:
        
        1. SGD (Stochastic Gradient Descent):
           Just go in the direction of the negative gradient with learning rate L
           
        2. Momentum:
           Remember where you came from and blend that with new gradient
           This helps escape local minima and smooth out the learning!
        """
        if self.optimizer == "sgd":
            # Simple approach: just adjust by the gradient
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
                
        elif self.optimizer == "momentum":
            # Fancier approach: remember momentum from last step
            for key in self.params:
                # Momentum term: carry forward 90% of last velocity, add 10% of new gradient
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + 
                                          (1. - beta) * self.grads[key])
                # Update using the momentum
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Use 'sgd' or 'momentum' optimizer, please!")

    def accuracy(self, y, output):
        """
        Check what percentage of predictions are correct.
        Compare the predicted class with the actual class.
        """
        return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))

    def train(self, x_train, y_train, x_test, y_test, epochs=10, 
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):
        """
        Train the neural network!
        
        Args:
            x_train, y_train: Training data and labels
            x_test, y_test: Test data and labels (for validation)
            epochs: How many times to go through the whole dataset
            batch_size: How many samples to process before updating weights
            optimizer: 'sgd' or 'momentum'
            l_rate: Learning rate (how big are our steps?)
            beta: Momentum parameter (higher = more memory of past)
        """
        # Store these for later use
        self.epochs = epochs
        self.batch_size = batch_size
        
        # How many batches per epoch?
        num_batches = -(-x_train.shape[0] // self.batch_size)
        
        # Set up the optimizer we want
        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_momemtum_optimizer()
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        
        # Main training loop
        for epoch in range(self.epochs):
            # Shuffle the data so we don't learn patterns from data order
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            # Process each batch
            for batch_idx in range(num_batches):
                # Get this batch
                begin = batch_idx * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0] - 1)
                x_batch = x_train_shuffled[begin:end]
                y_batch = y_train_shuffled[begin:end]
                
                # Forward pass: make predictions
                output = self.feed_forward(x_batch)
                
                # Backward pass: compute gradients
                _ = self.back_propagate(y_batch, output)
                
                # Update weights using the gradients
                self.optimize(l_rate=l_rate, beta=beta)

            # After each epoch, check how we're doing
            # On training data
            output_train = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output_train)
            train_loss = self.cross_entropy_loss(y_train, output_train)
            
            # On test data (unseen data - more important!)
            output_test = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output_test)
            test_loss = self.cross_entropy_loss(y_test, output_test)
            
            # Print progress
            print(template.format(epoch + 1, time.time() - start_time, 
                                train_acc, train_loss, test_acc, test_loss))