# Neural Networks from Scratch 

Building a neural network from the ground up using **only NumPy** - no fancy deep learning frameworks, just pure math and Linear Algebra!

This project builds a simple 3-layer neural network to classify handwritten digits from the MNIST dataset.

---

##  Project Overview

### What's Inside?

- **model.py** — The heart of the project! Contains the `DeepNeuralNetwork` class with all the neural network logic
- **train.py** — Script to train the model with customizable hyperparameters
- **NN-from-Scratch.ipynb** — Jupyter notebook with detailed explanations and visualizations

### The Network Architecture

```
Input Layer (784 nodes)
    ↓
Hidden Layer (64 nodes with ReLU/Sigmoid)
    ↓
Output Layer (10 nodes with Softmax)
```

**Why these sizes?**
- MNIST images are 28×28 pixels → 28×28 = 784 flattened inputs
- 64 hidden neurons = good balance between complexity and speed
- 10 output neurons = one for each digit (0-9)

---

##  Quick Start

### Installation

```bash
# Install required packages
pip install numpy scikit-learn matplotlib
```

### Train the Model

**Basic training (uses defaults):**
```bash
python train.py
```

**Custom hyperparameters:**
```bash
# Using ReLU activation with SGD optimizer
python train.py --activation relu --optimizer sgd --l_rate 0.05 --batch_size 128

# Using Sigmoid with Momentum optimizer (usually better!)
python train.py --activation sigmoid --optimizer momentum --l_rate 4 --beta 0.9 --batch_size 128
```

### Command-Line Arguments

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--activation` | sigmoid | sigmoid, relu | Activation function to use |
| `--optimizer` | momentum | sgd, momentum | Optimization algorithm |
| `--batch_size` | 128 | int | Samples per batch |
| `--l_rate` | 0.001 | float | Learning rate |
| `--beta` | 0.9 | float | Momentum parameter (momentum only) |

---

##  How It Works

### The Forward Pass (Prediction)

For each image, we compute:

```
Layer 1:  Z1 = W1 × X + b1
          A1 = activation(Z1)

Layer 2:  Z2 = W2 × A1 + b2
          Output = softmax(Z2)  ← Probabilities for each digit!
```

### The Backward Pass (Learning)

We calculate how much each weight contributed to the error:

```
1. Compute error at output: dZ2 = predicted - actual
2. Compute gradients: dW2, db2
3. Backprop to hidden layer: dA1
4. Account for activation: dZ1
5. Compute gradients: dW1, db1
```

### Weight Updates

Two strategies:

**SGD (Stochastic Gradient Descent):**
```
weight_new = weight_old - learning_rate × gradient
```

**Momentum (usually better!):**
```
velocity = beta × velocity_old + gradient
weight_new = weight_old - learning_rate × velocity
```

Momentum helps us:
- Move faster down the slope
- Avoid getting stuck in local minima
- Smooth out the training

---

##  Activation Functions

### ReLU (Rectified Linear Unit)
```python
relu(x) = max(0, x)
```
- **Pros:** Fast to compute, works well with deep networks
- **Cons:** Needs smaller learning rates
- **Use when:** You want simple, fast learning

### Sigmoid
```python
sigmoid(x) = 1 / (1 + e^(-x))
```
- **Pros:** Output is probability-like (0-1), smooth gradient
- **Cons:** Slower to compute, can saturate
- **Use when:** You want stable gradients

### Softmax (Output layer)
```python
softmax(z_i) = e^(z_i) / Σ(e^(z_j))
```
- Converts raw scores into probabilities
- Probabilities sum to 1 → perfect for classification!

---

##  Loss Functions

### Cross-Entropy Loss

```
L = -mean(y × log(y_predicted))
```

Measures how "wrong" our predictions are:
- Lower loss = better predictions
- This is what we're trying to minimize during training

---

##  Key Insights

### Weight Initialization
We use He initialization: `weight ~ N(0, √(1/n))`

Why? It prevents:
- **Vanishing gradients:** Gradients become too small to learn
- **Exploding gradients:** Gradients become too large and unstable

### Batch Processing
We process data in batches (not one sample at a time):
- Faster than processing individually
- Smoother gradient estimates
- Better hardware utilization

### Numerical Stability
In softmax, we subtract `max(x)` before exponentiating:
```python
exps = np.exp(x - x.max())  # Prevents overflow!
```

---

## Training Tips

### For ReLU networks:
- Use **smaller learning rates** (0.01 to 0.1)
- Use **SGD optimizer**
- Expect **faster convergence**

### For Sigmoid networks:
- Use **larger learning rates** (1.0 to 10.0)
- Use **Momentum optimizer** (usually better!)
- More stable but slower

### General Good Practice:
- Start with momentum optimizer (usually works best)
- Use smaller batch sizes if memory allows
- Monitor both training AND test accuracy (avoid overfitting!)

---

##  Understanding the Output

When you run `train.py`, you'll see output like:

```
Epoch 1: 12.34s, train acc=0.92, train loss=0.25, test acc=0.90, test loss=0.28
Epoch 2: 24.68s, train acc=0.94, train loss=0.18, test acc=0.92, test loss=0.22
...
```

**What does it mean?**
- **train acc:** How many training images we got right (92%)
- **train loss:** How wrong we were on training data (lower is better)
- **test acc:** How many test images we got right (90%)
- **test loss:** How wrong we were on unseen test data

** Watch out for overfitting:**
If `train acc` is much higher than `test acc`, the model is memorizing training data, not learning general patterns!

---

##  Code Structure

### model.py

```python
class DeepNeuralNetwork:
    __init__()              # Set up the network
    relu()                  # ReLU activation
    sigmoid()               # Sigmoid activation
    softmax()               # Softmax (output layer)
    initialize()            # Initialize weights
    feed_forward()          # Make predictions
    back_propagate()        # Compute gradients
    cross_entropy_loss()    # Measure error
    optimize()              # Update weights
    accuracy()              # Check performance
    train()                 # Main training loop
```

---



