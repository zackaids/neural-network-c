# neural-network-c

A lightweight, educational neural network framework implemented in pure C. This proof of concept demonstrates the fundamentals of deep learning with a clean, understandable API.

## Features

- **Feedforward Neural Networks**: Build multi-layer perceptrons with arbitrary architectures
- **Multiple Activation Functions**: Sigmoid, ReLU, and Tanh supported
- **Backpropagation**: Automatic gradient computation and weight updates
- **Xavier Initialization**: Smart weight initialization for better convergence
- **Clean API**: Simple, intuitive function calls for network creation and training
- **Pure C**: No external dependencies beyond the standard math library

## Project Structure

```
.
├── neural_net.h      # Header file with API definitions
├── neural_net.c      # Core implementation
├── example.c         # XOR problem demonstration
├── Makefile          # Build configuration
└── README.md         # This file
```

## Quick Start

### Building

```bash
make
```

### Running the Demo

```bash
make run
```

This will train a neural network to solve the XOR problem, a classic non-linearly separable classification task.

### Cleaning Build Files

```bash
make clean
```

## Usage Example

```c
#include "neural_net.h"

// Define network architecture: 2 inputs, 4 hidden neurons, 1 output
size_t architecture[] = {2, 4, 1};
NeuralNetwork *nn = nn_create(architecture, 3, 0.5);

// Prepare training data
double inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
double targets[] = {0.0, 1.0, 1.0, 0.0};

// Train the network
for (int epoch = 0; epoch < 5000; epoch++) {
    nn_train(nn, inputs, targets, 4);
}

// Make predictions
double test[] = {1.0, 0.0};
double *prediction = nn_predict(nn, test);
printf("Output: %.4f\n", prediction[0]);

// Cleanup
nn_destroy(nn);
```

## API Reference

### Network Management

- `NeuralNetwork* nn_create(size_t *layer_sizes, size_t num_layers, double learning_rate)`
  - Creates a new neural network with specified architecture
  - `layer_sizes`: array defining neurons per layer
  - `num_layers`: total number of layers (including input)
  - `learning_rate`: gradient descent step size

- `void nn_destroy(NeuralNetwork *nn)`
  - Frees all memory associated with the network

### Training & Prediction

- `void nn_train(NeuralNetwork *nn, double *input, double *target, size_t num_samples)`
  - Trains the network using backpropagation
  - Updates weights based on the difference between predictions and targets

- `double* nn_predict(NeuralNetwork *nn, double *input)`
  - Returns the network's output for given input
  - Output pointer is valid until next forward pass

- `double compute_loss(double *predictions, double *targets, size_t size)`
  - Computes mean squared error loss

### Activation Functions

- `ACTIVATION_SIGMOID`: Smooth S-curve, outputs in (0, 1)
- `ACTIVATION_RELU`: Rectified Linear Unit, faster training
- `ACTIVATION_TANH`: Hyperbolic tangent, outputs in (-1, 1)

## Current Limitations (Proof of Concept)

This is an educational proof of concept with intentional simplifications:

- Single sample training (no batch processing)
- Fixed mean squared error loss
- No regularization (L1/L2)
- No momentum or advanced optimizers
- Limited to dense (fully connected) layers
- No model serialization/deserialization

## Future Roadmap

Potential extensions for the full project:

1. **Optimization**
   - Mini-batch gradient descent
   - Adam, RMSprop, and other optimizers
   - Learning rate scheduling

2. **Architecture**
   - Convolutional layers (CNNs)
   - Recurrent layers (RNNs/LSTMs)
   - Dropout and batch normalization

3. **Features**
   - Save/load trained models
   - GPU acceleration (CUDA)
   - Data preprocessing utilities
   - Automatic differentiation

4. **Loss Functions**
   - Cross-entropy for classification
   - Custom loss functions

5. **Tools**
   - Visualization of training progress
   - Network architecture visualization
   - Performance profiling

## Technical Details

### Memory Management

The framework uses dynamic memory allocation for flexibility. All memory is manually managed:

- Weights and biases are heap-allocated
- Activation values are stored per layer
- Gradients are computed on-the-fly during backpropagation

### Numerical Stability

- Xavier/Glorot initialization prevents gradient vanishing/explosion
- Double precision floating point throughout
- Activation function derivatives are numerically stable

## Requirements

- C99-compatible compiler (gcc, clang)
- Standard math library (`-lm`)
- POSIX-compliant system (Linux, macOS, Unix)

## License

This is a proof of concept for educational purposes. Feel free to use and modify.

## Contributing

This is a proof of concept. For the full project, consider:

- Adding unit tests
- Improving documentation
- Implementing additional layer types
- Optimizing performance

## Acknowledgments

Built as a demonstration of neural network fundamentals in pure C, inspired by classic machine learning literature and modern deep learning frameworks.
