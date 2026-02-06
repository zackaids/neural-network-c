#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stddef.h>

// Activation function types
typedef enum
{
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_TANH
} ActivationType;

// Layer structure
typedef struct
{
    size_t input_size;
    size_t output_size;
    double *weights;     // weights matrix (output_size x input_size)
    double *biases;      // bias vector (output_size)
    double *activations; // activation outputs (output_size)
    double *z_values;    // pre-activation values (output_size)
    ActivationType activation;
} Layer;

// Neural Network structure
typedef struct
{
    Layer *layers;
    size_t num_layers;
    double learning_rate;
} NeuralNetwork;

// Core functions
NeuralNetwork *nn_create(size_t *layer_sizes, size_t num_layers, double learning_rate);
void nn_destroy(NeuralNetwork *nn);
void nn_forward(NeuralNetwork *nn, double *input);
void nn_train(NeuralNetwork *nn, double *input, double *target, size_t num_samples);
double *nn_predict(NeuralNetwork *nn, double *input);

// Layer functions
Layer *layer_create(size_t input_size, size_t output_size, ActivationType activation);
void layer_destroy(Layer *layer);
void layer_forward(Layer *layer, double *input);

// Activation functions
double sigmoid(double x);
double sigmoid_derivative(double x);
double relu(double x);
double relu_derivative(double x);
double tanh_activation(double x);
double tanh_derivative(double x);

// Utility functions
void randomize_weights(Layer *layer);
double compute_loss(double *predictions, double *targets, size_t size);

#endif // NEURAL_NET_H