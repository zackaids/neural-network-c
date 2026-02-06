#include "neural_net.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Activation functions
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_derivative(double x)
{
    return x > 0 ? 1.0 : 0.0;
}

double tanh_activation(double x)
{
    return tanh(x);
}

double tanh_derivative(double x)
{
    double t = tanh(x);
    return 1.0 - t * t;
}

// Apply activation function
static double apply_activation(double x, ActivationType type)
{
    switch (type)
    {
    case ACTIVATION_SIGMOID:
        return sigmoid(x);
    case ACTIVATION_RELU:
        return relu(x);
    case ACTIVATION_TANH:
        return tanh_activation(x);
    default:
        return x;
    }
}

// Apply activation derivative
static double apply_activation_derivative(double x, ActivationType type)
{
    switch (type)
    {
    case ACTIVATION_SIGMOID:
        return sigmoid_derivative(x);
    case ACTIVATION_RELU:
        return relu_derivative(x);
    case ACTIVATION_TANH:
        return tanh_derivative(x);
    default:
        return 1.0;
    }
}

// Layer creation
Layer *layer_create(size_t input_size, size_t output_size, ActivationType activation)
{
    Layer *layer = (Layer *)malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;

    layer->weights = (double *)malloc(output_size * input_size * sizeof(double));
    layer->biases = (double *)calloc(output_size, sizeof(double));
    layer->activations = (double *)malloc(output_size * sizeof(double));
    layer->z_values = (double *)malloc(output_size * sizeof(double));

    randomize_weights(layer);

    return layer;
}

void layer_destroy(Layer *layer)
{
    free(layer->weights);
    free(layer->biases);
    free(layer->activations);
    free(layer->z_values);
    free(layer);
}

void randomize_weights(Layer *layer)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand(time(NULL));
        seeded = 1;
    }

    // Xavier initialization
    double limit = sqrt(6.0 / (layer->input_size + layer->output_size));

    for (size_t i = 0; i < layer->output_size * layer->input_size; i++)
    {
        layer->weights[i] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
    }
}

void layer_forward(Layer *layer, double *input)
{
    // Compute z = W*x + b
    for (size_t i = 0; i < layer->output_size; i++)
    {
        layer->z_values[i] = layer->biases[i];
        for (size_t j = 0; j < layer->input_size; j++)
        {
            layer->z_values[i] += layer->weights[i * layer->input_size + j] * input[j];
        }
        layer->activations[i] = apply_activation(layer->z_values[i], layer->activation);
    }
}

// Neural Network creation
NeuralNetwork *nn_create(size_t *layer_sizes, size_t num_layers, double learning_rate)
{
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers - 1; // number of layer transitions
    nn->learning_rate = learning_rate;
    nn->layers = (Layer *)malloc(nn->num_layers * sizeof(Layer));

    for (size_t i = 0; i < nn->num_layers; i++)
    {
        ActivationType activation = (i == nn->num_layers - 1) ? ACTIVATION_SIGMOID : ACTIVATION_RELU;
        nn->layers[i] = *layer_create(layer_sizes[i], layer_sizes[i + 1], activation);
    }

    return nn;
}

void nn_destroy(NeuralNetwork *nn)
{
    for (size_t i = 0; i < nn->num_layers; i++)
    {
        free(nn->layers[i].weights);
        free(nn->layers[i].biases);
        free(nn->layers[i].activations);
        free(nn->layers[i].z_values);
    }
    free(nn->layers);
    free(nn);
}

void nn_forward(NeuralNetwork *nn, double *input)
{
    double *current_input = input;

    for (size_t i = 0; i < nn->num_layers; i++)
    {
        layer_forward(&nn->layers[i], current_input);
        current_input = nn->layers[i].activations;
    }
}

double *nn_predict(NeuralNetwork *nn, double *input)
{
    nn_forward(nn, input);
    return nn->layers[nn->num_layers - 1].activations;
}

double compute_loss(double *predictions, double *targets, size_t size)
{
    double loss = 0.0;
    for (size_t i = 0; i < size; i++)
    {
        double diff = predictions[i] - targets[i];
        loss += diff * diff;
    }
    return loss / (2.0 * size);
}

// Simple backpropagation training (single sample)
void nn_train(NeuralNetwork *nn, double *input, double *target, size_t num_samples)
{
    for (size_t sample = 0; sample < num_samples; sample++)
    {
        // Forward pass
        nn_forward(nn, input + sample * nn->layers[0].input_size);

        // Allocate space for gradients
        double **deltas = (double **)malloc(nn->num_layers * sizeof(double *));
        for (size_t i = 0; i < nn->num_layers; i++)
        {
            deltas[i] = (double *)malloc(nn->layers[i].output_size * sizeof(double));
        }

        // Backward pass - compute output layer delta
        Layer *output_layer = &nn->layers[nn->num_layers - 1];
        for (size_t i = 0; i < output_layer->output_size; i++)
        {
            double error = output_layer->activations[i] - target[sample * output_layer->output_size + i];
            deltas[nn->num_layers - 1][i] = error *
                                            apply_activation_derivative(output_layer->z_values[i], output_layer->activation);
        }

        // Backward pass - propagate deltas to hidden layers
        for (int l = nn->num_layers - 2; l >= 0; l--)
        {
            Layer *current = &nn->layers[l];
            Layer *next = &nn->layers[l + 1];

            for (size_t i = 0; i < current->output_size; i++)
            {
                double error = 0.0;
                for (size_t j = 0; j < next->output_size; j++)
                {
                    error += next->weights[j * next->input_size + i] * deltas[l + 1][j];
                }
                deltas[l][i] = error * apply_activation_derivative(current->z_values[i], current->activation);
            }
        }

        // Update weights and biases
        for (size_t l = 0; l < nn->num_layers; l++)
        {
            Layer *layer = &nn->layers[l];
            double *prev_activations = (l == 0) ? (input + sample * layer->input_size) : nn->layers[l - 1].activations;

            for (size_t i = 0; i < layer->output_size; i++)
            {
                for (size_t j = 0; j < layer->input_size; j++)
                {
                    layer->weights[i * layer->input_size + j] -=
                        nn->learning_rate * deltas[l][i] * prev_activations[j];
                }
                layer->biases[i] -= nn->learning_rate * deltas[l][i];
            }
        }

        // Free deltas
        for (size_t i = 0; i < nn->num_layers; i++)
        {
            free(deltas[i]);
        }
        free(deltas);
    }
}