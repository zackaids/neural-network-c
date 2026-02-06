#include "neural_net.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
    printf("=== Neural Network Framework - Proof of Concept ===\n\n");

    // XOR problem demonstration
    printf("Training on XOR problem...\n");

    // Network architecture: 2 inputs -> 4 hidden -> 1 output
    size_t architecture[] = {2, 4, 1};
    NeuralNetwork *nn = nn_create(architecture, 3, 0.5);

    // XOR training data
    double inputs[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0};

    double targets[] = {
        0.0,
        1.0,
        1.0,
        0.0};

    // Training loop
    printf("Training for 5000 epochs...\n");
    for (int epoch = 0; epoch < 5000; epoch++)
    {
        nn_train(nn, inputs, targets, 4);

        if (epoch % 1000 == 0)
        {
            // Calculate average loss
            double total_loss = 0.0;
            for (int i = 0; i < 4; i++)
            {
                double *prediction = nn_predict(nn, &inputs[i * 2]);
                total_loss += compute_loss(prediction, &targets[i], 1);
            }
            printf("Epoch %d - Average Loss: %.6f\n", epoch, total_loss / 4);
        }
    }

    printf("\n=== Test Results ===\n");
    for (int i = 0; i < 4; i++)
    {
        double *prediction = nn_predict(nn, &inputs[i * 2]);
        printf("Input: [%.0f, %.0f] -> Output: %.4f (Target: %.0f)\n",
               inputs[i * 2], inputs[i * 2 + 1], prediction[0], targets[i]);
    }

    // Cleanup
    nn_destroy(nn);

    return 0;
}