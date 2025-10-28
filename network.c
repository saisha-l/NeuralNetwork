#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 2
#define OUTPUT_NEURONS 1
#define TRAINING_SAMPLES 4
#define LEARNING_RATE 0.5
#define EPOCHS 10000

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double rand_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

int main() {
    srand(time(NULL));

    double X[TRAINING_SAMPLES][INPUT_NEURONS] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double y[TRAINING_SAMPLES] = {0, 1, 1, 0};

    double w_input_hidden[INPUT_NEURONS][HIDDEN_NEURONS];
    double w_hidden_output[HIDDEN_NEURONS];
    double b_hidden[HIDDEN_NEURONS];
    double b_output;

    for (int i = 0; i < INPUT_NEURONS; i++)
        for (int j = 0; j < HIDDEN_NEURONS; j++)
            w_input_hidden[i][j] = rand_weight();

    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        w_hidden_output[j] = rand_weight();
        b_hidden[j] = rand_weight();
    }
    b_output = rand_weight();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int sample = 0; sample < TRAINING_SAMPLES; sample++) {
            // Forward pass
            double hidden[HIDDEN_NEURONS];
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                hidden[j] = b_hidden[j];
                for (int i = 0; i < INPUT_NEURONS; i++)
                    hidden[j] += X[sample][i] * w_input_hidden[i][j];
                hidden[j] = sigmoid(hidden[j]);
            }

            double output = b_output;
            for (int j = 0; j < HIDDEN_NEURONS; j++)
                output += hidden[j] * w_hidden_output[j];
            output = sigmoid(output);

            double error = y[sample] - output;

            double delta_output = error * sigmoid_derivative(output);

            double delta_hidden[HIDDEN_NEURONS];
            for (int j = 0; j < HIDDEN_NEURONS; j++)
                delta_hidden[j] = delta_output * w_hidden_output[j] * sigmoid_derivative(hidden[j]);

            for (int j = 0; j < HIDDEN_NEURONS; j++)
                w_hidden_output[j] += LEARNING_RATE * delta_output * hidden[j];
            b_output += LEARNING_RATE * delta_output;

            for (int i = 0; i < INPUT_NEURONS; i++)
                for (int j = 0; j < HIDDEN_NEURONS; j++)
                    w_input_hidden[i][j] += LEARNING_RATE * delta_hidden[j] * X[sample][i];
            for (int j = 0; j < HIDDEN_NEURONS; j++)
                b_hidden[j] += LEARNING_RATE * delta_hidden[j];
        }
    }

    printf("Predictions:\n");
    for (int sample = 0; sample < TRAINING_SAMPLES; sample++) {
        double hidden[HIDDEN_NEURONS];
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            hidden[j] = b_hidden[j];
            for (int i = 0; i < INPUT_NEURONS; i++)
                hidden[j] += X[sample][i] * w_input_hidden[i][j];
            hidden[j] = sigmoid(hidden[j]);
        }

        double output = b_output;
        for (int j = 0; j < HIDDEN_NEURONS; j++)
            output += hidden[j] * w_hidden_output[j];
        output = sigmoid(output);

        printf("Input: [%g, %g] -> Output: %g\n", X[sample][0], X[sample][1], output);
    }

    return 0;
}
