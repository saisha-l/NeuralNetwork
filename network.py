import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0], [1], [1], [0]])

input_neurons = 2
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.5
epochs = 10000

np.random.seed(42)
w_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
w_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
b_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
b_output = np.random.uniform(-1, 1, (1, output_neurons))

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, w_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, w_hidden_output) + b_output
    final_output = sigmoid(final_input)

    # Compute error
    error = y - final_output

    # Backpropagation
    delta_output = error * sigmoid_derivative(final_output)
    delta_hidden = delta_output.dot(w_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    w_hidden_output += hidden_output.T.dot(delta_output) * learning_rate
    b_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    w_input_hidden += X.T.dot(delta_hidden) * learning_rate
    b_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

print("Neural Network Predictions:")
hidden_output = sigmoid(np.dot(X, w_input_hidden) + b_hidden)
final_output = sigmoid(np.dot(hidden_output, w_hidden_output) + b_output)
for i, x in enumerate(X):
    print(f"Input: {x} -> Output: {final_output[i][0]:.4f}")
