
import numpy as np                           # import numpy for numerical operations
import nnfs                                  # import nnfs to get the spiral dataset and consistent defaults
from nnfs.datasets import spiral_data        # import the spiral dataset generator that you used earlier

nnfs.init()                                  # initialize nnfs (sets float type and random seed behaviour)
np.random.seed(0)                            # set numpy RNG seed for reproducible results

# ---------------------------
# Dense (fully connected) layer
# ---------------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights with small random numbers (normal distribution) scaled by 0.1
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # initialize biases as zeros (1 row, n_neurons columns)
        self.biases = np.zeros((1, n_neurons))
        # placeholders for gradients that will be set during backprop
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        # placeholder to store inputs that will be needed in backward pass
        self.inputs = None

    def forward(self, inputs):
        # store inputs to use during the backward pass
        self.inputs = inputs
        # compute linear combination: inputs dot weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # gradient of weights is inputs^T dot dvalues
        self.dweights = np.dot(self.inputs.T, dvalues)
        # gradient of biases is sum of dvalues along samples axis (keep dims to match biases shape)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # gradient w.r.t. inputs to pass to previous layer (dvalues dot weights^T)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ---------------------------
# ReLU activation
# ---------------------------
class Activation_ReLU:
    def __init__(self):
        # store output and dinputs placeholders
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        # ReLU: output is max(0, input) elementwise
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # make a copy of incoming gradients
        self.dinputs = dvalues.copy()
        # zero gradient where output was zero or negative (i.e. where input <= 0)
        self.dinputs[self.output <= 0] = 0

# ---------------------------
# Softmax activation
# ---------------------------
class Activation_Softmax:
    def __init__(self):
        # store the output probabilities for use or debugging
        self.output = None

    def forward(self, inputs):
        # subtract max per sample for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize by sum per sample to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # store probabilities as output
        self.output = probabilities

# ---------------------------
# Loss base class
# ---------------------------
class Loss:
    def calculate(self, output, y):
        # compute sample losses via subclass implementation
        sample_losses = self.forward(output, y)
        # return mean loss across samples
        data_loss = np.mean(sample_losses)
        return data_loss

# ---------------------------
# Categorical Cross-Entropy Loss
# ---------------------------
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in batch
        samples = len(y_pred)
        # clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # if labels are single integers (sparse)
        if len(y_true.shape) == 1:
            # select the probabilities for the correct classes
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # if labels are one-hot encoded
        elif len(y_true.shape) == 2:
            # sum across classes to get the probability of correct class for each sample
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # negative log likelihoods for every sample
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# ---------------------------
# Combined Softmax activation and Categorical Cross-Entropy loss
# (provides an efficient backward implementation)
# ---------------------------
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        # create instances for convenience (we only need softmax forward here)
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        # placeholders for outputs and gradients
        self.output = None
        self.dinputs = None

    def forward(self, inputs, y_true):
        # forward pass through softmax activation
        self.activation.forward(inputs)
        # store output probabilities for backward or inspection
        self.output = self.activation.output
        # compute and return loss value using stored probabilities and true labels
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        # if labels are one-hot encoded, convert to class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # copy dvalues (which we expect to be the softmax output probabilities)
        self.dinputs = dvalues.copy()
        # subtract 1 from the probabilities of the correct classes
        self.dinputs[range(samples), y_true] -= 1
        # normalize by the number of samples (averaging gradients over batch)
        self.dinputs = self.dinputs / samples

# ---------------------------
# Simple Stochastic Gradient Descent (SGD) optimizer
# ---------------------------
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_momentums = {}
        self.bias_momentums = {}

    def update_params(self, layer):

        if layer not in self.weight_momentums:
            self.weight_momentums[layer] = np.zeros_like(layer.weights)
            self.bias_momentums[layer] = np.zeros_like(layer.biases)

        weight_updates = self.momentum * self.weight_momentums[layer] - self.learning_rate * layer.dweights
        bias_updates   = self.momentum * self.bias_momentums[layer] - self.learning_rate * layer.dbiases

        self.weight_momentums[layer] = weight_updates
        self.bias_momentums[layer] = bias_updates

        layer.weights += weight_updates
        layer.biases  += bias_updates

# ---------------------------
# Utility: calculate accuracy for predictions (sparse integer labels)
# ---------------------------
def calculate_accuracy(predictions, y_true):
    # predictions are indices (argmax of softmax)
    # compute mean of correct predictions
    return np.mean(predictions == y_true)

# ---------------------------
# Prepare data (spiral dataset)
# ---------------------------
# X shape: (samples, 2) because spiral_data returns 2D inputs
# y shape: (samples,) integer class labels
X, y = spiral_data(samples=100, classes=3)  # generate spiral dataset with 100 samples per class (total 300 samples)

# ---------------------------
# Create network layers
# ---------------------------
dense1 = Layer_Dense(2, 64)                      # first dense layer: 2 inputs -> 3 neurons
activation1 = Activation_ReLU()                 # ReLU activation for first layer
dense2 = Layer_Dense(64, 3)                      # second dense layer: 3 inputs -> 3 neurons (class scores)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()  # combined softmax + loss
optimizer = Optimizer_SGD(learning_rate=0.01, momentum=0.9)

# ---------------------------
# Training loop
# ---------------------------
# number of epochs (full forward + backward + update passes)
epochs = 10000                                  # run for 10001 epochs; you can reduce for speed

for epoch in range(epochs):
    # ---------- Forward pass ----------
    dense1.forward(X)                            # forward through dense layer 1
    activation1.forward(dense1.output)           # forward through ReLU activation
    dense2.forward(activation1.output)           # forward through dense layer 2 (produces logits)

    # compute loss (softmax + categorical cross-entropy combined)
    loss = loss_activation.forward(dense2.output, y)

    # ---------- Predictions and accuracy ----------
    predictions = np.argmax(loss_activation.output, axis=1)  # convert softmax probabilities to class predictions
    accuracy = calculate_accuracy(predictions, y)            # compute accuracy for this epoch

    # ---------- Backward pass ----------
    # backward through combined softmax and loss layer
    # we feed the probabilities (the outputs of softmax) as dvalues into the combined backward,
    # and the combined class+softmax backward will compute gradient dinputs for layer2
    loss_activation.backward(loss_activation.output, y)

    # backward pass into dense layer 2 using dinputs from loss_activation
    dense2.backward(loss_activation.dinputs)

    # backward pass through ReLU using gradient from dense2
    activation1.backward(dense2.dinputs)

    # backward pass into dense layer 1 using gradient from ReLU
    dense1.backward(activation1.dinputs)

    # ---------- Update weights ----------
    optimizer.update_params(dense1)               # update parameters of dense1 with SGD
    optimizer.update_params(dense2)               # update parameters of dense2 with SGD

    # ---------- Logging ----------
    # print progress every 1000 epochs (and the first epoch)
    if epoch % 1000 == 0:
        # print epoch number, accuracy (3 d.p.), and loss (3 d.p.)
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")

# ---------------------------
# Final evaluation: print final loss and accuracy
# ---------------------------
print("Training complete.")
print(f"Final loss: {loss:.4f}")
print(f"Final accuracy: {accuracy:.4f}")
# If you want to inspect final weights, you can print them:
# print("Dense1 weights:\n", dense1.weights)
# print("Dense1 biases:\n", dense1.biases)
# print("Dense2 weights:\n", dense2.weights)
# print("Dense2 biases:\n", dense2.biases)
