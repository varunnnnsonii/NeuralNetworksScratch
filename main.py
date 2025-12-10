import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, l2w=0.0, l2b=0.0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.l2w = l2w
        self.l2b = l2b

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.l2w > 0:
            self.dweights += 2 * self.l2w * self.weights
        if self.l2b > 0:
            self.dbiases += 2 * self.l2b * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = (np.random.rand(*inputs.shape) < self.rate) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Activation_Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)

class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct = y_pred[np.arange(samples), y_true]
        else:
            correct = np.sum(y_pred * y_true, axis=1)
        return -np.log(correct)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = dvalues.shape[1]
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs /= samples

class Activation_Softmax_Loss_CCE:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return np.mean(self.loss.forward(self.output, y_true))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[np.arange(samples), y_true] -= 1
        self.dinputs /= samples

class Optimizer_SGD:
    def __init__(self, lr=1.0, decay=0.0, momentum=0.0):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        self.wm = {}
        self.bm = {}

    def pre_update(self):
        if self.decay:
            self.current_lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))

    def update(self, layer):
        if layer not in self.wm:
            self.wm[layer] = np.zeros_like(layer.weights)
            self.bm[layer] = np.zeros_like(layer.biases)

        w_update = self.momentum * self.wm[layer] - self.current_lr * layer.dweights
        b_update = self.momentum * self.bm[layer] - self.current_lr * layer.dbiases

        self.wm[layer] = w_update
        self.bm[layer] = b_update

        layer.weights += w_update
        layer.biases += b_update

    def post_update(self):
        self.iterations += 1

def accuracy(preds, y):
    return np.mean(preds == y)

X, y = spiral_data(samples=300, classes=3)

n = len(X)
idx = np.arange(n)
np.random.shuffle(idx)

val_split = int(n * 0.2)
X_val = X[idx[:val_split]]
y_val = y[idx[:val_split]]
X_train = X[idx[val_split:]]
y_train = y[idx[val_split:]]

dense1 = Layer_Dense(2, 64, l2w=5e-4, l2b=5e-4)
act1 = Activation_ReLU()
drop1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(64, 3)
loss_act = Activation_Softmax_Loss_CCE()
opt = Optimizer_SGD(lr=0.05, decay=5e-4, momentum=0.9)

epochs = 20000
batch = 64

for epoch in range(epochs):

    opt.pre_update()

    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]

    for start in range(0, len(X_train), batch):
        end = start + batch
        xb = X_train[start:end]
        yb = y_train[start:end]

        dense1.forward(xb)
        act1.forward(dense1.output)
        drop1.forward(act1.output)
        dense2.forward(drop1.output)

        loss = loss_act.forward(dense2.output, yb)

        preds = np.argmax(loss_act.output, axis=1)

        loss_act.backward(loss_act.output, yb)
        dense2.backward(loss_act.dinputs)
        drop1.backward(dense2.dinputs)
        act1.backward(drop1.dinputs)
        dense1.backward(act1.dinputs)

        opt.update(dense1)
        opt.update(dense2)

    opt.post_update()

    if epoch % 200 == 0:
        dense1.forward(X_train)
        act1.forward(dense1.output)
        dense2.forward(act1.output)
        train_loss = loss_act.forward(dense2.output, y_train)
        train_preds = np.argmax(loss_act.output, axis=1)
        train_acc = accuracy(train_preds, y_train)

        dense1.forward(X_val)
        act1.forward(dense1.output)
        dense2.forward(act1.output)
        val_loss = loss_act.forward(dense2.output, y_val)
        val_preds = np.argmax(loss_act.output, axis=1)
        val_acc = accuracy(val_preds, y_val)

        print(
            f"epoch {epoch} | "
            f"train acc {train_acc:.3f} loss {train_loss:.3f} | "
            f"val acc {val_acc:.3f} loss {val_loss:.3f} | "
            f"lr {opt.current_lr:.5f}"
        )

np.savez("model.npz",
         w1=dense1.weights, b1=dense1.biases,
         w2=dense2.weights, b2=dense2.biases)

print("Model saved!")
