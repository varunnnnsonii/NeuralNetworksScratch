# **Advanced Neural Network From Scratch (Training + Regularization + Optimization)**

This project implements a **fully-trainable neural network from scratch** using only **NumPy** and the classic **spiral dataset**.
It extends the simple forward-pass version into a **complete training framework**, including:

✓ Backpropagation
✓ L2 regularization
✓ Dropout
✓ Momentum-based SGD optimizer
✓ Combined Softmax + Cross-Entropy
✓ Mini-batch training
✓ Train/validation split
✓ Model saving

The implementation closely follows deep-learning fundamentals used by PyTorch/TensorFlow — but coded entirely by hand for learning clarity.

---

##  **Features**

### **Core Model Components**

* Fully connected layers with trainable weights
* ReLU activation
* Softmax activation
* Softmax + Cross-Entropy combined backward pass (fast & stable)
* L2 weight/bias regularization
* Dropout for anti-overfitting

### **Training System**

* Mini-batch processing
* Momentum-based SGD optimizer
* Learning-rate decay
* Forward + backward propagation
* Loss tracking
* Accuracy tracking
* Train/validation split

### **Stability & Performance**

* Numerically stable Softmax
* Clipped log values for loss
* Momentum buffers
* Weight decay via L2 regularization

---

##  **Project Structure**

```
project/
│
├── best_implementation.py     # Full training pipeline
├── dataset.py                 # Spiral dataset logic (NNFS-style)
├── README.md                  # This documentation
└── theory.md                  # Mathematical & conceptual explanation
```

---

##  **Installation**

```bash
pip install numpy nnfs matplotlib
```

---

##  **How Training Works**

1. **Generate dataset**
2. **Shuffle + Split** into training (80%) and validation (20%)
3. **Forward pass** through:

   ```
   Dense → ReLU → Dropout → Dense → Softmax
   ```
4. **Compute loss** using Softmax + Cross-Entropy
5. **Backward pass** through all components
6. **Optimizer update** (SGD + momentum + decay)
7. Repeat for N epochs

The model improves accuracy gradually and avoids overfitting with dropout + L2 regularization.

---

##  **Typical Training Output**

Example:

```
epoch 0 | train acc 0.333 loss 1.096 | val acc 0.332 loss 1.097 | lr 0.05000
epoch 200 | train acc 0.85 loss 0.412 | val acc 0.82 loss 0.431 | lr 0.04762
...
```

---

##  **Model Saving**

Weights are stored as:

```
model.npz
 ├ w1, b1
 └ w2, b2
```

Load with:

```python
data = np.load("model.npz")
```

---

##  **Extend This Project**

* Add **Adam optimizer**
* Add **BatchNorm**
* Add **learning-rate scheduling visualizer**
* Train on other datasets (moons, circles, MNIST)
* Build a full class-based Model API

---

##  **License**

MIT License — free for learning, research, and modification.

---


### ****~ Varun D. Soni****