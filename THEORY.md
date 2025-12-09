# **Neural Network Theory — Mathematical Foundations & Conceptual Diagrams**

This document presents the theoretical foundations behind the implementation in this project. It includes core formulas, conceptual diagrams, and mathematical reasoning behind dense layers, activations, softmax, and loss functions.

---

# **1. Dense Layer Theory**

A **Dense (Fully Connected) Layer** performs the transformation:

[
\mathbf{z} = \mathbf{X} \mathbf{W} + \mathbf{b}
]

Where:

* ( \mathbf{X} \in \mathbb{R}^{N \times D} ) — input batch (N samples, D features)
* ( \mathbf{W} \in \mathbb{R}^{D \times H} ) — weight matrix (H = neurons)
* ( \mathbf{b} \in \mathbb{R}^{1 \times H} ) — bias vector
* ( \mathbf{z} \in \mathbb{R}^{N \times H} ) — output of the linear transformation

### **Diagram — Dense Layer**

```
Input (2 features)      Weights (2×3)         Output (3 neurons)
   [x1, x2]  ------>  [ w11 w12 w13 ]  --->  [ z1 z2 z3 ]
                      [ w21 w22 w23 ]
```

Each output neuron computes:
[
z_j = x_1 w_{1j} + x_2 w_{2j} + b_j
]

---

# **2. ReLU Activation Theory**

Rectified Linear Unit:
[
\text{ReLU}(z) = \max(0, z)
]

### **Diagram — ReLU**

```
      |
   z  |         ______
      |        /
      |_______/
      0       z
```

ReLU filters negative values → useful for non-linearity.

---

# **3. Softmax Activation Theory**

Softmax converts raw scores (logits) into probabilities:

[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
]

### **Numerical Stability Correction**

The stable version subtracts the maximum logit value:

[
\text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
]

### **Diagram — Softmax**

```
Raw Scores     Shifted (Stable)     Exponentiate     Normalize
 [4, 8, 2]  →  [−4, 0, −6]   →   [0.018, 1, 0.002]  → Probabilities
                                             ↓
                                   [0.017, 0.980, 0.002]
```

---

# **4. Categorical Cross‑Entropy Loss Theory**

For a single prediction:
[
L = -\log(p_{\text{correct}})
]

Where ( p_{correct} ) is the probability assigned to the true class.

### **For integer labels:**

[
L = -\log(y_{pred}[c])
]

### **For one‑hot labels:**

[
L = - \sum_{i=1}^K y_i \log(p_i)
]

### **Batch Loss:**

[
L_{mean} = \frac{1}{N} \sum_{n=1}^N L_n
]

### **Diagram — Loss Process**

```
Softmax Output: [0.7, 0.2, 0.1]
True Label: class 0
Loss = -log(0.7) = 0.3566
```

---

# **5. Forward Pass Pipeline (Complete System Diagram)**

```
           Input (X)
                │
                ▼
      ┌──────────────────┐
      │ Dense Layer 1    │  (XW + b)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ ReLU Activation  │
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Dense Layer 2    │
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Softmax Output   │
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Cross-Entropy    │
      └──────────────────┘
```

---

# **6. Mathematical Summary for Reference**

### **Dense Layer:**

[
Z = XW + b
]

### **ReLU:**

[
A = \max(0, Z)
]

### **Softmax:**

[
P_i = \frac{e^{Z_i - \max(Z)}}{\sum_{j} e^{Z_j - \max(Z)}}
]

### **Categorical Cross‑Entropy:**

[
L = -\log(P_{correct})
]

### **Final Mean Loss:**

[
L_{mean} = \frac{1}{N} , \sum L_n
]

---
~Varun D Soni