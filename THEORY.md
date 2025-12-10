# **Neural Network Theory — Mathematical Foundations & Visual Representations**

This document provides the full **mathematical**, **conceptual diagrams**, **computational flow**, and **visual explanations** behind the neural network implementation used in the project. It is written as a complete, professionally structured **Theory.md**.

---

# **1. Dense Layer**

A **Dense (Fully Connected) Layer** performs the transformation:

[
\mathbf{z} = \mathbf{X} \mathbf{W} + \mathbf{b}
]

Where:

* **X**: Input matrix of shape **(N × D)**
* **W**: Weight matrix **(D × H)**
* **b**: Bias vector **(1 × H)**
* **z**: Output **(N × H)**

### **Neuron Operation**

[
z_j = \sum_{i=1}^{D} x_i w_{ij} + b_j
]

### **Diagram — Dense Layer Mapping**

```
Input (2 features)      Weights (2×3)         Output (3 neurons)
   [x1, x2]  ------>  [ w11 w12 w13 ]  --->  [ z1 z2 z3 ]
                      [ w21 w22 w23 ]
```

---

# **2. ReLU Activation**

The **Rectified Linear Unit** introduces non-linearity:

[
\text{ReLU}(z) = \max(0, z)
]

### **Diagram — ReLU Shape**

```
      |
   z  |         ______
      |        /
      |_______/
      0       z
```

### **Backpropagation (Derivative)**

[
\frac{d}{dz} \text{ReLU}(z) =
\begin{cases}
1, & z>0 \
0, & z \le 0
\end{cases}
]

---

# **3. Dropout Regularization**

Dropout randomly disables neurons during training:

[
\text{output} = x \cdot \text{mask}
]
Where mask is sampled as:
[
\text{mask} = \frac{\text{Bernoulli}(p)}{p}
]
Ensuring expected value consistency during inference.

### **Diagram — Dropout Mechanics**

```
Input Layer → [1, 0, 1, 1, 0] dropout mask
               ↓   ↓   ↓
Output Layer → Some neurons removed during pass
```

---

# **4. Softmax Activation**

Softmax converts raw logits into **probabilities**:

[
P_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
]

### **Numerical Stability Fix**

[
P_i = \frac{e^{z_i - \max(z)}}{\sum_{j} e^{z_j - \max(z)}}
]

### **Softmax Flow Diagram**

```
Raw Scores     Shifted logits      Exponentiate    Normalize
 [4, 8, 2]  →  [-4, 0, -6]   →  [0.018, 1, 0.002] → Probabilities
                                            ↓
                                  [0.017, 0.980, 0.002]
```

---

# **5. Categorical Cross‑Entropy Loss**

For a single sample:
[
L = -\log(p_\text{correct})
]

### **Integer Labels**

[
L = -\log(p[y])
]

### **One‑hot Labels**

[
L = -\sum_{i=1}^{K} y_i \log(p_i)
]

### **Batch Loss**

[
L_\text{mean} = \frac{1}{N} \sum_{n=1}^N L_n
]

### **Diagram — Loss Computation**

```
Softmax Output: [0.7, 0.2, 0.1]
True Label: class 0
Loss = -log(0.7) = 0.3566
```

---

# **6. Combined Softmax + Loss Optimization**

During backprop:

### **Gradient Simplification**

[
\frac{\partial L}{\partial z_i} = P_i - y_i
]
This avoids computing separate Softmax and Loss derivatives.

### **Diagram — Simplification**

```
Dense → Softmax → Loss     becomes
Dense → (Softmax+Loss combined)
```

---

# **7. SGD Optimizer Theory (with Momentum & Decay)**

### **Learning Rate Decay**

[
\eta_t = \frac{\eta_0}{1 + d\cdot t}
]

### **Momentum Update**

[
v_t = \beta v_{t-1} - \eta g_t
]
[
w_t = w_{t-1} + v_t
]

### **Diagram — Optimization Trajectory**

```
Loss Surface:

Without Momentum → zig‑zag downhill
With Momentum    → smooth curved path
```

---

# **8. Backpropagation Through Each Layer**

### **Dense Layer**

[
\frac{\partial L}{\partial W} = X^T \delta
]
[
\frac{\partial L}{\partial b} = \sum \delta
]
[
\frac{\partial L}{\partial X} = \delta W^T
]

### **ReLU**

Zero gradient where input (z ≤ 0).

### **Dropout**

Gradient only flows through active neurons:
[
\delta = \delta \cdot \text{mask}
]

### **Softmax + CCE**

[
\delta_i = P_i - y_i
]

---

# **9. Full Forward Pass Pipeline**

```
           Input (X)
                │
                ▼
      ┌──────────────────┐
      │ Dense Layer 1    │  (XW+b)
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ ReLU Activation  │
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Dropout Layer    │
      └──────────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Dense Layer 2    │
      └──────────────────┘
                │
                ▼
      ┌─────────────────────────┐
      │ Softmax + CCE Combined │
      └─────────────────────────┘
```

---

# **10. Full Backpropagation Pipeline**

```
Loss Gradient
     │
     ▼
Softmax+CCE Combined
     │
     ▼
Dense 2 ← ReLU Mask from prev layer
     │
     ▼
Dropout (masked gradient)
     │
     ▼
Dense 1
```

### **10.1 Computational Graph for a Dense Layer**

```
X ---> [Multiply] ---> XW ---> [Add Bias] ---> Z ---> [Activation] ---> A
                ^                             ^
                |                             |
              dW = X^T * dZ                db = sum(dZ)
```

### **10.2 Full Neural Network Backprop Graph**

```
Input → Dense1 → ReLU → Dense2 → Softmax → Loss
  |         |       |         |         |
 dL/dX ← dL/dZ1 ← dL/dA1 ← dL/dZ2 ← dL/dA2
```

---


# **11. Spiral Dataset Visualization (Conceptual)**

```
Class 1 → Spiral A   ) ) )
Class 2 → Spiral B   ( ( (
Class 3 → Spiral C   < < <
```

The model learns to separate these **non‑linear intertwined spirals**, demonstrating the importance of:

* ReLU
* Multi‑layer structure
* Softmax
* Cross‑entropy

---

# **12. Optimization Curves**

### **12.1 Ideal Loss Curve**

```
Loss
 |\
 | \
 |  \
 |   \
 |    \______
 +------------------> Epochs
```

### **12.2 Exploding Gradient Curve**

```
Loss
 |      /-------\
 |     /         \
 |    /           \
 |___/             \____
 +------------------------------> Epochs
```

---

# **13. Training Dynamics**

### **13.1 Vanishing Gradients**

* Happens in deep networks
* Sigmoid/tanh saturation
* Poor initialization

Symptoms: slow learning, flat loss, early layers frozen.

Fixes: ReLU, He init, BatchNorm, Residuals.

### **13.2 Exploding Gradients**

* Very large weight updates
* Model becomes unstable / NaN

Fixes: gradient clipping, smaller LR, normalization.

---

# **14. L2 Regularization Theory**

L2 adds a weight penalty term: L_new = L_original + lambda * sum(W^2)

Effects:

* Prevents large weights
* Reduces overfitting
* Smooths boundaries

Gradient update adds: 2 * lambda * W

---

# **15. Momentum Optimizer Diagram**

Momentum accumulates a velocity term.

```
Gradient → [Momentum] → Velocity → Updated Weights
```

### **15.1 Ball Rolling Analogy**

```
     * (fast)
    *
   *
  *
 *_____ (min)
```

### **15.2 Equations (text-safe)**

Velocity: v_t = beta * v_(t-1) + (1 - beta) * g_t
Weights:  W = W - lr * v_t

Advantages: faster convergence, less oscillation, smoother descent.

# **16. Summary of Mathematical Formulas**

### **Dense Layer**

[ Z = XW + b ]

### **ReLU**

[ A = \max(0,Z) ]

### **Dropout**

[ A = X \cdot M ]

### **Softmax**

[ P_i = \frac{e^{z_i}}{\sum_j e^{z_j}} ]

### **Cross‑Entropy Loss**

[ L = -\log(P_\text{correct}) ]

### **SGD Momentum**

[ v = \beta v - \eta g ]
[ w = w + v ]

---
### ****~ Varun D. Soni****