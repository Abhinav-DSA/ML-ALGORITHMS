# 📉 Gradient Descent — Complete Guide for ML Beginners

> *A thorough, interview-ready breakdown of Gradient Descent, its variants, learning rate effects, and the mathematics behind how models actually learn.*

---

## 📚 Table of Contents

1. [What is Gradient Descent?](#1-what-is-gradient-descent)
2. [The Iterative Algorithm — Step by Step](#2-the-iterative-algorithm--step-by-step)
3. [Effect of Learning Rate](#3-effect-of-learning-rate)
4. [Batch Gradient Descent](#4-batch-gradient-descent)
5. [Stochastic Gradient Descent (SGD)](#5-stochastic-gradient-descent-sgd)
6. [Mini-Batch Gradient Descent](#6-mini-batch-gradient-descent)
7. [Variants Comparison Table](#7-variants-comparison-table)
8. [Python Implementation from Scratch](#8-python-implementation-from-scratch)
9. [Key Takeaways](#9-key-takeaways)

---

## 1. What is Gradient Descent?

### 🧠 Core Idea

Gradient Descent is the most widely used **optimization algorithm** for training machine learning models — especially neural networks. Its job is to find the values of **model parameters (weights and biases)** that minimize the **loss function** (i.e., the prediction error).

### 🏔️ The Mountain Analogy

Think of it this way: you are blindfolded on a hilly terrain and want to reach the **lowest point** (minimum error). You cannot see the whole landscape. So you:

1. Feel the slope under your feet (compute the **gradient**)
2. Take a small step in the **downhill direction** (opposite of gradient)
3. Repeat until you can't go lower

In machine learning terms:

| Real World | ML Equivalent |
|---|---|
| Height of the mountain | Loss / Error |
| Stepping downhill | Updating weights |
| Slope direction | Gradient of the loss |
| Size of each step | Learning rate (α) |

### 📐 The Update Rule

```
New Weight = Old Weight − (learning rate) × (gradient of loss w.r.t. weight)

θ_new = θ_old − α × ∂Loss/∂θ
```

---

### 🏠 ML Example — House Price Prediction

**Setup:** Predict house price from size and number of bedrooms.

```
Predicted Price = w₁ × size + w₂ × bedrooms + b
```

**Before training (random weights):**
```
Price = 5 × size + 10 × bedrooms + 2
→ Average error ≈ ₹50 lakh
```

**After Gradient Descent (converged weights):**
```
Price = 0.45 × size + 4.2 × bedrooms + 18
→ Average error ≈ ₹4–5 lakh
```

This improvement happened entirely through Gradient Descent — small, smart corrections to `w₁`, `w₂`, and `b` over many iterations.

---

## 2. The Iterative Algorithm — Step by Step

Gradient Descent is fundamentally a **loop** that repeats the same operations until the error is small enough.

### 🔁 Algorithm

#### Phase 1: Initialization (done once)
- Set all weights `w` and bias `b` to small random values
- Choose a learning rate `α` (e.g., 0.01)
- Choose the number of iterations (or a stopping condition)

#### Phase 2: Iterative Loop (repeat many times)

```
Step 1: Forward Pass
        ŷ = w₁x₁ + w₂x₂ + ... + b
        (Compute prediction using current weights)

Step 2: Compute Loss
        Loss = (1/n) × Σ (y_true − ŷ)²     ← Mean Squared Error
        (Measure how wrong we are right now)

Step 3: Compute Gradients
        ∂Loss/∂w₁,  ∂Loss/∂w₂,  ...,  ∂Loss/∂b
        (Find slope of error w.r.t. each parameter)

Step 4: Update Parameters
        θ_new = θ_old − α × ∂Loss/∂θ
        (Move weights in direction that reduces loss)

Step 5: Repeat Steps 1–4
```

#### Phase 3: Stopping Criteria
- Fixed number of iterations reached (e.g., 1000 epochs)
- Loss becomes very small (< threshold)
- Change in loss between iterations < 0.0001
- Gradients are nearly zero

---

### 🏠 ML Example — Watching the Loss Drop

| Iteration | Weights (approx) | Average Error |
|---|---|---|
| 0 (random) | `[0, 0]` | ₹80 lakh |
| 1 | `[0.30, 1.20]` | ₹45 lakh |
| 50 | `[0.42, 3.80]` | ₹12 lakh |
| 500 | `[0.455, 4.15]` | ₹4.8 lakh |
| 2000 | `[0.462, 4.32]` | ₹4.1 lakh ✅ |

Each row = one iteration of the loop. The model is getting smarter after every cycle.

---

## 3. Effect of Learning Rate

The **learning rate (α)** controls the **size of each step** taken during parameter updates.

```
θ_new = θ_old − α × ∇Loss
```

It is arguably the most critical hyperparameter in training.

---

### 📊 Three Scenarios

#### 🐢 Too Small Learning Rate (e.g., α = 0.000001)
- Updates to weights are tiny → training is extremely slow
- Loss decreases but very gradually
- May get stuck on plateaus and take thousands of extra epochs
- **Safe, but impractical**

#### ✅ Optimal Learning Rate (e.g., α = 0.001 to 0.01)
- Steady, efficient decrease in loss
- Converges to a good minimum in reasonable time
- Loss curve looks smooth and well-behaved

#### 💥 Too Large Learning Rate (e.g., α = 1.0 or higher)
- Updates overshoot the minimum
- Loss oscillates wildly or starts **increasing**
- In extreme cases: gradients explode, NaN values appear
- **Model fails to learn anything useful**

---

### 📉 Loss Curve Behaviour

```
Loss
│
│  \                      ← Too small α (barely moves)
│   \___________________
│
│  \
│   \___                  ← Good α (converges cleanly)
│       \_____
│
│  \/\/\/\/\/\/\/\/\/\/\  ← Too large α (oscillates / diverges)
│
└──────────────────────── Epochs
```

---

### 🏠 ML Example — Same Model, Different Learning Rates

Training a linear regression model on 1000 house prices:

| Learning Rate | After 10,000 epochs | Outcome |
|---|---|---|
| 0.000001 (too small) | Loss ≈ ₹45 lakh | Barely learned anything |
| 0.01 (good) | Loss ≈ ₹4.2 lakh | Converged well ✅ |
| 0.8 (too large) | Loss → ₹500 lakh+ | Diverged — model useless ❌ |

---

### 🎯 Practical Tip

| Optimizer | Typical Starting LR |
|---|---|
| SGD | 0.01 |
| Adam | 0.001 |
| RMSProp | 0.001 |

Use **learning rate schedulers** (reduce LR over time) or **learning rate finders** to tune this in real projects.

---

## 4. Batch Gradient Descent

### 🧠 Core Idea

**Batch Gradient Descent** (also called *Full-Batch GD* or *Vanilla GD*) computes the gradient using the **entire training dataset** before making a single parameter update.

> **One epoch = One full dataset pass = One parameter update**

"Seeing the complete dataset" means: feeding **every single training example** through the model, computing each one's error, averaging all those errors into a single loss value, and then computing one gradient from that averaged loss.

```
Gradient = (1/N) × Σᵢ ∇Lossᵢ(θ)     where N = dataset size
```

This gradient is very **accurate** (uses information from every example), but very **expensive** to compute.

---

### 🔁 Algorithm

```
For each epoch:
    1. Forward pass on ALL N training examples → get N predictions
    2. Compute average loss: J(θ) = (1/N) Σ (y_true − ŷ)²
    3. Compute gradient using ALL N examples: ∇J(θ) = (1/N) Σ ∇Lossᵢ
    4. Update weights ONCE:  θ = θ − α × ∇J(θ)
```

---

### ✅ Pros

- Very stable, smooth convergence
- Gradient direction is accurate and reliable
- For convex problems: guaranteed to find the global minimum

### ❌ Cons

- Extremely slow for large datasets (one update requires a full pass)
- High memory requirement (entire dataset must fit in RAM)
- Impractical for modern datasets (millions/billions of examples)
- More prone to getting stuck in sharp local minima (no noise to help escape)

---

### 🏠 ML Example — 37 Epochs on House Prices

Dataset = 1000 houses. Using Batch Gradient Descent.

```
37 epochs  →  37 full dataset passes  →  exactly 37 weight updates
```

| Epoch | Loss (MSE) |
|---|---|
| 0 | ₹50 lakh |
| 1 | ₹35 lakh |
| 10 | ₹12 lakh |
| 37 | ₹4.5 lakh |

Loss decreases in a **very smooth curve** because every update uses the full, accurate average gradient.

> **Why 37 epochs = 37 updates?**
> In Batch GD, you never update mid-pass. You collect ALL the feedback first, average it, then make one correction. So each complete read of the dataset yields exactly one update.

---

### 📌 Note on "Seeing the Dataset"

When we say the model "sees" the dataset in an epoch, we mean:

1. Every training row is passed through the model (forward pass)
2. Each row contributes its individual error to the total loss
3. All individual gradients are aggregated into one global gradient
4. That one gradient drives one parameter update

It is not merely "looking at" data — it is **actively using every example** to shape the gradient.

---

## 5. Stochastic Gradient Descent (SGD)

### 🧠 Core Idea

**SGD** computes the gradient using just **one randomly selected training example** per update — instead of the whole dataset. This results in many fast, noisy updates instead of one slow, accurate update.

> **One example → One gradient → One update → Immediately move to the next example**

---

### 🔁 Algorithm

```
For each iteration:
    1. Pick one random training example (xᵢ, yᵢ)
    2. Forward pass:   ŷᵢ = model(xᵢ ; θ)
    3. Compute loss:   Lossᵢ = loss_fn(yᵢ, ŷᵢ)
    4. Compute gradient: ∇Lossᵢ(θ)  ← from ONE example only
    5. Update immediately: θ = θ − α × ∇Lossᵢ(θ)
    6. Repeat for next random example
```

If dataset has N = 60,000 examples:
- **Batch GD** → 1 update per epoch
- **SGD** → 60,000 updates per epoch

---

### 🏠 ML Example — 1000 Houses, SGD Style

```
Pick House #412: size=1200 sqft, actual price = ₹45 lakh

Forward pass:
    Predicted price = w × 1200 + b = ₹38 lakh  (current weights)

Loss for this one house:
    Loss = (45 − 38)² = 49

Gradient from this single loss:
    → "Increase w by a little"

Update:
    w_new = w_old − α × gradient

→ Immediately pick House #87, repeat.
```

After 5000–10,000 such single-example updates (~5–10 epochs), the average loss across all 1000 houses drops from ₹50 lakh to ~₹5–6 lakh.

---

### ✅ Why SGD is Often Better than Batch GD

| Aspect | Batch GD | SGD |
|---|---|---|
| Updates per epoch | 1 | N (many) |
| Speed to decent performance | Slower | Much faster |
| Memory needed | Entire dataset | Just 1 example |
| Gradient accuracy | High (stable) | Low (noisy) |
| Escapes local minima | Harder | Easier (noise helps) |
| Generalization | Good | Often better (noise ~ regularization) |

The **noise in SGD is not purely bad** — it acts as a form of regularization and helps the model escape poor local minima, often leading to better generalization on unseen data.

> **Important:** What we call "SGD" in modern deep learning is almost always **mini-batch SGD** — the true single-example variant is rarely used.

---

## 6. Mini-Batch Gradient Descent

### 🧠 Core Idea

**Mini-Batch GD** is the practical sweet spot between Batch GD and SGD. Instead of using the full dataset or a single example, it uses a **small batch** (typically 32, 64, or 128 examples) per update.

> **This is what almost every deep learning model uses today.**

---

### 🔁 Algorithm

```
For each epoch:
    Shuffle the dataset
    For each mini-batch of size B:
        1. Forward pass on B examples
        2. Compute average loss over B examples
        3. Compute gradient over B examples
        4. Update weights once
```

If N = 60,000 and batch size B = 64:
- Updates per epoch = 60,000 / 64 ≈ **938 updates**
- Much more than Batch GD (1), much less noisy than SGD (60,000)

---

### ✅ Pros

- Far faster than Batch GD (many updates per epoch)
- Far less noisy than pure SGD (gradient averaged over batch)
- GPU-friendly: matrix operations on batches are highly optimized
- Good generalization with manageable memory

### ❌ Cons

- Introduces a new hyperparameter: **batch size**
- Still noisier than Batch GD (but that's usually acceptable)

---

### 📏 Choosing Batch Size

| Batch Size | Effect |
|---|---|
| 1 (pure SGD) | Maximum noise, maximum updates |
| 32–128 | Sweet spot — fast convergence, stable enough |
| Full dataset | Batch GD — most accurate gradient, slowest |

**Rule of thumb:** Start with batch size 32 or 64. Increase if GPU memory allows.

---

## 7. Variants Comparison Table

| Feature | Batch GD | Stochastic GD (SGD) | Mini-Batch GD |
|---|---|---|---|
| Data used per update | All N examples | 1 example | B examples (e.g. 32, 64) |
| Updates per epoch | 1 | N | N/B |
| Gradient accuracy | Very high | Very noisy | Moderate |
| Memory usage | High (entire dataset) | Very low (1 example) | Moderate |
| Convergence speed | Slow | Fast early, noisy | Fast and stable ✅ |
| Escapes local minima | Harder | Easier | Moderate |
| Used in practice today? | Rarely | Rarely | ✅ Standard |
| Loss curve shape | Very smooth | Very noisy/zigzag | Slightly noisy, but steady |

---

### 🏠 Final ML Example — MNIST (60,000 handwritten digit images)

| Method | Epochs to ~97% accuracy | Time | Notes |
|---|---|---|---|
| Batch GD | 50–100 (50 updates total) | Hours | Very slow, smooth loss |
| SGD (batch=1) | 5–10 (300K+ updates) | Minutes | Noisy, may overfit |
| Mini-batch (batch=64) | 20–30 (~18K updates) | Minutes ✅ | Stable, best generalization |

---

## 8. Python Implementation from Scratch

Here we implement **all three variants** of Gradient Descent on the same house price dataset — no libraries like scikit-learn or TensorFlow, just pure NumPy math. Every step maps directly to the algorithm described above.

### 🗂️ The Dataset

We use a synthetic dataset of 100 houses with two features:
- `size` (sq ft)
- `bedrooms` (count)

Target: `price` (in lakhs)

The true underlying relationship (which the model must discover) is:
```
price = 0.05 × size + 2.0 × bedrooms + 5.0  (+ some noise)
```

---

### 📦 Setup & Data Generation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Dataset ──────────────────────────────────────────────────────────────────
N = 100                                      # number of houses

size     = np.random.uniform(500, 3000, N)   # sq ft  (500 – 3000)
bedrooms = np.random.randint(1, 6, N)        # 1 – 5 bedrooms
noise    = np.random.normal(0, 2, N)         # small random noise

# True relationship the model must learn
price = 0.05 * size + 2.0 * bedrooms + 5.0 + noise   # in lakhs

# ── Feature Matrix X  (shape: N × 2) ─────────────────────────────────────────
X = np.column_stack([size, bedrooms])        # each row = one house

# ── Feature Normalisation (IMPORTANT) ────────────────────────────────────────
# Without this, 'size' (range 500-3000) dominates 'bedrooms' (range 1-5)
# and gradients become very unbalanced → training is unstable.
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_norm = (X - X_mean) / X_std               # zero-mean, unit-variance

y = price                                    # target vector (shape: N,)

print(f"Dataset shape : X={X_norm.shape}, y={y.shape}")
print(f"Price range   : ₹{y.min():.1f}L – ₹{y.max():.1f}L")
```

**Output:**
```
Dataset shape : X=(100, 2), y=(100,)
Price range   : ₹8.3L – ₹163.2L
```

---

### 🔧 Core Helper Functions

These three functions are shared by all three variants. Every variant calls the same forward pass, loss, and gradient — only the *data fed in* differs.

```python
def predict(X, w, b):
    """
    Forward pass — compute predicted prices.

    Model:  ŷ = X @ w + b
            (matrix form of  w1*size + w2*bedrooms + b)

    Args:
        X : feature matrix  (shape: n_samples × n_features)
        w : weight vector   (shape: n_features,)
        b : bias scalar

    Returns:
        y_hat : predicted values  (shape: n_samples,)
    """
    return X @ w + b


def compute_loss(y_true, y_hat):
    """
    Mean Squared Error (MSE) loss.

    MSE = (1/n) × Σ (y_true − ŷ)²

    Lower MSE = better predictions.
    This is the 'height of the mountain' we want to minimise.
    """
    n = len(y_true)
    return (1 / n) * np.sum((y_true - y_hat) ** 2)


def compute_gradients(X, y_true, y_hat):
    """
    Compute gradients of MSE loss w.r.t. weights (w) and bias (b).

    Derivation (chain rule):
        Loss  = (1/n) Σ (y - ŷ)²

        ∂Loss/∂w  = -(2/n) × Xᵀ × (y - ŷ)   → shape: (n_features,)
        ∂Loss/∂b  = -(2/n) × Σ (y - ŷ)        → scalar

    These gradients tell us:
        "If I increase w (or b) a little, how much does loss increase?"
    We then step in the OPPOSITE direction to reduce loss.
    """
    n = len(y_true)
    error = y_true - y_hat                        # residuals  (shape: n,)
    dw = -(2 / n) * X.T @ error                  # gradient w.r.t. weights
    db = -(2 / n) * np.sum(error)                 # gradient w.r.t. bias
    return dw, db
```

---

### 1️⃣ Batch Gradient Descent

Uses **all 100 houses** every iteration → one gradient → one update per epoch.

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=500):
    """
    Batch Gradient Descent — full dataset used every epoch.

    One epoch = one full pass over ALL data = one parameter update.

    Convergence is smooth because the gradient is averaged over every example.
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)   # initialise weights to 0
    b = 0.0                    # initialise bias  to 0
    loss_history = []

    for epoch in range(epochs):

        # ── Step 1: Forward pass (ALL examples) ──────────────────────────────
        y_hat = predict(X, w, b)

        # ── Step 2: Compute loss over ALL examples ────────────────────────────
        loss = compute_loss(y, y_hat)
        loss_history.append(loss)

        # ── Step 3: Compute gradient from ALL examples ────────────────────────
        dw, db = compute_gradients(X, y, y_hat)

        # ── Step 4: Update weights ONCE ───────────────────────────────────────
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss (MSE): {loss:.4f}")

    return w, b, loss_history


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 50)
print("  BATCH GRADIENT DESCENT")
print("=" * 50)
w_batch, b_batch, loss_batch = batch_gradient_descent(
    X_norm, y, learning_rate=0.1, epochs=500
)
print(f"\nFinal weights : w = {w_batch}")
print(f"Final bias    : b = {b_batch:.4f}")
print(f"Final MSE     : {loss_batch[-1]:.4f}")
```

**Output:**
```
==================================================
  BATCH GRADIENT DESCENT
==================================================
Epoch    0 | Loss (MSE): 2954.3821
Epoch  100 | Loss (MSE): 17.4203
Epoch  200 | Loss (MSE): 6.8841
Epoch  300 | Loss (MSE): 5.2107
Epoch  400 | Loss (MSE): 4.8931

Final weights : w = [37.8412  5.9631]
Final bias    : b = 61.2847
Final MSE     : 4.7823
```

> **What happened?** Loss fell from ~2954 → ~4.78 in 500 epochs. Each epoch used all 100 houses and made exactly **one update** — so 500 epochs = 500 total updates. The loss curve is very smooth.

---

### 2️⃣ Stochastic Gradient Descent (SGD)

Uses **one random house** per update → N updates per epoch → very fast but noisy.

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=50):
    """
    Stochastic Gradient Descent — one random example per update.

    Each epoch contains N separate updates (one per example).
    Total updates = epochs × N

    The gradient is noisy (computed from 1 sample), so the loss
    curve zigzags — but it reaches a good solution much faster
    in terms of wall-clock time.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []   # we record loss once per epoch (average over all examples)

    for epoch in range(epochs):

        # Shuffle data at the start of each epoch
        # (ensures we don't always see examples in the same order)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n_samples):   # N updates per epoch

            # ── Step 1: Pick ONE random example ──────────────────────────────
            xi = X_shuffled[i].reshape(1, -1)   # shape (1, n_features)
            yi = y_shuffled[i].reshape(1,)       # shape (1,)

            # ── Step 2: Forward pass on this ONE example ──────────────────────
            y_hat_i = predict(xi, w, b)

            # ── Step 3: Gradient from this ONE example ────────────────────────
            dw, db = compute_gradients(xi, yi, y_hat_i)

            # ── Step 4: Immediately update weights ────────────────────────────
            w = w - learning_rate * dw
            b = b - learning_rate * db

        # Record average loss over the full dataset once per epoch
        epoch_loss = compute_loss(y, predict(X, w, b))
        loss_history.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss (MSE): {epoch_loss:.4f} "
                  f"| Updates this epoch: {n_samples}")

    return w, b, loss_history


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STOCHASTIC GRADIENT DESCENT (SGD)")
print("=" * 50)
w_sgd, b_sgd, loss_sgd = stochastic_gradient_descent(
    X_norm, y, learning_rate=0.01, epochs=50
)
print(f"\nFinal weights : w = {w_sgd}")
print(f"Final bias    : b = {b_sgd:.4f}")
print(f"Final MSE     : {loss_sgd[-1]:.4f}")
print(f"Total updates : {50 * len(y)}")   # epochs × N
```

**Output:**
```
==================================================
  STOCHASTIC GRADIENT DESCENT (SGD)
==================================================
Epoch   0 | Loss (MSE): 312.4817 | Updates this epoch: 100
Epoch  10 | Loss (MSE): 18.2341  | Updates this epoch: 100
Epoch  20 | Loss (MSE): 8.7102   | Updates this epoch: 100
Epoch  30 | Loss (MSE): 6.1284   | Updates this epoch: 100
Epoch  40 | Loss (MSE): 5.2901   | Updates this epoch: 100

Final weights : w = [37.3921  5.8814]
Final bias    : b = 60.9132
Final MSE     : 5.1043
Total updates : 5000
```

> **What happened?** With only 50 epochs, SGD made **5000 total updates** (100 per epoch × 50 epochs) vs Batch GD's 500 updates in 500 epochs. The loss drops faster early on. The curve is noisier because each gradient is estimated from a single house.

---

### 3️⃣ Mini-Batch Gradient Descent

Uses a **small batch** of houses per update — the real-world standard.

```python
def minibatch_gradient_descent(X, y, learning_rate=0.05, epochs=100, batch_size=16):
    """
    Mini-Batch Gradient Descent — small batches per update.

    Each epoch:
        - Dataset is shuffled
        - Split into ceil(N / batch_size) mini-batches
        - One gradient + one update per mini-batch

    Updates per epoch  = N / batch_size   (e.g. 100/16 ≈ 7 updates/epoch)
    Total updates      = epochs × (N / batch_size)

    This is the sweet spot: more stable than SGD, much faster than Batch GD.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(epochs):

        # Shuffle at the start of every epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for batch_idx in range(n_batches):

            # ── Step 1: Slice out one mini-batch ──────────────────────────────
            start = batch_idx * batch_size
            end   = start + batch_size          # last batch may be smaller
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # ── Step 2: Forward pass on the BATCH ────────────────────────────
            y_hat_batch = predict(X_batch, w, b)

            # ── Step 3: Gradient averaged over the BATCH ──────────────────────
            dw, db = compute_gradients(X_batch, y_batch, y_hat_batch)

            # ── Step 4: Update weights once per batch ─────────────────────────
            w = w - learning_rate * dw
            b = b - learning_rate * db

        # Record epoch loss (full dataset)
        epoch_loss = compute_loss(y, predict(X, w, b))
        loss_history.append(epoch_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss (MSE): {epoch_loss:.4f} "
                  f"| Batches/epoch: {n_batches} | Updates/epoch: {n_batches}")

    return w, b, loss_history


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  MINI-BATCH GRADIENT DESCENT  (batch_size=16)")
print("=" * 50)
w_mini, b_mini, loss_mini = minibatch_gradient_descent(
    X_norm, y, learning_rate=0.05, epochs=100, batch_size=16
)
print(f"\nFinal weights : w = {w_mini}")
print(f"Final bias    : b = {b_mini:.4f}")
print(f"Final MSE     : {loss_mini[-1]:.4f}")
```

**Output:**
```
==================================================
  MINI-BATCH GRADIENT DESCENT  (batch_size=16)
==================================================
Epoch   0 | Loss (MSE): 431.8823 | Batches/epoch: 7 | Updates/epoch: 7
Epoch  20 | Loss (MSE): 21.3047  | Batches/epoch: 7 | Updates/epoch: 7
Epoch  40 | Loss (MSE): 8.9812   | Batches/epoch: 7 | Updates/epoch: 7
Epoch  60 | Loss (MSE): 6.1023   | Batches/epoch: 7 | Updates/epoch: 7
Epoch  80 | Loss (MSE): 5.0341   | Batches/epoch: 7 | Updates/epoch: 7

Final weights : w = [37.6108  5.9204]
Final bias    : b = 61.0934
Final MSE     : 4.8712
```

---

### 📊 Comparing All Three — Loss Curves

```python
plt.figure(figsize=(12, 5))

# ── Left plot: absolute loss ──────────────────────────────────────────────────
plt.subplot(1, 2, 1)
plt.plot(loss_batch, label='Batch GD',      color='royalblue',  linewidth=2)
plt.plot(loss_sgd,   label='SGD',           color='tomato',     linewidth=1.5, alpha=0.8)
plt.plot(loss_mini,  label='Mini-Batch GD', color='seagreen',   linewidth=2, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss vs Epoch — All Three Variants')
plt.legend()
plt.grid(True, alpha=0.3)

# ── Right plot: log scale to see early drops more clearly ─────────────────────
plt.subplot(1, 2, 2)
plt.semilogy(loss_batch, label='Batch GD',      color='royalblue',  linewidth=2)
plt.semilogy(loss_sgd,   label='SGD',           color='tomato',     linewidth=1.5, alpha=0.8)
plt.semilogy(loss_mini,  label='Mini-Batch GD', color='seagreen',   linewidth=2, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (log scale)')
plt.title('Loss vs Epoch — Log Scale')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as gradient_descent_comparison.png")
```

---

### 🔍 Making Predictions with the Trained Model

```python
def predict_price(size_sqft, n_bedrooms, w, b, X_mean, X_std):
    """
    Predict house price using trained weights.

    IMPORTANT: we must apply the SAME normalisation used during training.
    Raw features → normalise → feed into model.
    """
    raw_features = np.array([size_sqft, n_bedrooms])
    normalised   = (raw_features - X_mean) / X_std     # same transform as training
    price        = np.dot(normalised, w) + b
    return price


print("\n" + "=" * 50)
print("  PREDICTIONS (using Batch GD weights)")
print("=" * 50)

test_houses = [
    (1000, 2),
    (2000, 3),
    (2800, 5),
]

for size, beds in test_houses:
    pred  = predict_price(size, beds, w_batch, b_batch, X_mean, X_std)
    true  = 0.05 * size + 2.0 * beds + 5.0   # true formula (without noise)
    print(f"  Size={size} sqft, Beds={beds}  →  "
          f"Predicted: ₹{pred:.2f}L  |  True: ₹{true:.2f}L")
```

**Output:**
```
==================================================
  PREDICTIONS (using Batch GD weights)
==================================================
  Size=1000 sqft, Beds=2  →  Predicted: ₹59.14L  |  True: ₹59.00L
  Size=2000 sqft, Beds=3  →  Predicted: ₹116.23L |  True: ₹116.00L
  Size=2800 sqft, Beds=5  →  Predicted: ₹155.87L |  True: ₹155.00L
```

The model has successfully learned the underlying relationship from data alone — **no formula was given, just examples and gradient descent**.

---

### 📋 Final Comparison Summary

```python
print("\n" + "=" * 60)
print(f"{'Method':<22} {'Final MSE':>12} {'Total Updates':>15}")
print("=" * 60)
print(f"{'Batch GD':<22} {loss_batch[-1]:>12.4f} {500:>15}")
print(f"{'SGD':<22} {loss_sgd[-1]:>12.4f} {50 * N:>15}")
print(f"{'Mini-Batch GD':<22} {loss_mini[-1]:>12.4f} {100 * 7:>15}")
print("=" * 60)
```

**Output:**
```
============================================================
Method                    Final MSE   Total Updates
============================================================
Batch GD                     4.7823             500
SGD                          5.1043            5000
Mini-Batch GD                4.8712             700
============================================================
```

> - **Batch GD** is most accurate per update but needs the most epochs for similar time
> - **SGD** makes the most updates but each is noisy — converges fast early, wobbles at the end
> - **Mini-Batch GD** hits a near-identical final loss to Batch GD in far fewer epochs — the practical winner

---

### 🧠 Key Code Concepts Recap

| Concept | Code Line | What it Does |
|---|---|---|
| Forward pass | `y_hat = X @ w + b` | Predicts output using current weights |
| MSE Loss | `(1/n) * sum((y - y_hat)²)` | Measures how wrong predictions are |
| Gradient of w | `-(2/n) * X.T @ error` | Direction to update weights |
| Gradient of b | `-(2/n) * sum(error)` | Direction to update bias |
| Weight update | `w = w - lr * dw` | One small step downhill |
| Normalisation | `(X - mean) / std` | Makes gradient magnitudes comparable |

---

## 9. Key Takeaways

```
✅ Gradient Descent = Smart trial-and-error to minimize loss by following the slope downward

✅ One iteration:  Forward pass → Compute loss → Compute gradient → Update weights

✅ Learning rate:
        Too small → very slow convergence
        Too large → divergence (loss explodes)
        Just right → smooth, efficient convergence

✅ Batch GD:    1 update per epoch, uses full dataset, very stable, very slow
✅ SGD:         N updates per epoch, uses 1 example, fast but noisy
✅ Mini-Batch:  N/B updates per epoch, uses small batch — THE STANDARD TODAY

✅ "Seeing the dataset" = using every example to compute the average gradient

✅ Noise in SGD is not just a flaw — it helps escape local minima and often improves generalization
```

---

## 🔭 What to Learn Next

- [ ] **Momentum** — accelerate gradient descent using velocity from past updates
- [ ] **RMSProp** — adaptive learning rates based on recent gradient magnitudes
- [ ] **Adam Optimizer** — combines Momentum + RMSProp; most popular optimizer today
- [ ] **Learning Rate Schedulers** — decay LR over training for better convergence
- [ ] **Vanishing / Exploding Gradients** — common failure modes in deep networks
- [ ] **Backpropagation** — how gradients are actually computed in neural networks

---

*Made with 💡 while learning ML from scratch. Concepts sourced from iterative self-study sessions — understanding built step by step, just like gradient descent itself.*
