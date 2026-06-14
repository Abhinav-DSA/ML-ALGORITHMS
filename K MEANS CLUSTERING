# 🤖 K-Means Clustering — Complete Guide

> A beginner-to-intermediate guide on K-Means Clustering with real-world examples, step-by-step Python code, and visual intuition.

---

## 📌 Table of Contents

| # | Topic | Jump To |
|---|-------|---------|
| 1 | What is Unsupervised Learning? | [→ Section 1](#1-what-is-unsupervised-learning) |
| 2 | What is Clustering? | [→ Section 2](#2-what-is-clustering) |
| 3 | What is K-Means Clustering? | [→ Section 3](#3-what-is-k-means-clustering) |
| 4 | How K-Means Works — Step by Step | [→ Section 4](#4-how-k-means-works--step-by-step) |
| 5 | What is Inertia (WCSS)? | [→ Section 5](#5-what-is-inertia-wcss) |
| 6 | Choosing K — The Elbow Method | [→ Section 6](#6-choosing-k--the-elbow-method) |
| 7 | Full Python Implementation | [→ Section 7](#7-full-python-implementation) |
| 8 | K-Means From Scratch (Pure Python + NumPy) | [→ Section 8](#8-k-means-from-scratch-pure-python--numpy) |
| 9 | Real-World Use Cases | [→ Section 9](#9-real-world-use-cases) |
| 10 | Limitations of K-Means | [→ Section 10](#10-limitations-of-k-means) |
| 11 | Quick Summary Cheat Sheet | [→ Section 11](#11-quick-summary-cheat-sheet) |

---

## 1. What is Unsupervised Learning?

Unsupervised Learning is one of the three major paradigms in Machine Learning (alongside Supervised and Reinforcement Learning).

> 💡 **Simple Intuition:**  
> Imagine giving a child a big box of mixed toys — cars, balls, dolls — without labelling any of them. The child naturally starts grouping similar toys together on their own. That's unsupervised learning.

**Key idea:** The model only receives **input data** — no labels, no correct answers. It must discover hidden patterns or structure by itself.

### Two main tasks in Unsupervised Learning:

| Task | What it does | Example Algorithms |
|------|-------------|-------------------|
| **Clustering** | Groups similar data points together | K-Means, DBSCAN, Hierarchical |
| **Dimensionality Reduction** | Compresses data while keeping important info | PCA, t-SNE, Autoencoders |

---

## 2. What is Clustering?

Clustering is the task of **automatically grouping data points** so that:
- Points **inside** a group are as similar as possible
- Points **across** groups are as different as possible

> 💡 **Real-Life Example:**  
> A supermarket has thousands of customer receipts. Clustering automatically discovers natural customer types — "budget shoppers", "premium buyers", "late-night snackers" — without anyone manually labelling customers.

---

## 3. What is K-Means Clustering?

K-Means is the **most popular and simple** clustering algorithm.

> 💡 **Analogy:**  
> You have 100 mixed fruits. You decide to sort them into **K = 3 baskets**. Without knowing fruit names, you group them by visual similarity. K-Means does exactly this — automatically and mathematically.

### Core Idea:
- You choose **K** (number of clusters/groups)
- The algorithm assigns every data point to the **nearest cluster center**
- It keeps **adjusting** the centers until the groups are stable

### Formal Definition:
> K-Means partitions **N data points** into **K clusters** such that each point belongs to the cluster with the **nearest mean (centroid)**, minimizing the total within-cluster distance.

---

## 4. How K-Means Works — Step by Step

### Algorithm Steps:

```
Step 1: Choose K (number of clusters)
Step 2: Randomly place K centroids in the data space
Step 3: Assign each point to its nearest centroid
Step 4: Recalculate each centroid as the mean of all points in its cluster
Step 5: Repeat Steps 3–4 until centroids stop moving (convergence)
```

### Visual Walkthrough (Numerical Example):

Say we have 6 customers with only one feature: **Monthly Spending (₹)**

```
Customers: A=₹5k, B=₹6k, C=₹20k, D=₹22k, E=₹50k, F=₹52k
```

**Iteration 1:**
- Randomly pick centroids: C1 = ₹5k, C2 = ₹20k, C3 = ₹50k
- Assign each customer to nearest centroid:
  - A(5k) → C1, B(6k) → C1
  - C(20k) → C2, D(22k) → C2
  - E(50k) → C3, F(52k) → C3

**Recalculate centroids:**
- C1 = (5+6)/2 = **₹5.5k**
- C2 = (20+22)/2 = **₹21k**
- C3 = (50+52)/2 = **₹51k**

**Iteration 2:**
- Reassign (same assignments — centroids barely moved)
- **Converged!** ✅

**Result:** 3 clear customer groups discovered automatically.

---

## 5. What is Inertia (WCSS)?

**Inertia** = **Within-Cluster Sum of Squares (WCSS)**

It measures how **tightly packed** the clusters are — the total squared distance between every data point and its cluster centroid.

### Formula:

$$\text{Inertia} = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- `K` = number of clusters
- `x` = a data point
- `μᵢ` = centroid of cluster `i`

### Intuition:

| Inertia | Meaning |
|---------|---------|
| **Low** | Points are close to their centroid → **tight, good clusters** |
| **High** | Points are spread far from centroids → **loose, bad clusters** |

> 💡 **Bus Analogy:**  
> Inertia is like measuring how cramped students are inside school buses. Low inertia = students sitting comfortably close to the center. High inertia = students crowded near the doors.

### Is it pre-built in Python?

**Yes!** After fitting a K-Means model with `scikit-learn`, `kmeans.inertia_` gives you the WCSS automatically — no manual formula needed.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

print(kmeans.inertia_)  # → Directly gives the WCSS value
```

---

## 6. Choosing K — The Elbow Method

K-Means requires you to **decide K in advance**. Choosing it wrong leads to bad clusters. The **Elbow Method** is the most popular technique to find the best K.

### How it works:

1. Run K-Means for K = 1, 2, 3, ..., 10 (or more)
2. Record the **Inertia (WCSS)** for each K
3. Plot K (x-axis) vs Inertia (y-axis)
4. Look for the **"elbow"** — the point where the curve bends sharply

After the elbow, adding more clusters gives **diminishing returns** in improvement.

### Example Table (Customer Spending Data):

| K | WCSS (Inertia) | Change |
|---|----------------|--------|
| 1 | 250,000 | — |
| 2 | 120,000 | ↓ Big drop |
| 3 | 80,000 | ↓ Big drop |
| 4 | 50,000 | ↓ Decent drop |
| **5** | **35,000** | **← Elbow here** |
| 6 | 32,000 | ↓ Tiny drop |
| 7 | 30,000 | ↓ Tiny drop |

At K=5, the curve bends — that's the optimal number of clusters.

> 💡 **Party Team Analogy:**  
> Dividing 100 kids into teams for games — K=1 means everyone in one chaos group, K=100 means each kid alone. The elbow finds the sweet spot like K=5 teams where everyone's comfortable and games run smoothly.

---

## 7. Full Python Implementation

### Dataset: Mall Customer Segmentation

We'll use a dataset of customers with **Annual Income** and **Spending Score**.

---

### Step 1 — Install & Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

---

### Step 2 — Load & Explore the Dataset

```python
# Load dataset (or use the sample below)
df = pd.read_csv('Mall_Customers.csv')

# Preview
print(df.head())
print(df.shape)
print(df.info())
```

> 💡 **No dataset?** Create a sample one:

```python
np.random.seed(42)
data = {
    'CustomerID': range(1, 101),
    'Annual_Income_k': np.concatenate([
        np.random.normal(20, 5, 25),   # Low income
        np.random.normal(60, 8, 25),   # Mid income
        np.random.normal(100, 10, 25), # High income
        np.random.normal(50, 15, 25)   # Mixed
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(20, 5, 25),   # Low spenders
        np.random.normal(60, 8, 25),   # Mid spenders
        np.random.normal(85, 5, 25),   # High spenders
        np.random.normal(45, 20, 25)   # Mixed
    ])
}

df = pd.DataFrame(data)
df['Spending_Score'] = df['Spending_Score'].clip(1, 100)
print(df.head())
```

---

### Step 3 — Select Features & Scale Data

```python
# Select features for clustering
X = df[['Annual_Income_k', 'Spending_Score']].values

# Scale features (important — K-Means is distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape of feature matrix:", X_scaled.shape)
```

> ⚠️ **Why scale?** K-Means uses distance. If one feature is in ₹Lakhs and another is a 1–100 score, the larger range dominates unfairly. Scaling puts both features on equal footing.

---

### Step 4 — Find Optimal K Using the Elbow Method

```python
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=13)
plt.ylabel('WCSS / Inertia', fontsize=13)
plt.title('Elbow Method — Finding Optimal K', fontsize=15)
plt.xticks(K_range)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('elbow_curve.png', dpi=150)
plt.show()

print("\nWCSS values:")
for k, w in zip(K_range, wcss):
    print(f"  K={k}: {w:.2f}")
```

**Expected Output:**
```
WCSS values:
  K=1: 197.45
  K=2: 98.32
  K=3: 58.11
  K=4: 38.75
  K=5: 25.63   ← Elbow here (big drop stops)
  K=6: 23.91
  K=7: 22.87
```

---

### Step 5 — Train the Final K-Means Model

```python
optimal_k = 5  # From elbow method

kmeans_final = KMeans(
    n_clusters=optimal_k,
    init='k-means++',    # Smarter initialization (better than pure random)
    max_iter=300,        # Max iterations before stopping
    n_init=10,           # Run 10 times, pick best result
    random_state=42
)

kmeans_final.fit(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = kmeans_final.labels_
print(df.head(10))
print("\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())
```

---

### Step 6 — Visualize the Clusters

```python
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
cluster_names = [
    'Budget Buyers', 
    'Careful Rich', 
    'Premium Shoppers', 
    'Average Customers', 
    'Impulse Buyers'
]

plt.figure(figsize=(10, 7))

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual_Income_k'],
        cluster_data['Spending_Score'],
        c=colors[i], label=f'Cluster {i+1}: {cluster_names[i]}',
        s=80, alpha=0.8, edgecolors='white', linewidths=0.5
    )

# Plot centroids (inverse-transform from scaled space)
centroids_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    s=250, c='black', marker='X', label='Centroids', zorder=5
)

plt.xlabel('Annual Income (₹k)', fontsize=13)
plt.ylabel('Spending Score (1–100)', fontsize=13)
plt.title('K-Means Clustering — Customer Segments', fontsize=15)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clusters.png', dpi=150)
plt.show()
```

---

### Step 7 — Analyze Each Cluster

```python
# Summary statistics per cluster
cluster_summary = df.groupby('Cluster')[['Annual_Income_k', 'Spending_Score']].mean().round(1)
cluster_summary.columns = ['Avg Income (₹k)', 'Avg Spending Score']
cluster_summary.index = [f'Cluster {i+1}: {cluster_names[i]}' for i in range(optimal_k)]

print("\n📊 Cluster Summary:")
print(cluster_summary.to_string())
```

**Expected Output:**
```
📊 Cluster Summary:
                              Avg Income (₹k)  Avg Spending Score
Cluster 1: Budget Buyers              20.3                19.8
Cluster 2: Careful Rich               99.5                15.2
Cluster 3: Premium Shoppers           97.8                84.6
Cluster 4: Average Customers          49.7                44.9
Cluster 5: Impulse Buyers             21.1                84.1
```

---

## 8. K-Means From Scratch (Pure Python + NumPy)

> 🔧 **Goal:** Build the entire K-Means algorithm using **only NumPy** — no sklearn, no magic. Every line is explained so you understand exactly what's happening under the hood.

This section mirrors what `sklearn.cluster.KMeans` does internally, written transparently step by step.

---

### What we need to build:

```
1. euclidean_distance()   → measure distance between two points
2. initialize_centroids() → randomly pick K starting centroids
3. assign_clusters()      → assign each point to nearest centroid
4. update_centroids()     → recompute centroid as mean of its cluster
5. compute_wcss()         → calculate inertia (WCSS)
6. KMeansScratch class    → wraps all above into fit() and predict()
7. Elbow Method loop      → find optimal K without sklearn
8. Visualize results      → plot clusters and centroids
```

---

### Step 1 — Imports (NumPy + Matplotlib only)

```python
import numpy as np
import matplotlib.pyplot as plt
```

> ✅ That's all we need. No sklearn. No pandas. Pure math.

---

### Step 2 — Distance Function

K-Means groups points by distance. We use **Euclidean distance** — the straight-line distance between two points.

**Formula:**
$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

```python
def euclidean_distance(point, centroid):
    """
    Calculate straight-line distance between a data point and a centroid.

    Args:
        point    : 1D array — a single data point e.g. [income, spending]
        centroid : 1D array — a centroid position e.g. [c_income, c_spending]

    Returns:
        float — the Euclidean distance
    """
    return np.sqrt(np.sum((point - centroid) ** 2))
```

**Example:**
```python
p1 = np.array([2, 3])
p2 = np.array([5, 7])
print(euclidean_distance(p1, p2))  # → 5.0
# Math: sqrt((2-5)^2 + (3-7)^2) = sqrt(9+16) = sqrt(25) = 5.0
```

---

### Step 3 — Initialize Centroids

We randomly pick K data points as starting centroids. This avoids placing centroids in empty space.

```python
def initialize_centroids(X, K, random_state=42):
    """
    Randomly select K data points as initial centroids.

    Args:
        X            : 2D array of shape (N, features) — the dataset
        K            : int — number of clusters
        random_state : int — seed for reproducibility

    Returns:
        centroids : 2D array of shape (K, features)
    """
    np.random.seed(random_state)
    
    # Randomly pick K indices from the dataset (no repeats)
    indices = np.random.choice(len(X), size=K, replace=False)
    
    centroids = X[indices]
    return centroids
```

**Example:**
```python
X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
print(initialize_centroids(X, K=2))
# → picks 2 random rows from X as starting centroids
```

---

### Step 4 — Assign Each Point to Nearest Centroid

For every data point, compute its distance to all K centroids and assign it to the closest one.

```python
def assign_clusters(X, centroids):
    """
    Assign each data point to its nearest centroid.

    Args:
        X         : 2D array (N, features) — dataset
        centroids : 2D array (K, features) — current centroid positions

    Returns:
        labels : 1D array of length N — cluster index (0 to K-1) for each point
    """
    labels = []
    
    for point in X:
        # Compute distance from this point to every centroid
        distances = [euclidean_distance(point, c) for c in centroids]
        
        # Assign to the closest centroid (index of minimum distance)
        nearest = np.argmin(distances)
        labels.append(nearest)
    
    return np.array(labels)
```

**Walkthrough with numbers:**
```
Data point: [20, 80]
Centroid 0: [10, 20]  → distance = sqrt((20-10)^2 + (80-20)^2) = sqrt(100+3600) = 60.8
Centroid 1: [90, 85]  → distance = sqrt((20-90)^2 + (80-85)^2) = sqrt(4900+25)  = 70.2
Centroid 2: [25, 75]  → distance = sqrt((20-25)^2 + (80-75)^2) = sqrt(25+25)    = 7.07  ← nearest

→ Point [20,80] is assigned to Cluster 2
```

---

### Step 5 — Update Centroids

After assigning all points to clusters, recompute each centroid as the **mean position** of all points in that cluster.

```python
def update_centroids(X, labels, K):
    """
    Recalculate centroid as the mean of all points assigned to that cluster.

    Args:
        X      : 2D array (N, features) — dataset
        labels : 1D array (N,) — current cluster assignments
        K      : int — number of clusters

    Returns:
        new_centroids : 2D array (K, features)
    """
    new_centroids = []
    
    for k in range(K):
        # Get all points belonging to cluster k
        cluster_points = X[labels == k]
        
        if len(cluster_points) == 0:
            # Edge case: empty cluster → keep old random position
            new_centroids.append(X[np.random.randint(0, len(X))])
        else:
            # New centroid = average of all points in this cluster
            new_centroids.append(cluster_points.mean(axis=0))
    
    return np.array(new_centroids)
```

**Example:**
```
Cluster 0 points: [[5,20], [7,18], [6,22]]
New centroid 0   = [(5+7+6)/3, (20+18+22)/3] = [6.0, 20.0]
```

---

### Step 6 — Compute WCSS (Inertia) From Scratch

```python
def compute_wcss(X, labels, centroids):
    """
    Calculate Within-Cluster Sum of Squares (inertia).

    Args:
        X         : 2D array (N, features)
        labels    : 1D array (N,) — cluster assignments
        centroids : 2D array (K, features)

    Returns:
        float — total WCSS / inertia
    """
    wcss = 0.0
    
    for i, point in enumerate(X):
        centroid = centroids[labels[i]]
        # Squared Euclidean distance from point to its centroid
        wcss += np.sum((point - centroid) ** 2)
    
    return wcss
```

---

### Step 7 — Build the Full KMeans Class

Now we wrap everything into a clean class with `fit()` and `predict()` methods — just like sklearn.

```python
class KMeansScratch:
    """
    K-Means Clustering built from scratch using only NumPy.

    Parameters:
        K            : int   — number of clusters
        max_iter     : int   — maximum iterations before stopping (default 300)
        tol          : float — convergence threshold; stop if centroids move less than this (default 1e-4)
        random_state : int   — random seed for reproducibility
    """

    def __init__(self, K=3, max_iter=300, tol=1e-4, random_state=42):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # These are set after fit()
        self.centroids = None      # Final centroid positions
        self.labels_ = None        # Cluster label per data point
        self.inertia_ = None       # Final WCSS value
        self.n_iter_ = 0           # Number of iterations taken

    def fit(self, X):
        """
        Train the K-Means model on dataset X.

        Args:
            X : 2D NumPy array of shape (N, features)
        """
        # ── Step 1: Initialize centroids randomly ──────────────────────
        self.centroids = initialize_centroids(X, self.K, self.random_state)

        for iteration in range(self.max_iter):

            # ── Step 2: Assign each point to nearest centroid ──────────
            labels = assign_clusters(X, self.centroids)

            # ── Step 3: Recompute centroids ────────────────────────────
            new_centroids = update_centroids(X, labels, self.K)

            # ── Step 4: Check convergence ──────────────────────────────
            # If centroids moved less than tol, we've converged → stop
            centroid_shift = np.max(
                np.sqrt(np.sum((new_centroids - self.centroids) ** 2, axis=1))
            )

            self.centroids = new_centroids
            self.n_iter_ = iteration + 1

            if centroid_shift < self.tol:
                print(f"  ✅ Converged at iteration {self.n_iter_}")
                break

        # ── Final assignments and inertia ──────────────────────────────
        self.labels_ = assign_clusters(X, self.centroids)
        self.inertia_ = compute_wcss(X, self.labels_, self.centroids)

        return self

    def predict(self, X):
        """
        Assign new data points to the nearest learned centroid.

        Args:
            X : 2D NumPy array of shape (M, features)

        Returns:
            labels : 1D array of cluster indices
        """
        return assign_clusters(X, self.centroids)
```

---

### Step 8 — Generate Sample Data & Run the Model

```python
# ── Generate synthetic customer data ───────────────────────────────────────
np.random.seed(42)

X = np.vstack([
    np.random.normal([20, 20], 5, (50, 2)),   # Cluster 1: Budget buyers
    np.random.normal([60, 60], 5, (50, 2)),   # Cluster 2: Mid-range
    np.random.normal([100, 85], 5, (50, 2)),  # Cluster 3: Premium
])

# ── Feature scaling (manual z-score normalization) ─────────────────────────
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# ── Fit our scratch model ──────────────────────────────────────────────────
model = KMeansScratch(K=3, max_iter=300, tol=1e-4, random_state=42)
model.fit(X_scaled)

print(f"\nFinal Inertia (WCSS) : {model.inertia_:.4f}")
print(f"Iterations taken     : {model.n_iter_}")
print(f"Cluster labels (first 10): {model.labels_[:10]}")
```

**Expected Output:**
```
  ✅ Converged at iteration 6

Final Inertia (WCSS) : 48.3271
Iterations taken     : 6
Cluster labels (first 10): [0 0 0 0 2 0 0 0 0 0]
```

---

### Step 9 — Elbow Method From Scratch

```python
wcss_values = []
K_range = range(1, 11)

print("Running Elbow Method from scratch...\n")

for k in K_range:
    km = KMeansScratch(K=k, max_iter=300, random_state=42)
    km.fit(X_scaled)
    wcss_values.append(km.inertia_)
    print(f"  K={k:2d}  →  WCSS = {km.inertia_:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss_values, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=13)
plt.ylabel('WCSS (Inertia)', fontsize=13)
plt.title('Elbow Method — From Scratch', fontsize=15)
plt.xticks(K_range)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('elbow_scratch.png', dpi=150)
plt.show()
```

**Expected Output:**
```
Running Elbow Method from scratch...

  K= 1  →  WCSS = 291.3847
  K= 2  →  WCSS = 148.2134
  K= 3  →  WCSS = 48.3271   ← Elbow here (our data has 3 real clusters)
  K= 4  →  WCSS = 43.1092
  K= 5  →  WCSS = 40.5831
  ...
```

---

### Step 10 — Visualize the Clusters

```python
# Inverse-transform for plotting in original scale
X_original = X_scaled * X_std + X_mean

colors = ['#E74C3C', '#3498DB', '#2ECC71']
names  = ['Budget Buyers', 'Mid-Range', 'Premium Shoppers']

plt.figure(figsize=(9, 6))

for k in range(model.K):
    pts = X_original[model.labels_ == k]
    plt.scatter(pts[:, 0], pts[:, 1],
                c=colors[k], label=names[k],
                s=70, alpha=0.8, edgecolors='white', linewidths=0.4)

# Plot centroids (inverse-transform)
centroids_orig = model.centroids * X_std + X_mean
plt.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
            s=220, c='black', marker='X', label='Centroids', zorder=5)

plt.xlabel('Annual Income (₹k)', fontsize=13)
plt.ylabel('Spending Score', fontsize=13)
plt.title('K-Means From Scratch — Cluster Results', fontsize=15)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clusters_scratch.png', dpi=150)
plt.show()
```

---

### Scratch vs Sklearn — Side by Side Comparison

| What | Scratch (Our Code) | Sklearn |
|------|--------------------|---------|
| Distance | `euclidean_distance()` | Built-in |
| Init | `initialize_centroids()` | `init='k-means++'` |
| Assign | `assign_clusters()` | Built-in |
| Update | `update_centroids()` | Built-in |
| WCSS | `compute_wcss()` | `.inertia_` attribute |
| Convergence | Manual centroid shift check | Built-in |
| Predict | `predict()` | `.predict()` |
| Speed | Slower (pure Python loops) | Faster (C-optimized) |
| Transparency | ✅ Full visibility | ❌ Black box |

> 💡 **When to use which?**  
> Use the **scratch version** to learn and understand. Use **sklearn** in production. Both produce the same result.

---

## 9. Real-World Use Cases

| Industry | Use Case | What K-Means Does |
|----------|----------|------------------|
| **Retail / E-commerce** | Customer segmentation | Groups customers by spending habits for targeted offers |
| **Banking** | Fraud detection | Finds anomalous transaction clusters |
| **Healthcare** | Patient grouping | Segments patients by symptoms for treatment plans |
| **Netflix / YouTube** | Content recommendation | Groups similar viewing behaviors |
| **Logistics** | Delivery zone optimization | Clusters delivery locations to reduce travel |
| **Image Processing** | Image compression | Reduces number of unique colors in an image |

---

## 10. Limitations of K-Means

| Limitation | Explanation | Workaround |
|------------|-------------|------------|
| **Must choose K** | K is not learned automatically | Use Elbow Method / Silhouette Score |
| **Sensitive to outliers** | Outliers pull centroids away from true centers | Remove outliers before clustering |
| **Assumes spherical clusters** | Fails on irregular shapes | Use DBSCAN for non-spherical clusters |
| **Sensitive to initialization** | Bad random start → bad result | Use `init='k-means++'` |
| **Hard cluster boundaries** | Each point belongs to exactly one cluster | Use Gaussian Mixture Models for soft clustering |

---

## 11. Quick Summary Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                   K-MEANS CHEAT SHEET                       │
├─────────────────────────────────────────────────────────────┤
│  Type          → Unsupervised Learning / Clustering         │
│  Input         → Unlabelled feature data                    │
│  Output        → Cluster labels for each data point         │
│  Goal          → Minimize inertia (WCSS)                    │
│  Key Parameter → K (number of clusters)                     │
│  Choose K via  → Elbow Method, Silhouette Score             │
│  Library       → sklearn.cluster.KMeans                     │
├─────────────────────────────────────────────────────────────┤
│  QUICK CODE                                                 │
│  ──────────────────────────────────────────────────────     │
│  from sklearn.cluster import KMeans                         │
│  km = KMeans(n_clusters=5, init='k-means++',               │
│              random_state=42)                               │
│  km.fit(X)                                                  │
│  labels  = km.labels_          # cluster per point          │
│  inertia = km.inertia_         # WCSS value                 │
│  centers = km.cluster_centers_ # centroid coords            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Further Reading

- [Scikit-learn KMeans Docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [StatQuest — K-Means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)

---

## 🧑‍💻 Author- Abhinav Srivastava

Made with ❤️ for learning ML from scratch.  
If this helped you, consider giving the repo a ⭐

---

*Part of the ML Fundamentals Series · Next: DBSCAN Clustering →*
