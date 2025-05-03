# k-Nearest Neighbors (KNN)

k Nearest Neighbors is a non-parametric, instance-based learning algorithm used for classification (and regression).  It makes no assumptions about the underlying data distribution—rather, it classifies a new point by “voting” among its \(k\) closest neighbors in feature space.  Despite its simplicity, KNN often performs remarkably well when local structure dominates the decision boundary.

---

## 1. Problem Setup  
We have \(n\) labeled samples  
$$
X = \begin{bmatrix}
\mathbf{x}^{(1)} \\[4pt]
\mathbf{x}^{(2)} \\[2pt]
\vdots \\[2pt]
\mathbf{x}^{(n)}
\end{bmatrix}
\in \mathbb{R}^{n\times d},
\quad
\mathbf{y} = \begin{bmatrix}
y^{(1)} \\[2pt]
y^{(2)} \\[2pt]
\vdots \\[2pt]
y^{(n)}
\end{bmatrix}
\in \{0,1\}^n,
$$  
where each \(\mathbf{x}^{(i)}\in\mathbb{R}^d\) is a feature vector and \(y^{(i)}\) the binary label.

---

## 2. Distance Metric  
Compute the Euclidean distance between a query \(\mathbf{x}\) and each training point:
$$
\mathrm{dist}\bigl(\mathbf{x},\mathbf{x}^{(i)}\bigr)
= \|\mathbf{x} - \mathbf{x}^{(i)}\|_2
= \sqrt{\sum_{j=1}^d\bigl(x_j - x^{(i)}_j\bigr)^2}.
$$

---

## 3. Neighbor Selection  
For a query \(\mathbf{x}\), identify the \(k\) smallest distances:
$$
\mathcal{N}_k(\mathbf{x})
= \bigl\{\mathbf{x}^{(i_1)},\dots,\mathbf{x}^{(i_k)}\bigr\},
\quad
\mathrm{where}\;\; 
\mathrm{dist}(\mathbf{x},\mathbf{x}^{(i_1)}) \le \dots \le \mathrm{dist}(\mathbf{x},\mathbf{x}^{(i_k)}).
$$

---

## 4. Classification Rule  
Estimate the probability of the positive class by averaging neighbor labels:
$$
\hat p(\mathbf{x})
= \frac{1}{k}\sum_{\mathbf{x}^{(i)}\in\mathcal{N}_k(\mathbf{x})}y^{(i)}.
$$  
Then predict
$$
\hat y(\mathbf{x}) =
\begin{cases}
1, & \hat p(\mathbf{x}) \ge 0.5,\\
0, & \hat p(\mathbf{x}) < 0.5.
\end{cases}
$$

---

## 5. Regression Rule  
For a continuous target \(y\), simply take the neighbor average:
$$
\hat y(\mathbf{x})
= \frac{1}{k}\sum_{\mathbf{x}^{(i)}\in\mathcal{N}_k(\mathbf{x})}y^{(i)}.
$$

