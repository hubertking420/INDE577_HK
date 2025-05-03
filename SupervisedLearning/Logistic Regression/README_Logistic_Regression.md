# Logistic Regression

Logistic Regression is a powerful statistical method used primarily for binary classification problems. It models the probability of a binary response based on one or more predictor variables. By applying the logistic function, the output is transformed into a probability score that lies between 0 and 1. This score is then used to predict whether the input features belong to one category or another, making it particularly effective in situations where you are trying to predict the presence or absence of a characteristic (e.g., pass/fail, win/lose, healthy/sick). Logistic Regression is straightforward yet robust, providing a probabilistic framework and easy interpretability, which makes it a staple algorithm in the arsenal of machine learning techniques. Its efficiency and simplicity are particularly advantageous in applications involving large datasets and scenarios requiring quick, real-time decision-making.


## 1. Problem Setup  
We have a dataset of \(n\) independent samples  
$$
X = \begin{bmatrix}
\mathbf{x}^{(1)} \\[4pt]
\mathbf{x}^{(2)} \\[2pt]
\vdots \\[2pt]
\mathbf{x}^{(n)}
\end{bmatrix}
\in \mathbb{R}^{n \times d},
\quad
\mathbf{y} = \begin{bmatrix}
y^{(1)} \\[2pt]
y^{(2)} \\[2pt]
\vdots \\[2pt]
y^{(n)}
\end{bmatrix}
\in \{0,1\}^n
$$  
where each feature vector \(\mathbf{x}^{(i)}\in\mathbb{R}^d\) is (optionally) standardized, and each binary label \(y^{(i)}\) indicates class membership (e.g.\ white/red wine, pass/fail).

---

## 2. The Logistic Model  
Logistic regression posits that the **log-odds** of the positive class is a linear function of the inputs:  
$$
\log\frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})}
= \mathbf{w}^\top \mathbf{x} + b.
$$  
Equivalently, the model predicts class probability via the **sigmoid** (logistic) function:
$$
\hat p(\mathbf{x})
= P(y=1 \mid \mathbf{x})
= \sigma\bigl(\mathbf{w}^\top \mathbf{x} + b\bigr),
\quad
\sigma(z) = \frac{1}{1 + e^{-z}} \;\in (0,1).
$$

---

## 3. Loss Function: Binary Cross‐Entropy  
We choose the negative log-likelihood (a.k.a.\ binary cross‐entropy) as the cost to minimize:
$$
\mathcal{L}(\mathbf{w}, b)
= -\frac{1}{n}\sum_{i=1}^n \Bigl[
y^{(i)}\log\hat p(\mathbf{x}^{(i)}) 
\;+\;(1-y^{(i)})\log\bigl(1 - \hat p(\mathbf{x}^{(i)})\bigr)
\Bigr].
$$  
Minimizing \(\mathcal{L}\) forces predicted probabilities to match true labels.

---

## 4. Gradients  
Compute partial derivatives for gradient‐based optimization:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}}
= \frac{1}{n}\sum_{i=1}^n \bigl(\hat p(\mathbf{x}^{(i)}) - y^{(i)}\bigr)\,\mathbf{x}^{(i)},
\quad
\frac{\partial \mathcal{L}}{\partial b}
= \frac{1}{n}\sum_{i=1}^n \bigl(\hat p(\mathbf{x}^{(i)}) - y^{(i)}\bigr).
$$

---

## 5. Gradient Descent Updates  
Let \(\eta\) be the learning rate. For iteration \(t=1,\dots,T\):
$$
\mathbf{w}
\gets \mathbf{w} - \eta \,\frac{\partial \mathcal{L}}{\partial \mathbf{w}},
\qquad
b
\gets b - \eta \,\frac{\partial \mathcal{L}}{\partial b}.
$$  
In code form:
```python
for _ in range(n_iters):
    z     = X.dot(w) + b
    p     = sigmoid(z)                      # shape (n,)
    dw    = (1/n) * X.T.dot(p - y)         # shape (d,)
    db    = (1/n) * np.sum(p - y)          # scalar
    w    -= lr * dw
    b    -= lr * db
