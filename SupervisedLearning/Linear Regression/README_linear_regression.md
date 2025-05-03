# Linear Regression
Linear regression is a supervised learning method for modeling the relationship between one continuous target variable and one or more input features. It assumes that the expected value of the target is a linear combination of the inputs plus an intercept term. By minimizing the mean squared error between the predicted and actual values—typically via ordinary least squares or gradient descent—it finds the best‐fit line (or hyperplane) through the data. The learned coefficients tell you how much a one‐unit change in each standardized feature shifts the predicted output, making the model both interpretable and easy to inspect. Key assumptions include linearity, independence of errors, homoscedasticity (constant variance of errors), and normality of residuals for inference.


## 1. Problem Setup  
We have a training set of \(n\) samples  
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
\in \mathbb{R}^{n}
$$
where each feature vector $\mathbf{x}^{(i)}\in\mathbb{R}^d$ is standardized.

---

## 2. Linear Model  
We model the target as a linear function:  
$$
\hat y^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b,
$$  
with weight vector $\mathbf{w}\in\mathbb{R}^d$ and bias $b\in\mathbb{R}$.

---

## 3. Loss Function  
We minimize the Mean Squared Error (MSE):  
$$
\mathcal{L}(\mathbf{w}, b)
= \frac{1}{n}\sum_{i=1}^n \bigl(y^{(i)} - \hat y^{(i)}\bigr)^2
= \frac{1}{n}\sum_{i=1}^n \bigl(y^{(i)} - (\mathbf{w}^\top \mathbf{x}^{(i)} + b)\bigr)^2.
$$

---

## 4. Gradients  
Compute partial derivatives of $\mathcal{L}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}}
= -\frac{2}{n}\sum_{i=1}^n \mathbf{x}^{(i)}\bigl(y^{(i)} - \hat y^{(i)}\bigr),
\quad
\frac{\partial \mathcal{L}}{\partial b}
= -\frac{2}{n}\sum_{i=1}^n \bigl(y^{(i)} - \hat y^{(i)}\bigr).
$$

---

## 5. Gradient Descent Updates  
With learning rate $\eta$, for each iteration $t=1,\dots,T$:

$$
\mathbf{w}
\gets \mathbf{w} - \eta \,\frac{\partial \mathcal{L}}{\partial \mathbf{w}},
\qquad
b
\gets b - \eta \,\frac{\partial \mathcal{L}}{\partial b}.
$$

