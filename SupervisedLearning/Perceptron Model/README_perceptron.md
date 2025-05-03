# The Perceptron Model

The Perceptron is a foundational single-neuron model extensively used in machine learning for binary classification tasks. This model classifies data into two distinct categories by iteratively adjusting its weights based on input features. Inspired by the functionality of biological neurons, the Perceptron is both simple and effective, serving as a vital building block for more complex neural architectures. Its straightforward mechanism involves learning from training data to reduce classification errors, thereby improving accuracy over time. In this document, we will delve into the implementation and application of the Perceptron model using the college basketball dataset (cbb), illustrating how it can effectively differentiate between various performance metrics of teams.

## 1. Problem Setup  
We have a labeled dataset of \(n\) examples  

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
\in \{-1, +1\}^n,
$$

where each input vector $$\mathbf{x}^{(i)}$$ is $d$-dimensional (optionally standardized), and each label $y^{(i)}$ indicates one of two classes.


---

## 2. Linear Score & Activation  
The perceptron computes a linear score and then applies a hard threshold:

$$
z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b,
\qquad
\hat y^{(i)} = 
\begin{cases}
+1, & z^{(i)} \ge 0,\\
-1, & z^{(i)} < 0.
\end{cases}
$$

Here $\mathbf{w}\in\mathbb{R}^d$ is the weight vector, and $b\in\mathbb{R}$ is the bias (threshold offset).

---

## 3. Learning Rule  
Whenever the perceptron misclassifies a point, it updates parameters by:

$$
\text{If } y^{(i)}\,\hat y^{(i)} \le 0:
\quad
\mathbf{w} \;\gets\; \mathbf{w} + \eta\,y^{(i)}\,\mathbf{x}^{(i)},
\quad
b \;\gets\; b + \eta\,y^{(i)},
$$

where $\eta>0$ is the learning rate. A correctly classified example $(y^{(i)}\hat y^{(i)}>0)$ incurs no update.

---

## 4. Perceptron Criterion & Convergence  
The perceptron criterion penalizes only margin violations:

$$
L(\mathbf{w}, b)
= -\sum_{i=1}^n \mathbf{1}\bigl(y^{(i)}\,z^{(i)} \le 0\bigr)\,y^{(i)}\,z^{(i)}.
$$

- **Convergence Theorem**: If the data are linearly separable, the algorithm finds a separating hyperplane in finitely many updates.  
- If not separable, updates continue indefinitely, oscillating around the best approximate boundary.

---

## 5. Decision Boundary & Geometry  
The learned hyperplane

$$
\{\mathbf{x}\mid \mathbf{w}^\top\mathbf{x} + b = 0\}
$$

splits $\mathbb{R}^d$ into two half-spaces:

- Points with $\mathbf{w}^\top\mathbf{x}+b>0$ are assigned $+1$.  
- Points with $\mathbf{w}^\top\mathbf{x}+b<0$ are assigned $-1$.  

Each update $\mathbf{w}\gets\mathbf{w}+\eta\,y\,\mathbf{x}$ rotates and shifts the boundary to increase the signed margin $y\,z$.

---

## 6. Expressive Power & Limitations  
- A single perceptron can only learn **linearly separable** patterns.  
- It cannot model **nonlinear** decision surfaces or interactions.  
- Training halts when misclassifications reach zero (if separable), or oscillates if not.

---

## 7. Diagnostics & Extensions  
- **Error vs. Epoch**: number of updates per pass to monitor convergence.  
- **RMSE of Predictions**: $\sqrt{\tfrac{1}{n}\sum(\hat y - y)^2}$ highlights margin errors.  
- **ROC AUC** using raw scores $z$ measures ranking quality.  
- **Confusion Matrix** inspects class-specific error patterns.

To capture nonlinearity, extend to multi-layer architectures, kernel transformations, or ensemble multiple linear units.
