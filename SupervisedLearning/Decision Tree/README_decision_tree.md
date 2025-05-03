# Decision Tree

Decision Trees partition feature space into axis‐aligned regions by recursively splitting on individual features.  Below is the mathematical formulation and problem setup for a binary‐classification tree.
## 1. Problem Setup  
We have a dataset of $n$ labeled examples  
$$
X = \begin{bmatrix}
\mathbf{x}^{(1)} \\[4pt]
\mathbf{x}^{(2)} \\[2pt]
\vdots \\[2pt]
\mathbf{x}^{(n)}
\end{bmatrix}
\in \mathbb{R}^{n\times d},
\quad
\mathbf{y} = \bigl\{y^{(i)}\bigr\}_{i=1}^n,\quad y^{(i)}\in\{0,1\}.
$$  
Goal: learn a model $f:\mathbb{R}^d\to\{0,1\}$ that maps features $\mathbf{x}$ to class labels.
## 2. Node Impurity Measures  
At each node containing sample set $S$, we measure the “mixedness” of labels:

- **Gini Impurity**  
  $$
  G(S) = 1 - \sum_{c\in\{0,1\}} p_c^2,
  \quad
  p_c = \frac{1}{|S|}\sum_{i\in S}\mathbf{1}(y^{(i)}=c).
  $$

- **Entropy**  
  $$
  H(S) = -\sum_{c\in\{0,1\}} p_c\;\log_2(p_c).
  $$

Lower impurity means a purer node (more homogeneous labels).
## 3. Splitting Criterion  
For a candidate split on feature $j$ at threshold $t$, partition $S$ into
$$
S_{\text{left}} 
= \{\mathbf{x}^{(i)}\in S : x^{(i)}_j \le t\},
\quad
S_{\text{right}} 
= S\setminus S_{\text{left}}.
$$  
We choose $(j,t)$ to **maximize information gain**:
$$
\mathrm{IG}(S, j, t)
= I(S) \;-\; \frac{|S_{\text{left}}|}{|S|} I\bigl(S_{\text{left}}\bigr)
             \;-\; \frac{|S_{\text{right}}|}{|S|} I\bigl(S_{\text{right}}\bigr),
$$  
where $I(\cdot)$ is either $G(\cdot)$ or $H(\cdot)$.
## 4. Recursive Tree Growth  
1. **Start** at the root with $S =\{1,\dots,n\}$.  
2. **Find** best $(j^*,t^*)$ by scanning features and thresholds to maximize $\mathrm{IG}$.  
3. **Split** into left/right children, compute their impurities.  
4. **Recurse** on each child node until a stopping condition:  
   - All samples in the node share the same label.  
   - Maximum tree depth reached.  
   - Minimum samples per node threshold.  
## 5. Prediction  
For a new $\mathbf{x}$, traverse from the root:
1. At node splitting on $(j^*,t^*)$, go left if $x_j^* \le t^*$, else right.  
2. Repeat until a leaf is reached.  
3. **Leaf prediction**: majority class of training samples in that leaf, e.g.
$$
\hat f(\mathbf{x}) = 
\begin{cases}
0, & p_0 > p_1,\\
1, & \text{otherwise},
\end{cases}
\quad
p_c = \frac{\#\{y^{(i)}=c\}}{\text{\#samples in leaf}}.
$$
## 6. Regularization  
- **Overfitting control**:  
  - **Pruning**: remove splits that add little IG.  
  - **Max depth** / **min samples per leaf**: limit tree size.  
- Trees are interpretable and form the basis for ensembles (Random Forests, Gradient Boosting).