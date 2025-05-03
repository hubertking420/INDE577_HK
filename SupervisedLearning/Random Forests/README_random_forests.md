# Mathematical Foundations of Random Forests

A Random Forest is an ensemble of $B$ decision trees $\{T_b\}_{b=1}^B$ built on bootstrap samples and random feature subsets.  Its power comes from variance reduction and de-correlation of individual trees.

---

## 1. Single Decision Tree Split Criterion

At each node, a tree chooses a split that maximizes some measure of “purity gain.”  Two common criteria:

1. **Gini Impurity**  
   For a node containing data with class probabilities $\{p_k\}_{k=1}^K$,  
   $$
   G = \sum_{k=1}^K p_k (1 - p_k)
   = 1 - \sum_{k=1}^K p_k^2.
   $$
   A split into left/right children $L,R$ yields impurity decrease  
   $$
   \Delta G = G_{\text{parent}}
   - \frac{n_L}{n_{\text{parent}}} G_L
   - \frac{n_R}{n_{\text{parent}}} G_R.
   $$

2. **Entropy (Information Gain)**  
   $$
   H = -\sum_{k=1}^K p_k \log p_k,
   $$
   with split gain  
   $$
   \Delta H
   = H_{\text{parent}}
   - \frac{n_L}{n_{\text{parent}}} H_L
   - \frac{n_R}{n_{\text{parent}}} H_R.
   $$

---

## 2. Bootstrap Sampling (“Bagging”)

Each tree $T_b$ is trained on a bootstrap sample of size $n$ drawn **with replacement** from the original data:
$$
\{(x_i^{(b)},y_i^{(b)})\}_{i=1}^n
\sim
\text{SampleWithReplacement}\bigl\{(x_i,y_i)\bigr\}_{i=1}^n.
$$
This injects independence between trees and reduces variance when aggregating.

---

## 3. Random Feature Subsetting

At each split in tree $T_b$, rather than considering all $p$ features, only a random subset of $m \ll p$ features is evaluated.  This further de-correlates trees:
$$
\text{Choose split among a random set of }m\text{ features at each node.}
$$

---

## 4. Ensemble Prediction

### Regression

The forest prediction is the average of individual tree outputs:
$$
\hat f_{\text{RF}}(x)
= \frac{1}{B} \sum_{b=1}^B T_b(x).
$$

### Classification

For $K$ classes, use majority vote:
$$
\hat C_{\text{RF}}(x)
= \arg\max_{k=1,\dots,K}
\sum_{b=1}^B \mathbf{1}\bigl\{T_b(x)=k\bigr\}.
$$

---

## 5. Variance Reduction

Let each tree have variance $\sigma^2$ and pairwise correlation $\rho$.  The variance of the average is
$$
\mathrm{Var}\bigl(\hat f_{\text{RF}}(x)\bigr)
= \frac{1}{B^2}
\Bigl(B \sigma^2 + B(B-1)\rho \,\sigma^2\Bigr)
= \frac{\sigma^2}{B}\bigl[\,1 + (B-1)\rho\bigr]\!\big/B
\approx \rho\,\sigma^2 + \mathcal{O}\!\bigl(\tfrac{1}{B}\bigr).
$$
As $B\to\infty$, variance $\to \rho\,\sigma^2$, so **lower correlation** $\rho$ ⇒ **better** variance reduction.

---

## 6. Bias–Variance Trade-off

- **Bias:** Each deep tree has low bias.  
- **Variance:** Bagging + feature randomness reduces variance dramatically.  
- **Overall:** Random Forests achieve low bias without the high variance of a single tree.

---

## 7. Summary

1. **Purity-based splits** (Gini/Entropy) build expressive trees.  
2. **Bootstrap aggregation** smooths predictions by averaging.  
3. **Random feature selection** de-correlates trees.  
4. **Ensemble averaging** yields superior generalization through variance reduction.
