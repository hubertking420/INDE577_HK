# Principal Component Analysis (PCA)

PCA reduces data dimensionality by projecting onto axes that capture maximum variance.

## 1. Standardize
Center and scale each feature:
$$
z = \frac{x - \mu}{\sigma}
$$

## 2. Covariance Matrix
Compute on standardized data \(A\in\mathbb{R}^{n\times p}\):
$$
S = \frac{1}{n-1}\,A^T A
$$

## 3. Eigendecomposition
Find eigenpairs of \(S\):
$$
S\,v_i = \lambda_i\,v_i,
$$
where \(v_i\) are principal components and \(\lambda_i\) their variances.

## 4. Select Top \(k\)
Order \(\{\lambda_i\}\) descending and keep corresponding \(\{v_i\}_{i=1}^k\).

## 5. Project Data
Map \(A\) onto the new basis:
$$
Z = A\,[v_1,\,v_2,\,\dots,\,v_k].
$$

---

### Variance Explained
Fraction by component \(i\):
$$
\frac{\lambda_i}{\sum_j \lambda_j}.
$$

---

### SVD Perspective
Equivalently, via SVD:
$$
A = U\,\Sigma\,V^T,
$$
with columns of \(U\) as principal directions and \(\Sigma_{ii}^2\) proportional to \(\lambda_i\).

---

**In practice**, choose \(k\) so that the first \(k\) components explain a satisfactory total variance (e.g., ≥99%).  
