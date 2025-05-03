# Singular Value Decomposition (SVD)

**SVD** factorizes any real matrix $A\in\mathbb{R}^{m\times n}$ into three components:
$$
A = U\,\Sigma\,V^T,
$$
where  
- $U\in\mathbb{R}^{m\times r}$ has orthonormal **left singular vectors** as columns,  
- $\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_r)\in\mathbb{R}^{r\times r}$ is a diagonal matrix of **singular values** $\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0$,  
- $V\in\mathbb{R}^{n\times r}$ has orthonormal **right singular vectors** as columns,  
- $r=\mathrm{rank}(A)$.

---

## Mathematical Foundations

1. **Connection to Eigen‐Decomposition**  
   - $A A^T\,u_i = \sigma_i^2\,u_i$  
   - $A^T A\,v_i = \sigma_i^2\,v_i$  
   Thus, $\{u_i\}$ and $\{v_i\}$ are eigenvectors of $AA^T$ and $A^TA$ respectively.

2. **Optimal Low‐Rank Approximation**  
   Truncating to the top-$k$ singular values yields the best rank-$k$ approximation in Frobenius norm:
   $$
   A_k = U_k\,\Sigma_k\,V_k^T
   \quad\text{minimizes}\quad
   \|A - B\|_F \quad\text{over all rank-}k\;B.
   $$

3. **Variance & Energy**  
   The proportion of “energy” (variance) captured by the first $k$ modes is
   $$
   \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{j=1}^r \sigma_j^2}.
   $$

---

## Usage for Image Compression

1. **Compute SVD** on an $m\times n$ grayscale image matrix $A$.  
2. **Choose $k$** based on desired compression vs. fidelity (e.g., retain 90–99% energy).  
3. **Reconstruct**:
   $$
   A_k = U[:,1:k]\,\Sigma[1:k,1:k]\,V[:,1:k]^T.
   $$
4. **Results**:
   - **Low $k$**: high compression, blurred image, large error.  
   - **Higher $k$**: finer detail, lower error, more storage.  

---

**Key Takeaway:**  
SVD provides a principled way to compress and denoise images by keeping only the most significant singular modes, balancing storage efficiency against reconstruction accuracy.  
