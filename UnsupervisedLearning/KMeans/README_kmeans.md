# K-Means Clustering

**K-Means** is an unsupervised algorithm that partitions $n$ samples into $K$ clusters by minimizing within-cluster variance:

1. **Initialization**  
   Randomly place $K$ centroids in feature space.
2. **Assignment Step**  
   Assign each sample $x_i$ to the nearest centroid $c_j$:  
   $$
     \text{cluster}(x_i) \;=\; \arg\min_{j=1,\dots,K} \|x_i - c_j\|^2.
   $$
3. **Update Step**  
   Recompute each centroid as the mean of its assigned points:  
   $$
     c_j \;=\; \frac{1}{|\mathcal{C}_j|} \sum_{x_i \in \mathcal{C}_j} x_i.
   $$
4. **Iterate**  
   Repeat assignment and update until convergence (centroids stabilize).

---

## How We Used K-Means Here

- **Data**: 12 chemical properties (e.g. acidity, alcohol, sugar) of wine samples.
- **Clusters**: Set $$K=2$$ to discover two groups corresponding to **red** vs. **white** wines.
- **Analysis**:
  1. Ran K-Means on all features to assign each wine to Cluster 0 or Cluster 1.
  2. Plotted feature-wise bar charts of mean values in each cluster.
  3. Interpreted:
     - **Cluster 0**: higher residual sugar & sulfur dioxide → likely **white wines**.
     - **Cluster 1**: higher alcohol & volatile acidity → likely **red wines**.

This unsupervised approach reveals the natural grouping in the data without using the true labels.
