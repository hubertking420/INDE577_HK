# DBSCAN (Density‐Based Spatial Clustering of Applications with Noise)

**DBSCAN** is an unsupervised clustering algorithm that groups points based on density:

## Definitions:
- **Core Point**: A point with at least `min_samples` neighbors within radius `ε`.  
- **Border Point**: A non-core point that lies within `ε` of a core point.  
- **Noise Point**: Any point that is neither core nor border.
## Algorithm Steps
1. **Neighborhood Query**  
   For each unvisited point $p$, retrieve its ε-neighborhood:
   $$
   N_\varepsilon(p) = \{q \mid \|p - q\| \le \varepsilon\}.
   $$
2. **Core Check**  
   - If $|N_\varepsilon(p)| \ge \text{min\_samples}$, label $p$ a **core** point and start a new cluster.  
   - Otherwise, label $p$ **noise** (may later become border).
3. **Cluster Expansion**  
   For each core point, recursively add all points in its ε-neighborhood (cores → cores → …) and their border points to the cluster.
4. **Repeat**  
   Continue until all points are visited and assigned to either a cluster or labeled noise.

---

## Parameters

- **ε (eps)**: Radius of neighborhood around each point.  
- **min_samples**: Minimum number of points required to form a dense region.

---

## Strengths

- **Detects Arbitrary Shapes**: Finds clusters of any shape, not just spherical.  
- **Noise Detection**: Explicitly labels outliers.  
- **No $K$ Required**: Automatically determines the number of clusters.

---

## Limitations

- **Parameter Sensitivity**: Choosing `ε` and `min_samples` can be nontrivial.  
- **Varying Density**: Struggles when clusters have different densities.  
- **Scalability**: Neighborhood queries can be expensive for large datasets.

---

## Usage in Our Wine Analysis

- **Goal**: Discover dense regions in 12-dimensional chemical‐feature space (red vs. white wines) without predefining $K$.  
- **Outcome**:  
  - Identified one large
