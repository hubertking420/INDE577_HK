# Ensemble Learning

Ensembling combines multiple “weak” models to form a stronger overall predictor:

- **Bagging** (e.g. Random Forest)  
  Trains each model on a bootstrap sample and averages (or votes) to reduce **variance**.  
- **Boosting** (e.g. AdaBoost, Gradient Boosting)  
  Trains models **sequentially**, each one focusing on the mistakes of its predecessor to reduce **bias**.  
- **Stacking**  
  Learns a “meta-model” to combine the outputs of diverse base learners.

**Key Benefit:**  
By aggregating different perspectives or correcting errors iteratively, ensembles often generalize better than any single model.

---

# Introducing XGBoost

Extreme Gradient Boosting (XGBoost) is a high-performance implementation of gradient boosting:

1. **Sequential Learning**  
   Each new tree fits the **residuals** of the ensemble so far:  
   $$
   F_m(x) = F_{m-1}(x) + \nu\,h_m(x)\,,
   $$  
   where $\nu$ is the learning rate.  
2. **Gradient & Hessian Information**  
   Uses first and second derivatives of the loss $\mathcal{L}$ to choose optimal splits.  
3. **Regularization**  
   Penalizes tree complexity via  
   $$
   \Omega(h) = \gamma T + \tfrac12\lambda\sum w_j^2
   $$  
   to prevent overfitting.  
4. **Efficiency & Scalability**  
   Optimized C++ core, built-in parallelism, and out‐of‐core computation for large datasets.

**Why XGBoost?**  
It marries the power of boosting with advanced optimization and regularization, delivering state-of-the-art accuracy and speed on many real-world tasks.  
