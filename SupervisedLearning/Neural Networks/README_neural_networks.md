## Neural Networks
A neural network is a parameterized, layered function approximator inspired by biological neurons. It maps an input vector $\mathbf{x}\in\mathbb{R}^d$ to an output $\hat{\mathbf{y}}$ through a sequence of linear transforms and nonlinear activations. Modern deep networks stack many layers to learn hierarchical representations and solve tasks ranging from classification and regression to generative modeling.

- **Configurable Architecture**  
  - Accepts an arbitrary list of layer sizes (e.g. `[input_dim, hidden1, …, output_dim]`).

- **Parameter Initialization**  
  - Xavier (Glorot) uniform initialization for weights, zeros for biases.

- **Forward Pass**  
  - Computes successive pre-activations $z^{(\ell)} = W^{(\ell)}a^{(\ell-1)} + b^{(\ell)}$  
  - Applies sigmoid nonlinearity $\,a^{(\ell)} = \sigma(z^{(\ell)})$.

- **Loss & Metrics**  
  - Binary cross-entropy loss for training.  
  - Tracks both loss and RMSE each epoch to monitor convergence.

- **Backpropagation**  
  - Derives deltas via $\delta^{(\ell)} = (W^{(\ell+1)})^T\delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})$.  
  - Computes weight/bias gradients and updates with learning rate $\eta$.

- **Evaluation Tools**  
  - `plot_rmse()`: RMSE vs. epoch curve.  
  - `roc_auc_analysis()`: ROC curve + AUC score.  
  - `plot_confusion_matrix()`: final confusion matrix.

- **Reliability**  
  - Built-in unit tests verify decreasing loss/RMSE, correct shapes, and perfect fit on simple separable data.
