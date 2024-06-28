# AMaC-Optimizer
The AMaC (Adaptive Momentum and Cosine Annealing) Optimizer is a custom PyTorch optimizer that combines several advanced optimization techniques to improve training stability and performance.

# AMaC Optimizer-Vision

AMaC integrates multiple advanced optimization techniques into a single algorithm to enhance neural network training. By combining adaptive momentum estimation, cosine annealing, SWA, and other methods, AMaC achieves faster convergence, better stability, and improved generalization

## Features

- **Learning Rate Warm-up**: Gradual increase in learning rate at the beginning of training.
- **Cosine Annealing**: Smooth decrease in learning rate.
- **Gradient Centralization**: Centering gradients to improve stability.
- **Momentum Updates**: Using moving averages of the gradient and squared gradient.
- **Adaptive Noise Injection**: Injecting noise into updates to escape local minima.
- **Gradient Clipping**: Clipping gradients to prevent exploding.
- **Dynamic Weight Decay**: Adjusting weight decay dynamically during training.
- **Lookahead Mechanism**: Periodically updating slow weights with the current weights.
- **Stochastic Weight Averaging (SWA)**: Averaging model weights over time for smoother training trajectory.

## Usage

### Installation

Clone this repository and include the `AMaC_optimizer.py` in your project.
#### Sample Train Results Using MNIST 
Epoch 1/10, Loss: 1.1543, Accuracy: 69.09%
Epoch 2/10, Loss: 0.4018, Accuracy: 87.40%
Epoch 3/10, Loss: 0.2845, Accuracy: 91.22%
Epoch 4/10, Loss: 0.2040, Accuracy: 93.73%
Epoch 5/10, Loss: 0.1585, Accuracy: 95.17%
Epoch 6/10, Loss: 0.1323, Accuracy: 96.00%
Epoch 7/10, Loss: 0.1124, Accuracy: 96.55%
Epoch 8/10, Loss: 0.0996, Accuracy: 97.06%
Epoch 9/10, Loss: 0.0929, Accuracy: 97.23%
Epoch 10/10, Loss: 0.0898, Accuracy: 97.33%
```sh
git clone https://github.com/godsonj64/AMaC-Optimizer.git
