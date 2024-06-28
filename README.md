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

```sh
git clone https://github.com/godsonj64/AMaC-Optimizer.git
