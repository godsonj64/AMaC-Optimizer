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

## Comparison Table: Adam Optimizer vs. AMaC Optimizer

| Feature                           | Adam Optimizer                                   | AMaC Optimizer                                    |
|-----------------------------------|--------------------------------------------------|--------------------------------------------------|
| **Learning Rate**                 | Fixed or dynamically adjusted                    | Dynamically adjusted with warmup and cosine annealing |
| **First Moment Estimate**         | Yes (moving average of gradients)                | Yes (moving average of gradients)                |
| **Second Moment Estimate**        | Yes (moving average of squared gradients)        | Yes (moving average of squared gradients)        |
| **Bias Correction**               | Yes                                              | Yes                                              |
| **Weight Decay**                  | Fixed                                            | Dynamic                                          |
| **Gradient Clipping**             | No                                               | Yes                                              |
| **Noise Injection**               | No                                               | Yes (adaptive noise injection)                   |
| **Momentum**                      | Optional (e.g., AdamW)                           | Yes (with Lookahead mechanism)                   |
| **Lookahead Mechanism**           | No                                               | Yes                                              |
| **Stochastic Weight Averaging (SWA)** | No                                           | Yes                                              |
| **Learning Rate Warmup**          | Optional                                         | Yes                                              |
| **Gradient Centralization**       | No                                               | Yes                                              |
| **Optimizer Type**                | Adaptive                                         | Adaptive                                         |
| **Parameter Update Rule**         | Updates parameters based on the average of past gradients (momentum) and current gradient direction. | Updates parameters based on a combination of momentum, adaptive learning rates, noise injection, and Lookahead mechanism. |
| **Implementation Complexity**     | Moderate                                         | High                                             |
| **Use Case**                      | General-purpose optimization                     | Enhanced stability and convergence, particularly in noisy or complex optimization landscapes. |

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
![image](https://github.com/godsonj64/AMaC-Optimizer/assets/108623780/7ac74857-7866-44bd-9f34-c89cac9543bc)
![image](https://github.com/godsonj64/AMaC-Optimizer/assets/108623780/e4e85cb6-132d-44b3-ac8b-85046fc7bce6)
![image](https://github.com/godsonj64/AMaC-Optimizer/assets/108623780/7b22be08-1136-4b6b-afa1-2a577c089450)
![image](https://github.com/godsonj64/AMaC-Optimizer/assets/108623780/eef32efd-c4a0-46ab-a39b-14c89c32d190)
![image](https://github.com/godsonj64/AMaC-Optimizer/assets/108623780/92c3e542-2408-491f-8198-9ff1125d7f5e)

```sh
git clone https://github.com/godsonj64/AMaC-Optimizer.git
