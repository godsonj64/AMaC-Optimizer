import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import numpy as np


# Define the AMaC Optimizer class
class AMaC(optim.Optimizer):
    def __init__(self, params, total_steps, lr=0.001, beta1=0.9, beta2=0.999, mu=0.9, eps=1e-8, wd=0.01, alpha=0.5, k=5,
                 clip_value=1.0, warmup_steps=1000, swa_start=10, noise_factor=0.01):
        self.total_steps = total_steps  # Store total_steps as an instance variable
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, mu=mu, eps=eps, wd=wd, alpha=alpha, k=k, clip_value=clip_value,
                        warmup_steps=warmup_steps, swa_start=swa_start, noise_factor=noise_factor)
        super(AMaC, self).__init__(params, defaults)
        self.swa_weights = []
        self.swa_start = swa_start
        self.swa_n = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)
                    state['slow_weights'] = torch.clone(p.data).detach()
                    state['warmup_lr'] = group['lr'] / group['warmup_steps']
                m, v, u, slow_weights = state['m'], state['v'], state['u'], state['slow_weights']
                beta1, beta2, mu, eps, wd, alpha, k, clip_value, warmup_steps, noise_factor = group['beta1'], group[
                    'beta2'], group['mu'], group['eps'], group['wd'], group['alpha'], group['k'], group['clip_value'], \
                group['warmup_steps'], group['noise_factor']
                state['step'] += 1
                t = state['step']
                if t <= warmup_steps:
                    lr = state['warmup_lr'] * t
                else:
                    lr = group['lr'] * (
                                0.5 * (1 + math.cos(math.pi * (t - warmup_steps) / (self.total_steps - warmup_steps))))
                grad -= grad.mean()
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                delta = m_hat / (v_hat.sqrt() + eps)
                noise_level = noise_factor * (1 - t / (2 * warmup_steps))
                noise = noise_level * torch.randn_like(delta)
                delta.add_(noise)
                delta = torch.clamp(delta, -clip_value, clip_value)
                weight_decay = wd * (1 - t / (2 * warmup_steps))
                p.data.mul_(1 - lr * weight_decay)
                u.mul_(mu).add_(delta, alpha=-lr)
                p.data.add_(u)
                if t % k == 0:
                    slow_weights.mul_(alpha).add_(p.data, alpha=(1 - alpha))
                    p.data.copy_(slow_weights)
                if t >= warmup_steps and t % k == 0:
                    if len(self.swa_weights) == 0:
                        self.swa_weights = [torch.zeros_like(p.data) for p in group['params']]
                    for swa_w, p in zip(self.swa_weights, group['params']):
                        swa_w.mul_(self.swa_n / (self.swa_n + 1)).add_(p.data / (self.swa_n + 1))
                    self.swa_n += 1
        return loss

    def swap_swa_weights(self):
        for group in self.param_groups:
            for swa_w, p in zip(self.swa_weights, group['params']):
                p.data.copy_(swa_w)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Calculate total steps
def calculate_total_steps(loader, epochs):
    num_batches = len(loader)
    total_steps = num_batches * epochs
    return total_steps


num_epochs = 10
total_steps = calculate_total_steps(train_loader, num_epochs)

# Initialize the model, optimizer, and loss function
model = SimpleNet()
optimizer = AMaC(model.parameters(), total_steps=total_steps, lr=0.0001, clip_value=0.5, noise_factor=0.005, wd=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
train_losses = []
train_accuracies = []
gradient_magnitudes = []
momentum_updates = []
first_moments = []
second_moments = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect gradient magnitudes
        grad_magnitude = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_magnitude += p.grad.data.norm().item()
        gradient_magnitudes.append(grad_magnitude)

        # Collect momentum updates, first and second moments
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'u' in state:
                    momentum_updates.append(state['u'].norm().item())
                if 'm' in state:
                    first_moments.append(state['m'].norm().item())
                if 'v' in state:
                    second_moments.append(state['v'].norm().item())

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')


# Visualize the collected data
def plot_lr_schedule(warmup_steps, total_steps, initial_lr):
    steps = np.arange(total_steps)
    lr = np.zeros_like(steps, dtype=np.float32)
    for t in steps:
        if t <= warmup_steps:
            lr[t] = initial_lr / warmup_steps * t
        else:
            lr[t] = initial_lr * (0.5 * (1 + np.cos(math.pi * (t - warmup_steps) / (total_steps - warmup_steps))))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lr, label='Learning Rate')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_accuracy(train_loss, train_acc, val_loss=None, val_acc=None):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Training Accuracy')
    if val_acc is not None:
        plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_gradient_magnitude(gradients):
    steps = range(1, len(gradients) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, gradients, label='Gradient Magnitude')
    plt.xlabel('Steps')
    plt.ylabel('Magnitude')
    plt.title('Gradient Magnitude Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_moments(first_moments, second_moments):
    steps = range(1, len(first_moments) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, first_moments, label='First Moment (m)')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('First Moment Estimate Over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, second_moments, label='Second Moment (v)')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Second Moment Estimate Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_momentum_update(momentum_updates):
    steps = range(1, len(momentum_updates) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, momentum_updates, label='Momentum Update (u)')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Momentum Update Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the collected data
plot_lr_schedule(warmup_steps=1000, total_steps=total_steps, initial_lr=0.0005)
plot_loss_accuracy(train_losses, train_accuracies)
plot_gradient_magnitude(gradient_magnitudes)
plot_moments(first_moments, second_moments)
# Assuming we collect SWA accuracy separately, if not, this function can be omitted
# plot_swa_effect(base_accuracy, swa_accuracy)
plot_momentum_update(momentum_updates)




