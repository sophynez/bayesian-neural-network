import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn.functional import softplus
import pandas as pd


# Building blocks
# Bayesian Linear layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features))

        # Prior distributions
        self.weight_prior = dist.Normal(0, 1)
        self.bias_prior = dist.Normal(0, 1)

    def forward(self, x):
        # Reparameterization for weights
        weight_epsilon = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + softplus(self.weight_rho) * weight_epsilon

        # Reparameterization for biases
        bias_epsilon = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + softplus(self.bias_rho) * bias_epsilon

        # Linear transformation
        output = torch.matmul(x, weight.t()) + bias

        return output, dist.Normal(self.weight_mu, softplus(self.weight_rho)), dist.Normal(self.bias_mu, softplus(self.bias_rho))

# Bayesian Neural Network
class BayesianNeuralNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianNeuralNetwork, self).__init__()
        self.bayesian_layer = BayesianLinear(in_features, out_features)

    def forward(self, x):
        return self.bayesian_layer(x)
    

# Define a loss function (e.g., negative log-likelihood plus negative log-prior)
def calculate_loss(output_and_distribution, target, bnn):
    output, weight_distribution, bias_distribution = output_and_distribution
    neg_log_likelihood = torch.nn.functional.mse_loss(output, target)

    # Add negative log-prior terms for weights and biases
    weight_prior = dist.kl.kl_divergence(weight_distribution, bnn.bayesian_layer.weight_prior).sum()
    bias_prior = dist.kl.kl_divergence(bias_distribution, bnn.bayesian_layer.bias_prior).sum()

    # Total loss
    total_loss = neg_log_likelihood + weight_prior + bias_prior

    return total_loss


def fit_bnn(bnn, optimizer, inputs, targets, num_epochs=100):
    """
    Train the Bayesian Neural Network and return the trained model.

    Parameters:
    - bnn (BayesianNeuralNetwork): The Bayesian Neural Network model.
    - optimizer (torch.optim.Optimizer): The optimizer for training.
    - inputs (torch.Tensor): The input data.
    - targets (torch.Tensor): The target values.
    - num_epochs (int): The number of training epochs. Default is 1000.

    Returns:
    - BayesianNeuralNetwork: The trained Bayesian Neural Network model.
    - list: A list containing the training losses for each epoch.
    """
    training_losses = []

    for epoch in range(num_epochs):
        # Forward pass
        output = bnn(inputs)

        # Calculate loss
        loss = calculate_loss(output, targets, bnn)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Parameter update
        optimizer.step()

        # Save the training loss for monitoring
        training_losses.append(loss.item())

        # Print training progress if desired
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

    print("Training complete.")
    return bnn, training_losses


data = pd.read_csv('data\international-airline-passengers.csv')
data['year'] = data['Month'].str.split('-').str[0].astype(int)
data['month'] = data['Month'].str.split('-').str[1].astype(int)
data = data.drop('Month', axis=1)

# print(data)

features = data[['year', 'month']].values
target = data['Passengers'].values

inputs = torch.tensor(features, dtype=torch.float32)
targets = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

in_features = 2  # the input features: 'year' and 'month'
out_features = 1  # target variable: 'passengers' 

bnn = BayesianNeuralNetwork(in_features, out_features)
optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)

trained_bnn, train_losses = fit_bnn(bnn, optimizer, inputs, targets, num_epochs=100)

