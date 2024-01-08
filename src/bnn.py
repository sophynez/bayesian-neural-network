import torch
import torch.nn as nn
import torch.distributions as dist


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters represented as distributions
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features))

        # Prior distributions
        self.weight_prior = dist.Normal(0, 1)
        self.bias_prior = dist.Normal(0, 1)

    def forward(self, x):
        # Sample weights and biases during training
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + torch.log(1 + torch.exp(self.weight_rho)) * weight_epsilon
        bias = self.bias_mu + torch.log(1 + torch.exp(self.bias_rho)) * bias_epsilon
        # Compute output
        output = torch.matmul(x, weight.t()) + bias
        return output

# Example usage:
input_size = 10
output_size = 1
bnn = BayesianLinear(input_size, output_size)
input_data = torch.randn(100, input_size)
output_data = torch.randn(100, output_size)

# Training loop
optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = bnn(input_data)
    # Define negative log-likelihood and negative log-prior
    neg_log_likelihood = -dist.Normal(predictions, 1).log_prob(output_data).sum()
    neg_log_prior = - (bnn.weight_prior.log_prob(bnn.weight_mu).sum() +
                      bnn.weight_prior.log_prob(bnn.bias_mu).sum())
    # Loss is negative log posterior
    loss = neg_log_likelihood + neg_log_prior
    loss.backward()
    optimizer.step()

# Inference
with torch.no_grad():
    test_input = torch.randn(10, input_size)
    predictions = torch.stack([bnn(test_input) for _ in range(100)])
    mean_prediction = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)




