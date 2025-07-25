import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Use double precision for better numerical stability
dtype = torch.double

# Objective function (the black-box function we want to optimize)
def objective(X):
    return (6 * X - 2)**2 * torch.sin(12 * X - 4)

# Generate initial training data
train_X = torch.rand(10, 1, dtype=dtype)
train_Y = objective(train_X)

# Fit the Gaussian Process model
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Create test inputs for plotting
test_X = torch.linspace(0, 1, 100, dtype=dtype).unsqueeze(-1)
true_Y = objective(test_X).detach()

# Model predictions
model.eval()
posterior = model.posterior(test_X)
pred_mean = posterior.mean.detach()
pred_std = posterior.variance.sqrt().detach()

# Acquisition function
acq_func = LogExpectedImprovement(model, best_f=train_Y.max())
acq_values = acq_func(test_X.unsqueeze(1)).detach().squeeze()

# Optimize acquisition to suggest next point
candidate, _ = optimize_acqf(
    acq_func,
    bounds=torch.tensor([[0.0], [1.0]], dtype=dtype),
    q=1,
    num_restarts=5,
    raw_samples=20,
)

# Plot
plt.figure(figsize=(10, 6))

# Plot the true objective
plt.plot(test_X.numpy(), true_Y.numpy(), label="True Objective f(x)", color="black", linestyle="--")

# GP prediction mean and uncertainty
plt.plot(test_X.numpy(), pred_mean.numpy(), label="GP Prediction", color="blue")
plt.fill_between(
    test_X.squeeze().numpy(),
    (pred_mean - pred_std).squeeze().numpy(),
    (pred_mean + pred_std).squeeze().numpy(),
    alpha=0.3,
    label="Uncertainty (±1 std)",
    color="blue"
)

# Acquisition function (scaled)
acq_scaled = acq_values / acq_values.max() * true_Y.max()
plt.plot(test_X.numpy(), acq_scaled.numpy(), label="LogExpectedImprovement (scaled)", color="green")

# Training points
plt.scatter(train_X.numpy(), train_Y.numpy(), label="Train Points", color="red")

# Suggested next point
plt.axvline(candidate.item(), color="purple", linestyle="--", label=f"Next Suggestion: {candidate.item():.3f}")

plt.title("Bayesian Optimization with BoTorch")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("botorch_plot.png")

