import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvnorm

from torchdensityratio import rulsif_fit, rulsif_predict

# define the two multivariate normal distributions
n_dim = 2
mean = np.ones(n_dim)
cov_x = np.eye(n_dim) / 8
cov_y = np.eye(n_dim) / 2

# generate samples from the two distributions
n_samples = 3000
rng = np.random.default_rng(0)
x = torch.from_numpy(mvnorm.rvs(size=n_samples, mean=mean, cov=cov_x, random_state=rng))
y = torch.from_numpy(mvnorm.rvs(size=n_samples, mean=mean, cov=cov_y, random_state=rng))

# fit the RuLSIF model
alpha = 0.0
sigmas = torch.as_tensor([0.1, 0.3, 0.5, 0.7, 1.0], dtype=x.dtype)
lambdas = torch.as_tensor([0.01, 0.02, 0.03, 0.04, 0.05], dtype=x.dtype)
mdl = rulsif_fit(x, y, alpha, sigmas, lambdas, seed=int(rng.integers(0, 2**32)))

# estimate the density ratio over a 2d grid
n_vals = 200
vals = torch.linspace(0, 2, n_vals, dtype=x.dtype)
grid = torch.dstack(torch.meshgrid(vals, vals, indexing="xy")).reshape(-1, 2)
predicted = rulsif_predict(mdl, grid).reshape(n_vals, n_vals)

# compute also the true density ratio for sake of the comparison
grid_np = grid.numpy()
pdf_x = mvnorm.pdf(grid_np, mean, cov_x)
target = pdf_x / (alpha * pdf_x + (1 - alpha) * mvnorm.pdf(grid_np, mean, cov_y))
target = target.reshape(n_vals, n_vals)

# plot the results
vals_np = vals.numpy()
levels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5]
_, axs = plt.subplots(1, 2, constrained_layout=True)
axs[0].contourf(vals_np, vals_np, target, levels=levels)
axs[0].set_title("True density ratio")
axs[1].contourf(vals_np, vals_np, predicted.numpy(), levels=levels)
axs[1].set_title("RuLSIF density ratio")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.show()
