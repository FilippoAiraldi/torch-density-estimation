import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

from torchdensityestimation.ratio import rulsif_fit, rulsif_predict

# define the two univariate normal distributions
mean = 0
std_x = 1 / 8
std_y = 1 / 2

# generate samples from the two distributions
n_samples = 500
rng = np.random.default_rng(0)
x = torch.from_numpy(
    norm.rvs(size=n_samples, loc=mean, scale=std_x, random_state=rng)
).unsqueeze(1)
y = torch.from_numpy(
    norm.rvs(size=n_samples, loc=mean, scale=std_y, random_state=rng)
).unsqueeze(1)

# fit the RuLSIF model
alpha = 0.1
mdl = rulsif_fit(x, y, alpha, seed=int(rng.integers(0, 2**32)))

# estimate the density ratio over the real line
n_vals = 200
vals = torch.linspace(-1, 2, n_vals, dtype=x.dtype)
predicted = rulsif_predict(mdl, vals.reshape(-1, 1))

# compute also the true density ratio for sake of the comparison
vals_np = vals.numpy()
pdf_x = norm.pdf(vals_np, mean, std_x)
target = pdf_x / (alpha * pdf_x + (1 - alpha) * norm.pdf(vals_np, mean, std_y))

# plot the results
_, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(vals_np, target)
ax.plot(vals_np, predicted.numpy(), linestyle="--")
ax.set_title("True and RuLSIF density ratio")
plt.show()
