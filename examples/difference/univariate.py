import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm

from torchdensityestimation.difference import lsdd_fit, lsdd_predict

# define the two univiariate normal distributions
mean_x = 0
mean_y = 1
std = 1

# generate samples from the two distributions
n_samples = 50
rng = np.random.default_rng(0)
x = torch.from_numpy(
    norm.rvs(size=n_samples, loc=mean_x, scale=std, random_state=rng)
).unsqueeze(1)
y = torch.from_numpy(
    norm.rvs(size=n_samples, loc=mean_y, scale=std, random_state=rng)
).unsqueeze(1)

# # fit the LSDD model
mdl = lsdd_fit(x, y, seed=int(rng.integers(0, 2**32)))

# estimate the density difference over the real line
n_vals = 1_000
vals = torch.linspace(-5, 5, n_vals, dtype=x.dtype)
predicted = lsdd_predict(mdl, vals.unsqueeze(1))

# compute also the true density ratio for sake of the comparison
vals_np = vals.numpy()
pdf_x = norm.pdf(vals_np, mean_x, std)
pdf_y = norm.pdf(vals_np, mean_y, std)
target = pdf_x - pdf_y

# plot the results
_, ax = plt.subplots(1, 1, constrained_layout=True)
ax.plot(vals_np, target)
ax.plot(vals_np, predicted.numpy(), linestyle="--")
ax.set_title("True and LSDD density difference")
plt.show()
