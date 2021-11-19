import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# Load data
exp_sample_num, exp_count = np.loadtxt("Fiesta_30092021.txt",
                                       unpack=True, skiprows=2)
bkgd_sample_num, bkgd_count = np.loadtxt("Fiesta_Background_30092021.txt",
                                         unpack=True, skiprows=2)

# Data Processing
mean_bkgd_count = np.average(bkgd_count)

count_rate = (exp_count - mean_bkgd_count) #/ 6
bkgd_count_rate = bkgd_count #/ 6
count_rate_uncert = ((exp_count ** 0.5 / 6) ** 2 + (bkgd_count ** 0.5 / 6) ** 2)
time = exp_sample_num * 6

mu = np.average(count_rate)
sigma = np.std(count_rate, ddof=1)
mu_bkgd = np.average(bkgd_count_rate)
sigma_bkgd = np.std(bkgd_count_rate)

# Poisson
# poisson_x = np.arange(10, 31)
poisson_x = np.arange(80, 161)
poisson_y = poisson.pmf(poisson_x, mu=mu)
poisson_x_bkgd = np.arange(0, 10)
poisson_y_bkgd = poisson.pmf(poisson_x_bkgd, mu=mu_bkgd)

# Gaussian
# norm_x = np.arange(10, 30, 0.1)
norm_x = np.arange(80, 160, 0.1)
norm_y = norm.pdf(norm_x, loc=mu, scale=sigma)
norm_x_bkgd = np.arange(0, 9, 0.01)
norm_y_bkgd = norm.pdf(norm_x_bkgd, loc=mu_bkgd, scale=sigma_bkgd)

# Plots
count, edges, patches = plt.hist(count_rate, bins=13, density=True, rwidth=0.94, label='Measured Data Normalized Histogram')
plt.bar(poisson_x, poisson_y, color='black', width=0.2, zorder=2)
plt.scatter(poisson_x, poisson_y, color='black', marker='.', zorder=3, label=f'Poisson Distribution PMF'
                                                                 f'\n(mu = {round(mu, 3)})')
plt.plot(norm_x, norm_y, label=f'Gaussian Distribution PDF\n'
                                             f'(mu = {round(mu, 3)}; sigma = {round(sigma, 3)})', zorder=4)
plt.legend(prop={'size': 7})
plt.title('Distribution of Geiger Counter Click Counts Measuring a Fiesta Plate')
plt.xlabel('Counts over 6 seconds')
plt.ylabel('Probability Density or Probability Mass')
plt.savefig('random_decay_data.png')
plt.show()
plt.close()

count_bkgd, edges_bkgd, patches_bkgd = plt.hist(bkgd_count_rate, bins=9, density=True, rwidth=0.94, label='Measured Data Normalized Histogram')
plt.bar(poisson_x_bkgd, poisson_y_bkgd, color='black', width=0.06, zorder=2)
plt.scatter(poisson_x_bkgd, poisson_y_bkgd, color='black', zorder=3, label=f'Poisson Distribution PMF'
                                                                 f'\n(mu = {round(mu_bkgd, 3)})')
plt.plot(norm_x_bkgd, norm_y_bkgd, label=f'Gaussian Distribution PDF\n'
                                             f'(mu = {round(mu_bkgd, 3)}; sigma = {round(sigma_bkgd, 3)})', zorder=4)
plt.legend(prop={'size': 7})
plt.title('Distribution of Geiger Counter Click Counts Measuring\nBackground Radiation')
plt.xlabel('Counts over 6 seconds')
plt.ylabel('Probability Density or Probability Mass')
plt.savefig('random_decay_background.png')
plt.show()
plt.close()



