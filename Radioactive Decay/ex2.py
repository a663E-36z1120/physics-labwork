import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def f(x, a, b):
    return a * x + b


def g(x, a, b):
    return b * np.e ** (a * x)


# Caesium 20s
# Plate 6s

def chi_squared(predict, meas, uncer, par=2):
    hold = 0
    for i in range(0, len(meas)):
        top = (meas[i] - predict[i]) ** 2
        bottom = uncer[i] ** 2
        hold += top / bottom
    chi_sq = hold / (predict.size - par)
    return chi_sq


exp_sample_num, exp_count = np.loadtxt("./Experiment1_30092021.txt",
                                       unpack=True, skiprows=2)
bkgd_sample_num, bkgd_count = np.loadtxt("./Background_30092021.txt",
                                           unpack=True, skiprows=2)
mean_bkgd_count = np.average(bkgd_count)

count_rate = (exp_count - mean_bkgd_count) / 20
count_rate_uncert = ( (exp_count ** 0.5 / 20) ** 2 + (bkgd_count ** 0.5 /20) ** 2) ** 0.5
time = exp_sample_num * 20


# Non-linear Regression
popt_nl, pcov_nl = curve_fit(g, time, count_rate, sigma=count_rate_uncert,
                             absolute_sigma=True, p0=[1/200, 60])
pstd_nl = np.sqrt(np.diag(pcov_nl))

# Linear Regression
log_count_rate = np.log(count_rate)
log_count_rate_uncert = (1 / count_rate) * count_rate_uncert

popt_l, pcov_l = curve_fit(f, time, log_count_rate, sigma=log_count_rate_uncert,
                           absolute_sigma=True, p0=[-200, 1200])
pstd_l = np.sqrt(np.diag(pcov_l))

# Predictions
non_lin_predict = g(time, popt_nl[0], popt_nl[1])
lin_predict = f(time, popt_l[0], popt_l[1])
non_lin_predict_log = np.log(non_lin_predict)
lin_predict_e = np.e ** lin_predict
#
# Reduced Chi-squared's
chi_squared_nl = chi_squared(non_lin_predict, count_rate, count_rate_uncert)
chi_squared_l = chi_squared(lin_predict, log_count_rate, log_count_rate_uncert)
chi_squared_nl_l = chi_squared(non_lin_predict_log, log_count_rate, log_count_rate_uncert)
chi_squared_l_e = chi_squared(lin_predict_e, count_rate, count_rate_uncert)

# Plots
# Non-linear
plt.errorbar(time, count_rate, yerr=count_rate_uncert, fmt='.', label='Measured Values')
plt.plot(time, non_lin_predict, marker='.', label=f'Predicted Values (Non Linear Model)\n'
                                                    f'I(t) = ({round(popt_nl[1], 1)}±{round(pstd_nl[1], 1)}) * e^(({round(popt_nl[0], 5)}±{round(pstd_nl[0], 5)}) t)\n'
                                                    f'Reduced Chi^2 = {round(chi_squared_nl,1)}')
plt.plot(time, lin_predict_e, marker='.', linestyle=':', label=f'Predicted Values (Linear Model in Exponential (base e) Scale)\n'
                                                            f'I(t) = e^(({round(popt_l[0], 5)}±{round(pstd_l[0], 5)})t + ({round(popt_l[1], 2)}±{round(pstd_l[1], 2)}))\n'
                                                            f'Reduced Chi^2 = {round(chi_squared_l_e,1)}')
plt.xlabel('Time (s)')
plt.ylabel('Geiger Counter Click Rate, I (Counts/s)')
plt.title('Geiger Counter Click Rate vs Time Non-linear Plot')
plt.legend(prop={'size':7.5})
plt.savefig("ex2_nn_plot.png")
plt.show()

# Linear
plt.errorbar(time, log_count_rate, yerr=log_count_rate_uncert, fmt='.', label='Measured Values')
plt.plot(time, lin_predict, marker='.', linewidth=1, label=f'Predicted Values (Linear Model)\n'
                                                    f'ln(I(t)) = ({round(popt_l[0], 5)}±{round(pstd_l[0], 5)})t + ({round(popt_l[1], 2)}±{round(pstd_l[1], 2)})\n'
                                                    f'Reduced Chi^2 = {round(chi_squared_l,1)}')
plt.plot(time, non_lin_predict_log, marker='.', linestyle=':', label=f'Predicted Values (Non-linear Model in log (base e) Scale)\n'
                                                    f'ln(I(t)) = ln(({round(popt_nl[1], 1)}±{round(pstd_nl[1], 1)}) * e^(({round(popt_nl[0], 5)}±{round(pstd_nl[0], 5)}) t))\n'
                                                    f'Reduced Chi^2 = {round(chi_squared_nl_l,1)}')#

plt.xlabel('Time, t (s)')
plt.ylabel('Geiger Counter Click Rate in log (base e) Scale, ln(I (Counts/s))')
plt.title('Geiger Counter Click Rate vs Time linear Plot')
plt.legend(prop={'size':7.5})
plt.savefig("ex2_plot_lin.png")
plt.show()


