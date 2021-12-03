import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def f(x, a, b):
    return a * x ** 0.5 + b


def g(x, a, b):
    return a * x + b


def chi_squared(predict, meas, uncer, par=2):
    hold = 0
    for i in range(0, len(meas)):
        top = (meas[i] - predict[i]) ** 2
        bottom = uncer[i] ** 2
        hold += top / bottom
    chi_sq = hold / (predict.size - par)
    return chi_sq


# load and process data
V, V_err, diameter, diatmeter_err = \
    np.loadtxt("raw_voltage.csv", delimiter=',', unpack=True, skiprows=1,
               usecols=range(0, 4))

# radius of electron trajectory in SI units
radius_SI = diameter / 100 / 2
radius_SI_err = diatmeter_err / 100

# non-linear Regression
popt_nl, pcov_nl = curve_fit(f, V, radius_SI, sigma=radius_SI_err,
                             absolute_sigma=True)
pstd_nl = np.sqrt(np.diag(pcov_nl))

# linear Regression
root_V = V ** 0.5
root_V_err = 0.5 * V_err / V * root_V

popt_l, pcov_l = curve_fit(g, root_V, radius_SI, sigma=root_V_err,
                           absolute_sigma=True)
pstd_l = np.sqrt(np.diag(pcov_l))

# Predictions
V_range = np.arange(50, 320, 1)
V_root_range = V_range ** 0.5
non_lin_predict = (f(V_range, popt_nl[0], popt_nl[1]))
co_non_lin_predict = (f(V_range, popt_l[0], popt_l[1]))
lin_predict = (g(V_root_range, popt_l[0], popt_l[1]))
co_lin_predict = g(V_root_range, popt_nl[0], popt_nl[1])

# Reduced Chi-squared's
chi_squared_nl = chi_squared(f(V, popt_nl[0], popt_nl[1]),
                             radius_SI, radius_SI_err)
chi_squared_co_nl = chi_squared(f(V, popt_nl[0], popt_nl[1]),
                                radius_SI, radius_SI_err)
chi_squared_l = chi_squared((g(root_V, popt_l[0], popt_l[1])),
                            radius_SI, radius_SI_err)
chi_squared_co_l = chi_squared((g(root_V, popt_l[0], popt_l[1])),
                               radius_SI, radius_SI_err)

# data plots
plt.errorbar(V, radius_SI, xerr=V_err, yerr=radius_SI_err, fmt='.',
             color='black', label='Measured Values')
plt.plot(V_range, non_lin_predict,
         label=f'Non-linear Model\n'
               f'r = ({round(popt_nl[0], 4)}±{round(pstd_nl[0], 4)}) V$^{{1/2}}$ + '
               f'({round(popt_nl[1], 3)}±{round(pstd_nl[1], 3)})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_nl, 1)}')

plt.plot(V_range, co_non_lin_predict, linestyle='--',
         label=f'Linear Model\n'
               f'r = ({round(popt_l[0], 3)}±{round(pstd_l[0], 3)}) V$^{{1/2}}$ + '
               f'({round(popt_l[1], 2)}±{round(pstd_l[1], 2)})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_co_nl, 1)}')

plt.xlabel('Potential Difference, V (V)')
plt.ylabel('Electron Circular Trajectory Radius, r (m)')
plt.title('Non-linear Model of Potential Difference vs Electron Trajectory Radius \n'
          'at (1.25±0.07) Amps Input Current')
plt.legend(prop={'size': 8})
plt.savefig("voltage_non-linear.png")
plt.show()

plt.errorbar(root_V, radius_SI, xerr=root_V_err, yerr=radius_SI_err, fmt='.',
             color='black', label='Measured Values')
plt.plot(V_root_range, lin_predict,
         label=f'Non-linear Model\n'
               f'r = ({round(popt_nl[0], 4)}±{round(pstd_nl[0], 4)}) $\sqrt{{V}}$ + '
               f'({round(popt_nl[1], 3)}±{round(pstd_nl[1], 3)})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_co_l, 1)}')
plt.plot(V_root_range, co_lin_predict, linestyle='--',
         label=f'Linear Model\n'
               f'r = ({round(popt_l[0], 3)}±{round(pstd_l[0], 3)}) $\sqrt{{V}}$ + '
               f'({round(popt_l[1], 2)}±{round(pstd_l[1], 2)})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_l, 1)}')

plt.xlabel('Square Root of Potential Difference, $\sqrt{V}$ ($V^{1/2}$)')
plt.ylabel('Electron Circular Trajectory Radius, r (m)')
plt.title('Linear Model of Potential Difference vs Electron Trajectory Radius \n'
          'at (1.25±0.07) Amps Input Current')
plt.legend(prop={'size': 8})
plt.savefig("voltage_linear.png")
plt.show()
