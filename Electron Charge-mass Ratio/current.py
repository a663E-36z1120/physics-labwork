import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mu_0 = 4 * math.pi * 10 ** (-7)  # Vacuum permeability
n = 130  # turns (Coil turns)
R = 12.5 / 2 / 100  # m (Coil radius)
R_err = 0.2 / 100


def f(x, a, b):
    return a/x + b


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
I, I_err, diameter, diatmeter_err = \
    np.loadtxt("raw_current.csv", delimiter=',', unpack=True, skiprows=1,
               usecols=range(0, 4))

# radius of electron trajectory in SI units
radius_SI = diameter / 100 / 2
radius_SI_err = diatmeter_err / 100

# Magnetic field generated by the coils
B_c = (4/5) ** (3/2) * mu_0 * n * I / R
B_c_err = ((I_err/I)**2 + (R_err/R)**2 ) ** 0.5 * B_c

# non-linear Regression
popt_nl, pcov_nl = curve_fit(f, I, radius_SI, sigma=radius_SI_err,
                             absolute_sigma=True)
pstd_nl = np.sqrt(np.diag(pcov_nl))


# linear Regression
inverse_radius = 1 / radius_SI
inverse_radius_err = radius_SI_err / radius_SI * inverse_radius

popt_l, pcov_l = curve_fit(g, I, inverse_radius, sigma=inverse_radius_err,
                           absolute_sigma=True)
pstd_l = np.sqrt(np.diag(pcov_l))

popt_B, pcov_B = curve_fit(g, inverse_radius, B_c, sigma=B_c_err,
                           absolute_sigma=True)
pstd_B = np.sqrt(np.diag(pcov_B))


# Predictions
I_range = np.arange(0.8, 2.3, 0.01)
non_lin_predict = (f(I_range, popt_nl[0], popt_nl[1]))
lin_predict = (g(I_range, popt_l[0], popt_l[1]))
inverse_lin_predict = 1/lin_predict
inverse_non_lin_predict = 1/non_lin_predict

inverse_radius_range = np.sort(inverse_radius)
B_c_predict = (g(inverse_radius_range, popt_B[0], popt_B[1]))

# Reduced Chi-squared's
chi_squared_nl = chi_squared(f(I, popt_nl[0], popt_nl[1]),
                             radius_SI, radius_SI_err)
chi_squared_l = chi_squared((g(I, popt_l[0], popt_l[1])),
                            inverse_radius, inverse_radius_err)
chi_squared_inv_l = chi_squared(1/(g(I, popt_l[0], popt_l[1])),
                                radius_SI, radius_SI_err)
chi_squared_inv_nl = chi_squared(1/f(I, popt_nl[0], popt_nl[1]),
                                 inverse_radius, inverse_radius_err)

chi_squared_B = chi_squared(g(inverse_radius, popt_B[0], popt_B[1]), B_c,
                            B_c_err)


# data plots
plt.errorbar(I, radius_SI, xerr=I_err, yerr=radius_SI_err, fmt='.',
             color='black', label='Measured Values')
plt.plot(I_range, non_lin_predict,
         label=f'Non-linear Model\n'
               f'r = ({round(popt_nl[0], 3)}±{round(pstd_nl[0], 3)}) / I + '
               f'({round(popt_nl[1], 3)}±{round(pstd_nl[1], 3)})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_nl, 1)}')

plt.plot(I_range, inverse_lin_predict, linestyle='--',
         label=f'Inverse of Linear Model\n'
               f'r = 1 / [({round(popt_l[0])}±{round(pstd_l[0])}) I + '
               f'({round(popt_l[1])}±{round(pstd_l[1])})]\n'
               f'Reduced $\chi^2$ = {round(chi_squared_inv_l, 1)}')

plt.xlabel('Input Current, I (A)')
plt.ylabel('Electron Circular Trajectory Radius, r (m)')
plt.title('Non-linear Model of Input Current vs Electron Trajectory Radius \n'
          'at (195.0±0.4) Volts Potential Difference')
plt.legend(prop={'size':8})
plt.savefig("current_non-linear.png")
plt.show()

plt.errorbar(I, inverse_radius, xerr=I_err, yerr=inverse_radius_err, fmt='.',
             color='black', label='Measured Values')
plt.plot(I_range, inverse_non_lin_predict,
         label=f'Inverse of Non-linear Model\n'
               f'1/r = 1 / [({round(popt_nl[0], 3)}±{round(pstd_nl[0], 3)}) / I + '
               f'({round(popt_nl[1], 3)}±{round(pstd_nl[1], 3)})]\n'
               f'Reduced $\chi^2$ = {round(chi_squared_inv_nl, 1)}')
plt.plot(I_range, lin_predict, linestyle='--',
         label=f'Linear Model\n'
               f'1/r = ({round(popt_l[0])}±{round(pstd_l[0])}) I + '
               f'({round(popt_l[1])}±{round(pstd_l[1])})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_l, 1)}')

plt.xlabel('Input Current, I (A)')
plt.ylabel('Inverse of Electron Trajectory Radius, 1/r ($m^{-1}$)')
plt.title('Linear Model of Input Current vs Electron Circular Trajectory Radius \n'
          'at (195.0±0.4) Volts Potential Difference')
plt.legend(prop={'size':8})
plt.savefig("current_linear.png")
plt.show()

plt.errorbar(inverse_radius, B_c,
             xerr=inverse_radius_err, yerr=B_c_err, fmt='.',
             color='black', label='Calculated Values')

plt.plot(inverse_radius_range, B_c_predict, linestyle='--',
         label=f'Linear Model\n'
               f'$B_c$ = ({round(popt_B[0], 5):.5f}±{round(pstd_B[0], 5)}) 1/r + '
               f'({round(popt_B[1], 4)}±{round(pstd_B[1], 4)})\n'
               f'Reduced $\chi^2$ = {round(chi_squared_B, 2)}')

plt.xlabel('Inverse of Electron Trajectory Radius, 1/r ($m^{-1}$)')
plt.ylabel('Magnetic Field Generated By Coils, $B_c$ (Tesla)')
plt.title('Linear Model of Magnetic Field Generated by Coils and\n'
          'Inverse Electron Trajectory Radius')
plt.legend(prop={'size':8})
plt.savefig("magnetic_field.png")
plt.show()

