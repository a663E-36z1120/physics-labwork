import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def f(x, a, b):
    return a * x + b


def chi_squared(predict, meas, uncer, par=2):
    hold = 0
    for i in range(0, len(meas)):
        top = (meas[i] - predict[i])**2
        bottom = uncer[i]**2
        hold += top/bottom
    chi_sq = hold/(predict.size-par)
    return chi_sq


I, I_uncert, dip_V, freq, freq_uncert = \
    np.loadtxt("data.csv", delimiter=',', unpack=True, skiprows=1, usecols=range(1,6))

popt, pcov = curve_fit(f, I, freq, sigma=freq_uncert, absolute_sigma=True)
pstd = np.sqrt(np.diag(pcov))

predict = f(I, popt[0], popt[1])
chi_squared_nl = chi_squared(predict, freq, freq_uncert)

plt.errorbar(I, freq, xerr=I_uncert, yerr=freq_uncert, label='Measured Values',
             fmt='.', color='black', markersize=1, zorder=1)
plt.plot(I, predict, label=f'Predicted Values\n'
                                                    f'$\\nu$ = ({round(popt[0])}±{round(pstd[0])})I + ({round(popt[1])}±{round(pstd[1])})\n'
                                                    f'Reduced $\chi^2$ = {round(chi_squared_nl, 1)}')

plt.title('Current Applied to Coil vs Electron Spin Resonance'
          '\nFrequency of DPPH Molecules')
plt.xlabel('Current, I (A)')
plt.ylabel('Resonance Frequency, $\\nu$ (MHz)')
plt.legend()
plt.savefig('fig.png')
plt.show()
