import numpy as np
import matplotlib.pyplot as plt

# constants
SAMPLE_RATE = 10  # Hertz
PIXEL_DISPLACEMENT_CONVERSION = 540  # pixels per mm
PIXEL_DISPLACEMENT_CONVERSION_ERROR = 1  # pixels per mm
VOLTAGE_ERROR = 5  # Volts
e_REFERENCE = 1.60217662 * 10 ** (-19)


# models
def get_Q_m1(v1, U):
    return 2 * 10 ** (-10) * (v1 ** (3 / 2)) / U


def get_Q_m1_err(v1, v1_err, U, U_err):
    return ((3 / 2 * v1_err / v1) ** 2 + (U_err / U) ** 2) ** 0.5 * get_Q_m1(v1, U)


def get_Q_m2(v1, v2, U):
    return 2 * 10 ** (-10) * (v1 + v2) * (v2 ** (1 / 2)) / U


def get_Q_m2_err(v1, v1_err, v2, v2_err, U, U_err):
    return ((v1_err ** 2 + v2_err ** 2) / ((v1 + v2) ** 2) + (
            1 / 2 * v2_err / v2) ** 2 + (U_err / U) ** 2) ** 0.5 * get_Q_m2(v1, v2, U)


def gcd_finding_algorithm(Q_data: np.ndarray, error_threshold: float,
                          accuracy_param: float) -> np.ndarray:
    """
    Q_data: Calculated charge values

    error_threshold: Threshold above which gcd will be considered valid (For
                     instance, gcd smaller than the uncertainty of charge
                     will not be considered valid)

    accuracy_param: Since np.gdc operates on ints, this parameter determines to
                    which decimal point we round the calculated charge to when
                    calculating gcd.
    """

    # Prepares array for integer operation
    Q_int = Q_data * accuracy_param

    # Calculate gcd of every permutation of Q value pairs
    lst = []
    for n in Q_int:
        for m in Q_int:
            gcd = np.gcd(round(n), round(m))

            # Discard gcds smaller than the error threshold
            if gcd > error_threshold * accuracy_param:
                lst.append(gcd)
    min_gcd = min(lst)
    rlst = []

    # Account for integer multiples of gcd for some permuations of
    # Q value pairs
    for n in lst:
        rlst.append(n / round(n / min_gcd))

    return np.array(rlst) / accuracy_param


# load and process data
trial, V_suspend, V_rise, pixel_rate_up, pixel_rate_down = \
    np.loadtxt("data.csv", delimiter=',', unpack=True, skiprows=1)

V_err = np.full(len(trial), VOLTAGE_ERROR)

vel_up = - pixel_rate_up / PIXEL_DISPLACEMENT_CONVERSION / 1000 * SAMPLE_RATE
vel_up_err = (
                     PIXEL_DISPLACEMENT_CONVERSION_ERROR / PIXEL_DISPLACEMENT_CONVERSION) * vel_up
vel_down = pixel_rate_down / PIXEL_DISPLACEMENT_CONVERSION / 1000 * SAMPLE_RATE
vel_down_err = (
                       PIXEL_DISPLACEMENT_CONVERSION_ERROR / PIXEL_DISPLACEMENT_CONVERSION) * vel_down

Q_m1 = get_Q_m1(vel_down, V_suspend)
Q_m1_err = get_Q_m1_err(vel_down, vel_down_err, V_suspend, V_err)
Q_m2 = get_Q_m2(vel_down, vel_up, V_rise)
Q_m2_err = get_Q_m2_err(vel_down, vel_down_err, vel_up, vel_up_err, V_rise,
                        V_err)

Q_m1_gcd = gcd_finding_algorithm(Q_m1, max(Q_m1_err), 10 ** 22)
Q_m2_gcd = gcd_finding_algorithm(Q_m2, max(Q_m2_err), 10 ** 22)

# plots
binwidth = 1 * 10 ** -20
plt.hist(Q_m1, bins=np.arange(min(Q_m1), max(Q_m1) + binwidth, binwidth), alpha=0.8, label='Method 1')
plt.hist(Q_m2, bins=np.arange(min(Q_m2), max(Q_m2) + binwidth, binwidth), alpha=0.8, label='Method 2')
plt.yticks(np.arange(0, 3, 1))
plt.ylabel('Frequency')
plt.xlabel('Calculated Charge, Q (Coulombs)')
plt.title('Histogram of Distribution of Calculated Charge on \nMeasured Oil Droplets According to Both Methods')
plt.legend()
plt.savefig('hist1.png')
plt.show()

binwidth = 1 * 10 ** -21
plt.hist(Q_m1_gcd, bins=np.arange(0 * e_REFERENCE, 2 * e_REFERENCE + binwidth, binwidth), alpha=0.8,
         label=f'Method 1\n$\mu$={round(np.mean(Q_m1_gcd)*10**19, 2)}e-19, $\sigma$={round(np.std(Q_m1_gcd, ddof=1)*10**19, 2)}e-19')
plt.hist(Q_m2_gcd, bins=np.arange(0 * e_REFERENCE, 2 * e_REFERENCE + binwidth, binwidth), alpha=0.8,
         label=f'Method 2\n$\mu$={round(np.mean(Q_m2_gcd)*10**19, 2)}e-19, $\sigma$={round(np.std(Q_m2_gcd, ddof=1)*10**19, 2)}e-19')
plt.xticks(np.arange(0, 2 * e_REFERENCE, e_REFERENCE))
plt.xlim(0, 1.7 * 10**-19)
plt.ylabel('Frequency')
plt.xlabel('Estimated Values of $e$ (Coulombs)')
plt.title(f'Histogram of Distribution of Estimated Values of $e$ \nAccording to Both Methods '
          'from the GCD-finding Algorithm')
plt.legend()
plt.savefig('hist2.png')
plt.show()
