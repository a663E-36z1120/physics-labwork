import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.807  # m/s
m = 0.1999  # grams

dt = 0.01 # s
t_0 = 0.0 # s
Duration = 7.0 # s
v_0 = 0

measured_period = 0.722 # s
omega_0 = 2 * np.pi / measured_period
time_meas_dist, dist_meas, dist_err = np.loadtxt("./position.txt", unpack=True, skiprows=2)
time_meas_vel, vel_meas = np.loadtxt("./velocity.txt", unpack=True, skiprows=2)
y_adj = (max(dist_meas) + min(dist_meas))/2
time_adjusted = time_meas_dist - 0.550
dist_meas_si = (dist_meas - y_adj) / 100
dist_err_si = dist_err / 100
vel_meas_si = vel_meas / 100
y_0 = (max(dist_meas_si) - min(dist_meas_si))/2

k = omega_0 ** 2 * m


# Functions
def euler_y(coordinate, velocity, delta_time=dt):
    return coordinate + delta_time * velocity


def euler_v(velocity, coordinate, ang_freq = omega_0, delta_time=dt):
    return velocity - delta_time * ang_freq ** 2 * coordinate


def symplectic_y(coordinate, velocity, delta_time=dt):
    return coordinate + delta_time * velocity


def symplectic_v(velocity, coordinate, delta_time=dt):
    return velocity - delta_time * k / m * coordinate


def calc_energy(y, v):
    return 1/2 * m * v**2 + 1/2 * k * y**2


time_sim = np.arange(t_0, Duration + dt, dt)
euler_coordinates = np.zeros(len(time_sim))
euler_velocities = np.zeros(len(time_sim))
symplectic_coordinates = np.zeros(len(time_sim))
symplectic_velocities = np.zeros(len(time_sim))

euler_coordinates[0] = y_0
euler_velocities[0] = v_0
symplectic_coordinates[0] = y_0
symplectic_velocities[0] = v_0
for i in range(1, len(time_sim)):
    euler_coordinates[i] = euler_y(euler_coordinates[i - 1], euler_velocities[i - 1])
    euler_velocities[i] = euler_v(euler_velocities[i - 1], euler_coordinates[i - 1])
    symplectic_coordinates[i] = symplectic_y(symplectic_coordinates[i - 1], symplectic_velocities[i - 1])
    symplectic_velocities[i] = symplectic_v(symplectic_velocities[i-1], symplectic_coordinates[i])

energies_sim_euler = calc_energy(euler_coordinates, euler_velocities)
energies_sim_symplectic = calc_energy(symplectic_coordinates, symplectic_velocities)
energies_meas = calc_energy(dist_meas_si, vel_meas_si)


# Plots
plt.plot(time_sim, euler_coordinates, label='Simulated Values (Forward Euler),', zorder=2)
plt.plot(time_sim, symplectic_coordinates, label='Simulated Values (Symplectic Euler),'
         , zorder=2)
plt.plot([], [], linestyle='', label='for $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s')
plt.errorbar(time_adjusted, dist_meas_si, yerr=dist_err_si, markersize=1,
             label='Measured Values', fmt='.', color='black', zorder=1)
plt.ylabel('Displacement from Equilibrium Position y (m)')
plt.xlabel('Time t (s)')
plt.title(f'Displacement vs Time of a Spring Mass')
plt.legend()
plt.savefig('0.png')
plt.show()


plt.plot(time_sim, euler_velocities, label='Simulated Values (Forward Euler),', zorder=2)
plt.plot(time_sim, symplectic_velocities, label='Simulated Values (Symplectic Euler),', zorder=2)
plt.plot([], [], linestyle='', label='for $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s')
plt.errorbar(time_adjusted, vel_meas_si, markersize=1,
             label='Measured Values', fmt='.', color='black', zorder=1)
plt.ylabel('Velocity, v (m/s)')
plt.xlabel('Time, t (s)')
plt.title(f'Velocity vs Time of a Spring Mass')
plt.legend()
plt.savefig('1.png')
plt.show()


plt.plot(time_sim, energies_sim_euler, label='Simulated Values (Forward Euler),', zorder=2)
plt.plot(time_sim, energies_sim_symplectic, label='Simulated Values (Symplectic Euler),', zorder=2)
plt.plot([], [], linestyle='', label='for $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s')
plt.errorbar(time_adjusted, energies_meas, markersize=1, fmt='.',
             label='Measured Values', color='black', zorder=1)
plt.ylabel('Energy, E (J)')
plt.xlabel('Time, t (s)')
plt.title(f'Energy vs Time of a Spring Mass')
plt.legend()
plt.savefig('2.png')
plt.show()

plt.plot(euler_coordinates, euler_velocities, label='Simuated Values (Forward Euler),', zorder=2)
plt.plot(symplectic_coordinates, symplectic_velocities, label='Simuated Values (Symplectic Euler),', zorder=2)
plt.plot([], [], linestyle='', label='for $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s')
plt.xlim([-0.75, 0.75])
plt.ylim([-5, 5])
plt.errorbar(dist_meas_si, vel_meas_si, xerr=dist_err_si, label='Measured Values', fmt='.', color='black', markersize=1, zorder=1)
plt.ylabel('Velocity, v (m/s)')
plt.xlabel('Displacement from Equilibrium Position, y (m)')
plt.title(f'(Full Scale) Phase Plot of the Motion of a Spring Mass')
plt.legend(prop={'size':6})
plt.savefig('3.png')
plt.show()

plt.plot(euler_coordinates, euler_velocities, label='Simuated Values (Forward Euler),', zorder=2)
plt.plot(symplectic_coordinates, symplectic_velocities, label='Simuated Values (Symplectic Euler),', zorder=2)
plt.plot([], [], linestyle='', label='for $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s')
plt.xlim([-0.04, 0.04])
plt.ylim([-0.32, 0.32])
plt.errorbar(dist_meas_si, vel_meas_si, xerr=dist_err_si, label='Measured Values', fmt='.', alpha=0.5, zorder=1, color='black')
plt.ylabel('Velocity, v (m/s)')
plt.xlabel('Displacement from Equilibrium Position, y (m)')
plt.title(f'(Zoomed In) Phase Plot of the Motion of a Spring Mass')
plt.legend(prop={'size':9})
plt.savefig('4.png')
plt.show()
