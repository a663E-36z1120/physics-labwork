import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.807  # m/s
m = 0.2011  # grams

dt = 0.01 # s
t_0 = 0.0 # s
Duration = 135 # s
v_0 = 0
gamma = 0.0266

measured_period = 6.8/9 # s
omega_0 = 2 * np.pi / measured_period
time_meas_dist, dist_meas, dist_err = np.loadtxt("./position_damped.txt", unpack=True, skiprows=2)
time_meas_vel, vel_meas = np.loadtxt("./velocity_damped.txt", unpack=True, skiprows=2)
y_adj = (max(dist_meas) + min(dist_meas))/2
time_adjusted = time_meas_dist - 0.290
dist_meas_si = (dist_meas - y_adj) / 100
dist_err_si = dist_err / 100
vel_meas_si = vel_meas / 100
y_0 = (max(dist_meas_si) - min(dist_meas_si))/2

k = omega_0 ** 2 * m

# Functions

def symplectic_y(coordinate, velocity, delta_time=dt):
    return coordinate + delta_time * velocity

def symplectic_v(velocity, coordinate, damping, delta_time=dt):
    return velocity - delta_time * k / m * coordinate + damping * delta_time

def damping(velocity, coordinate, delta_time=dt):
    return -((omega_0) ** 2) * coordinate - gamma * velocity

def calc_energy(y, v):
    return 1/2 * m * v**2 + 1/2 * k * y**2

def envolope(t):
    return max(dist_meas_si) * np.e ** -(t * gamma / 2)

def energy_decay(t):
    return max(energies_sim_symplectic) * np.e ** -(t * gamma)



time_sim = np.arange(t_0, Duration + dt, dt)

symplectic_coordinates = np.zeros(len(time_sim))
symplectic_velocities = np.zeros(len(time_sim))
dampings = np.zeros(len(time_sim))

symplectic_coordinates[0] = y_0
symplectic_velocities[0] = v_0
dampings[0] = 0
for i in range(1, len(time_sim)):
    dampings[i] = damping(symplectic_velocities[i-1], symplectic_coordinates[i-1])
    symplectic_velocities[i] = symplectic_v(symplectic_velocities[i-1], symplectic_coordinates[i], dampings[i])
    symplectic_coordinates[i] = symplectic_y(symplectic_coordinates[i - 1], symplectic_velocities[i])





energies_sim_symplectic = calc_energy(symplectic_coordinates, symplectic_velocities)
energies_meas = calc_energy(dist_meas_si, vel_meas_si)

envolope_pred = envolope(time_adjusted)
energy_pred = energy_decay(time_adjusted)



# Plots
plt.plot(time_sim, symplectic_coordinates, label='Simulated Values (Symplectic Euler),\nfor $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s'
         , zorder=2, linewidth=0.4)
plt.plot(time_adjusted, envolope_pred, label='Amplitude Envolope $y =\pm A e^{-(\gamma t)/2}$,\nfor $\gamma=$' + f'{round(gamma,3)} and A={round(max(dist_meas_si),3)}'
         , zorder=2, linewidth=3, ls='-', color='orange')
plt.plot(time_adjusted, -envolope_pred, label='_Hidden', zorder=2, linewidth=3, ls='-', color='orange')
plt.errorbar(time_adjusted, dist_meas_si, markersize=0.5,
             label='Measured Values', fmt='.', color='black', zorder=1)
plt.ylim([-0.1, 0.1])
plt.ylabel('Displacement from Equilibrium Position, y (m)')
plt.xlabel('Time, t (s)')
plt.title(f'Displacement vs Time of a Damped Spring Mass')
plt.legend(prop={'size':8})
plt.savefig('i.png')
plt.show()


plt.plot(time_sim, energies_sim_symplectic, label='Simulated Values (Symplectic Euler),\nfor $\omega_0=$' + f'{round(omega_0,1)} rad/s and T={round(measured_period,2)} s',
         zorder=2,
         linewidth=0.4)
plt.plot(time_adjusted, energy_pred, label='Model Prediction\n$E = A e^{-\gamma t}$, for $\gamma=$' + f'{round(gamma,3)} and A={round(max(energy_pred),3)}'
         , zorder=3, linewidth=3, ls='--')
plt.errorbar(time_adjusted, energies_meas, markersize=0.5, fmt='.',
             label='Measured Values', color='black', zorder=1)
plt.ylabel('Energy, E (J)')
plt.xlabel('Time, t (s)')
plt.title(f'Energy vs Time of a Damped Spring Mass')
plt.legend()
plt.savefig('ii.png')
plt.show()


