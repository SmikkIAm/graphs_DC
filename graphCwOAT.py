import numpy as np
import matplotlib.pyplot as plt

# Parameters
T_initial = 100    # Initial object temp (°C) 
k = 0.1            # Cooling rate (hr^-1)
T0 = 10            # Mean ambient temp (°C)
A = 5              # Oscillation amplitude (°C)
omega = 2*np.pi/24 # Angular frequency (daily cycle)
days = 3           # Number of days   

# Time grid 
t = np.linspace(0, 24*days, 1000)  

# Particular solution coefficients
B = A*k/(k**2 + omega**2)
D = -A*omega/(k**2 + omega**2)

# Integration constant to satisfy initial condition
C = T_initial - T0 - B*np.sin(omega*0) - D*np.cos(omega*0)

# Complete solution
T = T0 + C*np.exp(-k*t) + B*np.sin(omega*t) + D*np.cos(omega*t)

# Ambient temperature
T_env = T0 + A*np.sin(omega*t)

# ODE for direction field
def cooling_ode(T_val, t_val):
    return k*(T0 + A*np.sin(omega*t_val) - T_val)

# Dynamic plot ranges
T_min = min(T_env.min(), T.min()) - 5
T_max = max(T_env.max(), T.max(), T_initial) + 10

# Create figure
plt.figure(figsize=(14, 8))

# Plot direction field
t_grid, T_grid = np.meshgrid(
    np.linspace(0, 24*days, 20),
    np.linspace(T_min, T_max, 20)
)
slopes = cooling_ode(T_grid, t_grid)
plt.quiver(t_grid, T_grid, 
           np.ones_like(t_grid), slopes,
           color='gray', scale=250, width=0.002, alpha=0.6)

# Plot temperatures
plt.plot(t, T, label=f'Object temperature $T(t)$', 
         color='blue', linewidth=2.5, zorder=3)
plt.plot(t, T_env, '--', label='Ambient $T_{env}(t)$', 
         color='red', linewidth=2, zorder=2)

# Mark initial temperature
plt.scatter(0, T_initial, color='green', s=100, zorder=4,
            label=f'Initial temp ({T_initial}°C)')

# Formatting
plt.xticks(np.arange(0, 24*days+1, 12), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Time (hours)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(0, 24*days)
plt.ylim(T_min, T_max)

plt.title(f'Cooling with Oscillating Ambient Temperature)', 
          fontsize=18, pad=20)
plt.tight_layout()
plt.show()