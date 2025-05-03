import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.1                # growth rate coefficient
tax0 = 0.002            # tax coefficient
W_eq = r / tax0         # equilibrium wealth (50 kUSD in this case)
years = 100             # Time period
W0_low = 10             # kUSD (below equilibrium)
W0_high = 100           # kUSD (above equilibrium)

# Time grid for direction field
time_points = np.linspace(0, years, 15)      
wealth_points = np.linspace(W0_low-5, W0_high+10, 15)   
time_grid, wealth_grid = np.meshgrid(time_points, wealth_points)

# ODE for direction field
dWdt = r * wealth_grid - tax0 * wealth_grid**2
dtdt = np.ones_like(dWdt) 

# Fine time grid for analytical solutions
fine_time_grid = np.linspace(0, years, 500)

# Analytical solution function
def analytical_solution(W0, t):
    return (r * W0) / (tax0 * W0 + (r - tax0 * W0) * np.exp(-r * t))

# Initial wealth values for example trajectories
W0_eq = W_eq    # kUSD (exactly at equilibrium)

# Calculate trajectories
growing_solution = analytical_solution(W0_low, fine_time_grid)
decaying_solution = analytical_solution(W0_high, fine_time_grid)
equilibrium_solution = analytical_solution(W0_eq, fine_time_grid)

# Create figure
plt.figure(figsize=(14, 8))

# Plot direction field
plt.quiver(time_grid, wealth_grid, 
           dtdt, dWdt,
           color='gray', scale=1/1.5, width=0.0025, alpha=0.6,
           angles='xy', scale_units='xy', pivot='mid')

# Plot trajectories
plt.plot(fine_time_grid, growing_solution, '-', 
         label=f'Growth ($W_0={W0_low}$ kUSD)',
         color='green', linewidth=2.5, zorder=3)
plt.plot(fine_time_grid, decaying_solution, '-', 
         label=f'Decay ($W_0={W0_high}$ kUSD)',
         color='red', linewidth=2.5, zorder=3)

# Plot equilibrium line
plt.axhline(W_eq, ls='--', 
            label=rf'Equilibrium $W_{{eq}} = r/\tau_0 = {W_eq:.0f}$ kUSD',
            color='black', linewidth=2, alpha=0.7, zorder=2)

# Formatting
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Time (years)', fontsize=16)
plt.ylabel('Wealth (kUSD)', fontsize=16)
from matplotlib.lines import Line2D
param_legend = [
    Line2D([0], [0], color='w', marker='o', markerfacecolor='w', markersize=0,
           label=fr'$r = {r}$: Growth rate coefficient'),
    Line2D([0], [0], color='w', marker='o', markerfacecolor='w', markersize=0,
           label=fr'$\tau_0 = {tax0}$: Tax coefficient')
]
handles, labels = plt.gca().get_legend_handles_labels()
legend_handles = param_legend + handles
plt.legend(handles=legend_handles, 
           fontsize=11,
           title_fontsize=12,
           framealpha=0.9,
           loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(0, years)
plt.ylim(0, max(decaying_solution) * 1.1) 

plt.title('Wealth Dynamics with Progressive Taxation', 
          fontsize=18, pad=20)
plt.tight_layout()
plt.show()