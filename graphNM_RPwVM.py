from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# === Physical and Problem Constants ===
g = 9.81       # Acceleration due to gravity (m/s²)
k = 0.15       # Air resistance coefficient (kg/m)
alpha = 20     # Mass expulsion rate (kg/s)
u = 2000       # Exhaust velocity (m/s)
m0 = 1000      # Initial mass of the rocket (kg)

# === Time Domain for Simulation ===
t0 = 0         # Initial time (seconds)
tf = 20        # Final time (seconds)

# === Numerical Methods Implementations ===

def euler(N):
    """Basic Euler method implementation"""
    h = (tf - t0)/N
    t = np.linspace(t0, tf, N+1)
    v = np.zeros(N+1)
    for i in range(N):
        m = m0 - alpha * t[i]
        v[i+1] = v[i] + h * (-g - (k/m) * v[i]**2 + (alpha * u) / m)
    return t, v

def improved_euler(N):
    """Improved Euler (Heun's) method implementation"""
    h = (tf - t0)/N
    t = np.linspace(t0, tf, N+1)
    v = np.zeros(N+1)
    for i in range(N):
        # Current mass
        m = m0 - alpha * t[i]
        
        # Predictor step (Euler)
        v_pred = v[i] + h * (-g - (k/m) * v[i]**2 + (alpha * u) / m)
        
        # Corrector step (using mass at next time step)
        m_next = m0 - alpha * (t[i] + h)
        v_corr = -g - (k/m_next) * v_pred**2 + (alpha * u) / m_next
        
        # Average of current and corrected slope
        v[i+1] = v[i] + (h/2) * (
            (-g - (k/m) * v[i]**2 + (alpha * u) / m) + v_corr
        )
    return t, v

def rk4(N):
    """Runge-Kutta 4th order method implementation"""
    h = (tf - t0)/N
    t = np.linspace(t0, tf, N+1)
    v = np.zeros(N+1)
    for i in range(N):
        m = m0 - alpha * t[i]
        k1 = h * (-g - (k/m) * v[i]**2 + (alpha * u) / m)

        m = m0 - alpha * (t[i] + h/2)
        k2 = h * (-g - (k/m) * (v[i] + k1/2)**2 + (alpha * u) / m)

        k3 = h * (-g - (k/m) * (v[i] + k2/2)**2 + (alpha * u) / m)

        m = m0 - alpha * (t[i] + h)
        k4 = h * (-g - (k/m) * (v[i] + k3)**2 + (alpha * u) / m)

        v[i+1] = v[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, v

# === Solution Comparison Plot ===
for N in [5, 10, 20, 40]:
    t_euler, v_euler = euler(N)
    t_imp, v_imp = improved_euler(N)
    t_rk4, v_rk4 = rk4(N)
    t_exact, v_exact = rk4(10000)

    plt.figure(figsize=(10, 6))
    plt.plot(t_exact, v_exact, label="Exact (RK4 N=10000)", color='blue')
    plt.plot(t_euler, v_euler, '--', label=f"Euler (N={N})", color='red')
    plt.plot(t_imp, v_imp, ':', label=f"Improved Euler (N={N})", color='purple')
    plt.plot(t_rk4, v_rk4, '-.', label=f"RK4 (N={N})", color='green')
    plt.title("Velocity vs. Time (Solution Comparison)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Error Analysis ===

# Error vs Step Size (Log-Log)
Ns = [5, 10, 20, 40, 80, 160]
h_vals = [(tf - t0)/N for N in Ns]
euler_errors = []
imp_errors = []
rk4_errors = []

for N in Ns:
    _, v_e = euler(N)
    _, v_imp = improved_euler(N)
    _, v_rk = rk4(N)
    t_ref, v_ref = rk4(10000)
    
    # Interpolate reference solution to match current time points
    interp_func = interp1d(t_ref, v_ref, kind='cubic')
    t_current = np.linspace(t0, tf, N+1)
    v_ref_interp = interp_func(t_current)
    
    euler_errors.append(np.max(np.abs(v_e - v_ref_interp)))
    imp_errors.append(np.max(np.abs(v_imp - v_ref_interp)))
    rk4_errors.append(np.max(np.abs(v_rk - v_ref_interp)))

plt.figure(figsize=(10, 6))
plt.loglog(h_vals, euler_errors, 'o--', label="Euler Error", color='red')
plt.loglog(h_vals, imp_errors, 'd-.', label="Improved Euler Error", color='purple')
plt.loglog(h_vals, rk4_errors, 's-', label="RK4 Error", color='green')
plt.title("Error vs. Step Size (Log-Log)")
plt.xlabel("Step Size (h)")
plt.ylabel("Max Absolute Error")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

# Error Over Time (N=20)
N = 20
t = np.linspace(t0, tf, N+1)
_, v_euler = euler(N)
_, v_imp = improved_euler(N)
_, v_rk4 = rk4(N)
t_ref, v_ref = rk4(10000)
interp_func = interp1d(t_ref, v_ref, kind='cubic')
v_ref_interp = interp_func(t)

plt.figure(figsize=(10, 6))
plt.plot(t, np.abs(v_euler - v_ref_interp), '--', label="Euler Error", color='red')
plt.plot(t, np.abs(v_imp - v_ref_interp), ':', label="Improved Euler Error", color='purple')
plt.plot(t, np.abs(v_rk4 - v_ref_interp), '-.', label="RK4 Error", color='green')
plt.title(f"Error Evolution Over Time (N={N})")
plt.xlabel("Time (s)")
plt.ylabel("Absolute Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.yscale("log")
plt.show()