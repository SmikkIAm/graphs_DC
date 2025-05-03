# Scientific Visualizations with Python

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0%2B-blueviolet)

This repository contains three Python implementations demonstrating scientific and mathematical concepts through visualization.

## Projects

### 1. Cooling with Oscillating Ambient Temperature
**File**: `cooling_oscillation.py`   

Models temperature dynamics of an object in an oscillating environment:
- Solves: dT/dt = -k(T - T_env(t))
- Features:
  - Analytical solution of non-homogeneous ODE
  - Direction field visualization
  - Dynamic plot scaling
  - Professional LaTeX labels

### 2. Wealth Dynamics with Progressive Taxation
**File**: `wealth_dynamics.py`   

Models wealth accumulation with taxation:
- Solves: dW/dt = rW - τ₀W²
- Features:
  - Logistic equation solution
  - Direction field with scaling
  - Multiple trajectories
  - Equilibrium analysis

### 3. Rocket Velocity Simulation
**File**: `rocket_simulation.py`  
**Visualization**:  

Compares numerical methods for rocket dynamics:
- Models:
  - Gravity (g)
  - Air resistance (k)
  - Mass expulsion (α)
- Methods compared:
  - Euler
  - Improved Euler
  - RK4

## Installation

```bash
pip install numpy matplotlib scipy
