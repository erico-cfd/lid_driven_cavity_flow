# Lid-Driven Cavity Flow Solver (Projection Method)

This repository contains a high-performance 2D CFD solver for the **Lid-Driven Cavity** problem. The simulation captures the development of a primary vortex and secondary eddies within a square enclosure driven by a moving top boundary.

---

## Governing Equations

The solver solves the incompressible Navier-Stokes equations in non-dimensional form, characterized by the Reynolds Number ($Re$).

### 1. Momentum Equation
The velocity field $u = (u, v)$ is governed by:

$$
\frac{\partial u}{\partial t} + (u \cdot \nabla) u = -\nabla p + \frac{1}{Re} \nabla^2 u
$$

### 2. Continuity Equation (Incompressibility)
The flow must satisfy the divergence-free constraint:

$$
\nabla \cdot u = 0
$$

---

## Numerical Implementation

* **Algorithm:** **Projection Method**. 
    1. An intermediate velocity field $u^*$ is calculated by accounting for advection and diffusion.
    2. A **Pressure Poisson Equation** is solved to enforce incompressibility: $\nabla^2 p = \frac{\nabla \cdot u^*}{\Delta t}$.
    3. The velocity is corrected using the pressure gradient.
* **Pressure Solver:** Uses the **Successive Over-Relaxation (SOR)** iterative method with an acceleration parameter $\omega = 1.5$.
* **Performance:** Critical calculation loops are decorated with `@njit` (**Numba**) to compile Python code into machine code, significantly reducing execution time.
* **Grid:** Employs a **Staggered MAC Grid** to ensure pressure-velocity coupling and avoid numerical oscillations.

---

## Configuration

* **Reynolds Number ($Re$):** $1000$
* **Grid Resolution:** $50 \times 50$ (Adjustable via `Nx`, `Ny`)
* **Time Step ($\Delta t$):** $10^{-4}$
* **Boundary Conditions:**
    * **Top Wall ($y=L$):** Moving lid ($U=1.0$).
    * **Bottom/Side Walls:** No-slip ($u=0, v=0$).

---

## Usage

### Dependencies
Ensure you have the following installed:
* `numpy`
* `matplotlib`
* `numba`

### Running the Solver
Execute the Python script to start the simulation:
```bash
python Lid_Driven_Cavity_Flow.py
