import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time as time

# Separar processamento de post proc

# Método da projeção
inicio=time.time()
Re=1000
t_final = 20
tol = 1e-6
Lx = 1.0
Ly = 1.0
Nx = 50
Ny = 50

# SOR !

omega = 1.5

dt=1e-4
dx = Lx/Nx
dy = Ly/Ny

x=np.linspace(0,Lx,Nx+1)
y=np.linspace(0,Ly,Ny+1)

esp=-(1/dx**2 + 1/dy**2)


u = np.zeros((Nx+1, Ny+2))
v = np.zeros((Nx+2, Ny+1))
p = np.zeros((Nx+2, Ny+2))
U=np.zeros(Nx+1)
for i in range(Nx+1):
    U[i] = 1.0

for i in range(Nx+1):
  u[i,Ny]=2 * U[i]
u_star = np.copy(u)
v_star = np.copy(v)

t = 0.0

# Bellow is the code to calculate most properties of the flow, Numba was used to speed this up, you can just remove it if you dont want it. (remove njit)

##########################################################################################

#U star

@njit
def calcular_u_star(u, v, u_star, Nx, Ny, dx, dy, Re, dt):
    for i in range(1,Nx):
        for j in range(0,Ny):
            v_int = 0.25 * (v[i,j+1] + v[i-1,j+1] + v[i,j] + v[i-1,j])
            adv = (u[i,j] * (u[i+1,j]-u[i-1,j])/(2 * dx) + v_int * (u[i,j+1]-u[i,j-1])/(2 * dy))
            visc = (u[i+1,j]-2 * u[i,j] + u[i-1,j]) * 1/(Re*dx**2) + (u[i,j+1] - 2 * u[i,j] + u[i,j-1]) * 1/(Re * dy**2)
            u_star[i,j] = u[i,j] + dt * (visc-adv)

    for i in range(0,Nx+1):
        u_star[i,-1] = -u_star[i,0]
        u_star[i,Ny] = 2.0 - u_star[i,Ny-1]

    return u_star
@njit
def calcular_v_star(u, v, v_star, Nx, Ny, dx, dy, Re, dt):
    for i in range(0,Nx):
        for j in range(1,Ny):
            u_int = 0.25 * (u[i+1,j] + u[i,j] + u[i+1,j-1] + u[i,j-1])
            adv = u_int * (v[i+1,j] - v[i-1,j]) / (2*dx) + v[i,j] * (v[i,j+1] - v[i,j-1]) / (2*dy)
            visc = (v[i+1,j]-2 * v[i,j] + v[i-1,j]) * 1/(Re*dx**2) + (v[i,j+1] - 2 * v[i,j] + v[i,j-1]) * 1/(Re * dy**2)
            v_star[i,j] = v[i,j] + dt * (visc-adv)

    for j in range(0,Ny+1):
        v_star[-1,j] = -v_star[0,j]
        v_star[Nx,j] = -v_star[Nx-1,j]

    return v_star
@njit
def resolver_pressao(p, u_star, v_star, Nx, Ny, dx, dy, dt, omega, tol):
    error = 100

    while error > tol:
        r_max = 0.0
        for i in range(0, Nx):
            for j in range(0, Ny):
                div = (u_star[i+1, j] - u_star[i, j]) / (dt*dx) + (v_star[i, j+1] - v_star[i, j]) / (dt*dy)

                if   i == 0   and j == 0:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r = div - ((p[i+1, j] - p[i, j]) / dx**2 + (p[i, j+1] - p[i, j]) / dy**2)

                elif i == 0   and j == Ny-1:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r  = div - ((p[i+1, j] - p[i, j]) / dx**2 + (- p[i, j] + p[i, j-1]) / dy**2)

                elif i == Nx-1 and j == 0:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r  = div- ((- p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - p[i, j]) / dy**2)

                elif i == Nx-1 and j == Ny-1:
                    Lambda = -(1/dx**2 + 1/dy**2)
                    r  = div -  ((- p[i, j] + p[i-1, j]) / dx**2 + (- p[i, j] + p[i, j-1]) / dy**2)

                elif i == 0 and 0 < j < Ny-1:
                    Lambda = -(1/dx**2 + 2/dy**2)
                    r = div - ((p[i+1, j] - p[i, j]) / dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2)

                elif i == Nx-1 and 0 < j < Ny-1:
                    Lambda = -(1/dx**2 + 2/dy**2)
                    r  = div - ((- p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2)

                elif j == 0 and 0 < i < Ny-1:
                    Lambda = -(2/dx**2 + 1/dy**2)
                    r  = div - ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - p[i, j]) / dy**2)

                elif j == Nx-1 and 0 < i < Ny-1:
                    Lambda = -(2/dx**2 + 1/dy**2)
                    r  = div - ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 + (- p[i, j] + p[i, j-1]) / dy**2)

                else:
                    Lambda = -(2/dx**2 + 2/dy**2)
                    r = div -  ((p[i+1, j] - 2*p[i, j] + p[i-1, j]) / dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1]) / dy**2)

                r = r / Lambda
                p[i, j] += omega * r

                if abs(r) > r_max:
                    r_max = abs(r)

        error = r_max

# Boundary Update

    for i in range(Nx):
        p[i, -1] = p[i,  0]
        p[i,  Ny] = p[i, Ny-1]
    for j in range(Ny):
        p[-1, j] = p[ 0, j]
        p[Nx,  j] = p[Nx-1, j]

    p[-1, -1] = p[0, 0]
    p[-1,  Ny] = p[0, Ny-1]
    p[Nx,  -1] = p[Nx-1, 0]
    p[Nx,   Ny] = p[Nx-1, Ny-1]

    return p
@njit
def corrigir_u(u, u_star, p, Nx, Ny, dx, dt):
    for i in range(Nx):
        for j in range(Ny+1):
            u[i,j] = u_star[i,j] - dt * (p[i,j] - p[i-1,j])/dx
    return u
@njit
def corrigir_v(v, v_star, p, Nx, Ny, dy, dt):
    for i in range(-1,Nx):
        for j in range(1,Ny):
            v[i,j]= v_star[i,j] - dt * (p[i,j] - p[i,j-1])/dy
    return v

# This block calculates stream function, uncomment it if you want it, but I'll warn you, this makes the code painfully slow.
"""   
def calcular_stream_function(u, v, Nx, Ny, dx, dy, tol, omega):
    psi = np.zeros((Nx+1,Ny+1))
    Lambda =  -(2/dx**2 + 2/dy**2)
    error_stream_line_function = 100

    while error_stream_line_function > tol:
        r_max = 0.0
        for i in range(1,Nx):
            for j in range(1,Ny):
                r =-( (v[i,j] - v[i-1,j]) ) /dx + ( (u[i,j] - u[i,j-1]) ) /dy - ((psi[i+1,j]-2*psi[i,j] + psi[i-1,j])/dx**2 + (psi[i,j+1]-2*psi[i,j]+psi[i,j-1])/dy**2)
                r=r/Lambda
                psi[i,j] = psi[i,j] + omega * r

                if np.abs(r) > r_max:
                    r_max = np.abs(r)

        error_stream_line_function = r_max

    return psi
"""

##########################################################################################################

while t < t_final:
    u_star = calcular_u_star(u, v, u_star, Nx, Ny, dx, dy, Re, dt)
    v_star = calcular_v_star(u, v, v_star, Nx, Ny, dx, dy, Re, dt)
    p = resolver_pressao(p, u_star, v_star, Nx, Ny, dx, dy, dt, omega, tol)
    u = corrigir_u(u, u_star, p, Nx, Ny, dx, dt)
    v = corrigir_v(v, v_star, p, Nx, Ny, dy, dt)
  #psi = calcular_stream_function(u, v, Nx, Ny, dx, dy, tol, omega)
    t=t+dt
    #print(time.time()-inicio)
#print(psi)

uplot = np.zeros((Nx+1, Ny+1), float)
vplot = np.zeros((Nx+1, Ny+1), float)
vel_mag = np.zeros((Nx+1, Ny+1), float)

# Compute averaged velocities and velocity magnitude

for i in range(0, Nx+1):
    for j in range(0, Ny+1):
        uplot[i, j] = 0.5 * (u[i, j] + u[i, j-1])
        vplot[i, j] = 0.5 * (v[i, j] + v[i-1, j])
        vel_mag[i, j] = np.sqrt(uplot[i, j]**2 + vplot[i, j]**2)
for j in range(0, Ny):
    vel_mag[Nx,j]=0


print(time.time()-inicio) #Print final time for the simulation




X, Y = np.meshgrid(x,y, indexing='ij')  # Index i e j to match arrays


# Plotting the results:

################ Plot vlocity


contour = plt.contourf(X, Y, vel_mag, 100, cmap='jet') 

plt.streamplot(
    x, y,
    uplot.T, vplot.T,
    color='white',
    density=2,
    linewidth=0.8
)

plt.colorbar(contour, label='Velocity vector magnitude')
plt.title('Velocity field (mesh {}x{}) - Re = {}'.format(Nx, Ny, Re))
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()


############## Plot pressure
"""
p -= np.mean(p)

plt.figure(figsize=(8, 6))
vmin = np.min(p[:-1,:-1])
vmax = np.max(p[:-1,:-1])
lim = max(abs(vmin), abs(vmax))

contour = plt.contourf(X, Y, p[:-1,:-1], 100, cmap='seismic', vmin=-lim, vmax=lim)
plt.colorbar(contour, label='Pressão')

plt.title('Pressure field (mesh {}x{}) - Re = {}'.format(Nx, Ny, Re))
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()


"""
"""
################# Vorticity - Uncomment if and only if this insterests you.

# Calcular vorticidade
omega = np.zeros((Nx+1, Ny+1))

for i in range(1, Nx):
    for j in range(1, Ny):
        dudy = (uplot[i, j+1] - uplot[i, j-1]) / (2*dy)
        dvdx = (vplot[i+1, j] - vplot[i-1, j]) / (2*dx)
        omega[i, j] = dvdx - dudy

# zeroing boundary
omega[0, :] = 0
omega[-1, :] = 0
omega[:, 0] = 0
omega[:, -1] = 0

# Plot 
plt.figure(figsize=(8,6))
contour = plt.contourf(X, Y, omega, 100, cmap='magma')
plt.colorbar(contour, label='Vorticidade')
plt.title('Campo de Vorticidade - Re = {}'.format(Re))
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.show()

"""
