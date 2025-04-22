import numpy as np
#import math
import json
import matplotlib.pyplot as plt
import pandas as pd
import time
print('-----------MULTIPLE DROPLET SIMULATION-----------')

#region ####### FUNCTION DEFINITIONS ####################
def load_data(file_path='data.json'):
    with open(file_path, 'r') as file:
        return json.load(file) 

def calc_turbulent_timescale(k, epsilon, C_mu):         # Calculates turbulent length and time scales
    L_L = C_mu**(3/4) * k**(3/2) / epsilon
    T_L = L_L / np.sqrt(2/3 * k)
    print(f'\n\nThe Lagrangian time scale = {T_L:.5f} seconds \n \n')
    return (L_L, T_L)

def calc_particle_Reynolds_no(rho_g, d_p, k, mu_g):     # Calculates particle Reynolds number
    Re_p = rho_g * d_p * np.sqrt(2 * k) / mu_g
    print(f'Particle Reynolds no. = {Re_p:.2f} \n \n')
    return Re_p

def calc_particle_relax_time(rho_p, d_p, mu_g, Re_p):   # Calculates the particle relaxation time with a correction based on Re
    tau_p_prime = rho_p * d_p**2 / (18 * mu_g)
    tau_p = tau_p_prime / (1 + 0.15 * Re_p**0.687 )
    print(f'Particle relaxation time = {tau_p:.5f} seconds \n \n')
    return tau_p 

def calc_wiener_increment(dt, N_p):                     # Calculates the Wiener increment in multiple directions
    dW = np.sqrt(dt) * np.random.normal(0, 1, N_p)      # Generates N_p random values used for Wiener increment        
    return dW

def calc_u_g_p(dW, dt, T_L, u_g_p_prev):            # Calculates the updated gas velocity at the particle
    u_g_p = u_g_p_prev - u_g_p_prev * dt / T_L + np.sqrt( 4*k / (3*T_L) ) * dW
    
    return u_g_p

def calc_u_p(u_g_p, u_p_prev, dt, tau_p):           
    u_p = u_g_p - ( u_g_p - u_p_prev)*np.exp(-dt/tau_p)  
    
    return u_p

def calc_x_p(x_p_prev, u_p, dt, L_c):
    x_p = x_p_prev + u_p*dt

    for i in range(0, N_p):
        if x_p[i] > L_c:
            x_p[i] -= L_c
        if x_p[i] < 0:
            x_p[i] += L_c

    return x_p

def update_res(u_g_p_prev, u_p_prev, x_p_prev, i, dt, tau_p, L_c, N_p):
    dW[i] = calc_wiener_increment(dt, N_p)
    ens_avg_dW[i] = np.mean(dW[i])
    ens_var_dW[i] = np.var(dW[i])

    u_g_p[i] = calc_u_g_p(dW[i], dt, T_L, u_g_p_prev)
    ens_avg_ugp[i] = np.mean(u_g_p[i])
    ens_var_ugp[i] = np.var(u_g_p[i])

    u_p[i] = calc_u_p(u_g_p[i], u_p_prev, dt, tau_p)
    ens_avg_up[i] = np.mean(u_p[i])
    ens_var_up[i] = np.var(u_p[i])

    x_p[i] = calc_x_p(x_p_prev, u_p[i], dt, L_c)
    ens_avg_xp[i] = sum(x_p[i])/N_p
# endregion

data = load_data()
start_time = time.time()

## Assign variables__________________________________
# Simulation parameters
L_c = data['simulation_params']['domain_size']      # cube domain size [m]
k = data['simulation_params']['turb_kin_energy_k']    
epsilon = data['simulation_params']['dissipation_rate_e']
C_mu = data['simulation_params']['C_mu_constant']
T = data['simulation_params']['sim_time']           # simulation duration [s]

# Droplet parameters
N_p = 1000                                           # number of droplets in simulation
d_p = data['droplet_params']['diameter_p']          # droplet diameter [m]
rho_p = data['droplet_params']['density_p']         # droplet density [kgm-3]

# Gas parameters
rho_g = data['gas_params']['density_gas']               # gas density [kgm-3]
mu_g = data['gas_params']['viscosity']                  # gas viscosity [kg/ms]

L_L, T_L    = calc_turbulent_timescale(k, epsilon, C_mu)
Re_p        = calc_particle_Reynolds_no(rho_g, d_p, k, mu_g)
tau_p       = calc_particle_relax_time(rho_p, d_p, mu_g, Re_p)
dt          = 0.05 * T_L               # Timestep defined as 1/20 of lagrangian timescale
N           = int(T / dt)               # Number of iterations to solve
t           = np.linspace(0, T, N)
index_settled = int(10*T_L / dt)             # Assumes flow settles after 10*T_L

print(f'Simulation timestep dt = {dt:.5f} seconds \n')

# Preallocate arrays 2D
u_g_p       = np.zeros((N, N_p))            # u'_g@p
u_p         = np.zeros((N, N_p))            # u'_p
x_p         = np.zeros((N, N_p))            # x_p
dW          = np.zeros((N, N_p))            # dW
ens_avg_dW  = np.zeros(N)            # ensemble averages 
ens_avg_ugp = np.zeros(N)
ens_avg_up  = np.zeros(N)
ens_avg_xp  = np.zeros(N)
ens_var_dW  = np.zeros(N)
ens_var_up  = np.zeros(N)
ens_var_ugp = np.zeros(N)

# Initial conditions
u_g_p[0,:]    = [0.0] 
u_p[0,:]      = [0.0]
x_p[0,:]      = np.linspace(0, L_c, N_p)     #initial position in middle of cube L_c

for i in range(1, N):
    update_res(u_g_p[i-1], u_p[i-1], x_p[i-1], i, dt, tau_p, L_c, N_p)

# PLOT RESULTS _____________________________________________________________________________________________
#region ####### PLOTTING FUNCTION DEFINITIONS ####################
def plot_norm_dist(data, title):
    mu, std = scipy.stats.norm.fit(data)
    plt.hist(data, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)

    plt.title(f'{title}: Normal Fit ($\mu$={mu:.2f}, $\sigma$={std:.2f})')
    plt.xlabel(f'{title}')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def compare_norm_dist(data1, data2, title, data1_label, data2_label):
    
    mu1, std1 = scipy.stats.norm.fit(data1)
    mu2, std2 = scipy.stats.norm.fit(data2)

    x = np.linspace(-0.5, 0.5, 500)
    p1 = scipy.stats.norm.pdf(x, mu1, std1)
    p2 = scipy.stats.norm.pdf(x, mu2, std2)
    plt.plot(x, p1, 'r', linewidth=2, label=f'{data1_label} ($\mu$={mu1:.2f}, $\sigma$={std1:.2f})')
    plt.plot(x, p2, 'b', linewidth=2,label=f'{data2_label} ($\mu$={mu2:.2f}, $\sigma$={std2:.2f})')
    plt.title(f'{title}')
    plt.legend()
    plt.xlabel('Velocity')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def plot_histogram(data):
    plt.figure(2, figsize=(10, 6))
    plt.hist(data, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.grid(True)
    plt.show()

def plot_dW_t(t, dw):
    plt.figure(2, figsize=(10, 6))
    plt.plot(t, dW, label="dWx")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Wiener Increment')
    plt.grid(True)
    plt.title('Wiener Increment with Time (2D)')
    plt.show()

def plot_up_ugp_t(t, u_p, u_g_p):
    plt.figure(3, figsize=(10, 6))
    plt.plot(t, u_p, label="u'_p in X")
    plt.plot(t, u_g_p, label="u'_g@p in X")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.grid(True)
    plt.title('Particle and Gas Velocities with Time (1D)')
    plt.show()

def plot_xp_t(t, x_p):
    plt.figure(4, figsize=(10, 6))
    plt.plot(t, x_p, '.', markersize=1)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid(True)
    plt.title(f'Particle position with time (1D) - {N_p} Particles')
    plt.show()

def plot_ens_avg_with_time(ens_avg_dW, ens_avg_xp, t):
    plt.figure(5, figsize=(10, 6))
    plt.plot(t, ens_avg_dW, linewidth=1, label="Ensemble Avg dW")
    plt.plot(t, ens_avg_xp, label="Ensemble Avg Xp", linewidth=1)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Ensemble Average [m]')
    plt.grid(True)
    plt.title(f'Ensemble average evolution with time (1D) - {N_p} Particles')
    plt.show()

def plot_plots(t, x_p, ens_avg_dW, ens_avg_ugp, ens_avg_up, dW): # Plots all desired plots 
    #plot_norm_dist(u_p, 'Particle Velocity')
    #plot_norm_dist(u_g_p, 'Fluid Velocity')
    #compare_norm_dist(u_p, u_g_p, 'Comaprison of Normal Distribution Functions', 'Particl Vel', 'Fluid Vel')
    #plot_dW_t(t, dW)
    #plot_up_ugp_t(t, u_p, u_g_p)
    plot_xp_t(t, x_p)
    #plot_ens_avg_with_time(ens_avg_dW, dW, t)
    #plot_ens_avg_with_time(ens_avg_ugp, ens_avg_up, t)

#plot_plots(t,x_p,ens_avg_dW, ens_avg_ugp, ens_avg_up, dW)

def master_plot(N_p, t, x_p, ens_avg_ugp, ens_avg_up, 
                ens_var_ugp, ens_var_up, dW, ens_avg_dW):
    fig, axs = plt.subplots(2, 2, figsize=(10,8), sharex=True)
    fig.suptitle(f'Turbulent dispersion of {N_p} droplets')

    #Plot 1
    axs[0, 0].plot(t[::10], x_p[::10], '.', markersize=1)
    axs[0, 0].set_title("a) Time response of particles")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Position [m]")
        
    #Plot 2
    axs[0, 1].plot(t, ens_avg_ugp, 'b', label=r'Ens. Avg. $u_{g@p}$')
    axs[0, 1].plot(t, ens_avg_up, 'r', label=r'Ens. Avg. $u_{p}$')
    axs[0, 1].hlines(np.mean(ens_avg_ugp), 0, 10, 'b', linewidth=1, linestyle='--')
    axs[0, 1].hlines(np.mean(ens_avg_up), 0, 10, 'r', linewidth=1, linestyle='--')
    axs[0, 1].plot
    axs[0, 1].set_title("b) Ensemble average of velocities")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Velocity [m/s]")
    axs[0, 1].set_ylim([-0.03, 0.03])
    axs[0, 1].legend()

    #Plot 3
    axs[1, 0].plot(t, ens_var_ugp, 'b', label=r'Ens. Var. $u_{g@p}$')
    axs[1, 0].plot(t, ens_var_up, 'r', label=r'Ens. Var. $u_{p}$')
    axs[1, 0].hlines(np.mean(ens_var_ugp),0, 10, 'b', linewidth=1, linestyle='--')
    axs[1, 0].hlines(np.mean(ens_var_up), 0, 10,'r', linewidth=1, linestyle='--')
    axs[1, 0].set_title("c) Ensemble variance of velocities")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Variance Vel")
    axs[1, 0].set_ylim([0, 0.015])
    axs[1, 0].legend()

    #Plot 4
    axs[1, 1].plot(t, dW[:,0], '.', color='gray', markersize=2, alpha=0.7)
    axs[1, 1].plot(t, ens_avg_dW, '.', color='blue', markersize=2, alpha=0.7)
    axs[1, 1].plot(t, ens_var_dW, '.', color='red', markersize=2, alpha=0.7)
    axs[1, 1].hlines(np.mean(ens_avg_dW),0, 10, 'b', linewidth=1, linestyle='--')
    axs[1, 1].hlines(np.mean(ens_var_dW),0, 10, 'r', linewidth=1, linestyle='--')    
    axs[1, 1].set_title("d) Wiener function")
    axs[1, 1].legend(['dW[0]', 'Ens. Avg. dW', 'Ens. Var. dW', 'Mean Ens. Avg.', 'Mean Ens. Var.'])
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylim([-0.02, 0.02])


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave room for suptitle
    plt.show()
#endregion 

print(f'CPU Time of calculation: {time.time()-start_time} seconds')

master_plot(N_p, t, x_p, ens_avg_ugp, ens_avg_up, 
            ens_var_ugp, ens_var_up, dW, ens_avg_dW)


# SAVE DATA _______________________________________________________________________________________________
def save_data(t, T, x_p, u_p, u_g_p, T_L, k, tau_p):
    import os

    save_dir = "/d/mherok/Documents/Droplet Transport in HIT/1D/Multiple droplets/Postprocessing"
    file_path = os.path.join(save_dir, f"Results_T={T}s_Np={N_p}.csv")

    np.savez_compressed(file_path,
                        t=t,
                        x_p = x_p,
                        u_p = u_p,
                        u_g_p = u_g_p,
                        T_L = T_L,
                        k = k,
                        tau_p = tau_p)

#save_data(t, T, x_p, u_p, u_g_p, T_L, k, tau_p)

## COMMENTS - THINGS TO DO
# 1.    Remove dW list. Increment can be calculated directly in u_g_p function
# 2.    Currently, periodicity condition is enforced by looping over positions of 
#       all droplets at a given timestep and checking against L_c. Could be expensive for
#       very large numbers of droplets.