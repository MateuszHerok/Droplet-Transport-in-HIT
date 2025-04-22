import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
print('-----------POSTPROCESSING-----------')
# This code will read results from 1D_droplet_transport.py and postprocess them
# It calculates:
#   1. Velocity variance of u_p and u_g_p
#   2. Autocorrelation function for u_g_p
#   3. Covariance of u_p and u_g_p
# All results are compared to Tchen-Hinze theory

#region ####### FUNCTIONS ####################""
def import_data(file_name):
    data = np.load(file_name)

    t = data['t'] 
    x_p = data['x_p'] 
    u_p = data['u_p'] 
    u_g_p = data['u_g_p'] 
    T_L = data['T_L']  
    k = data['k']  
    tau_p = data['tau_p']  
    print(f'\nImported data from {file_name}\n')
    return t, x_p, u_p, u_g_p, T_L, k, tau_p

def trim_transient_data(t, x_p, u_p, u_g_p, T_L, threshold): # Trims data to omit transient behaviour. Threshold means how many T_L durations to skip, ie 10
    dt = t[1]-t[0]
    index_settled = int(T_L / dt)
    index_settled = threshold * index_settled 
    t = t[index_settled:]
    x_p = x_p[index_settled:]
    u_p = u_p[index_settled:]
    u_g_p = u_g_p[index_settled:]
    print(f'\nTrimmed data to ignore {threshold}*T_L\n'
          f'Length of data: {len(t)}')
    return t, x_p, u_p, u_g_p, dt

def calc_variance(u_p, u_g_p): # Calculates the variance of u_p and u_g_p, gives error to Tchen-Hinze result
    var_u_p = np.var(u_p)
    var_u_g_p = np.var(u_g_p)
    return var_u_p, var_u_g_p

def percentage_error(measured, reference):
    return ((measured - reference) / reference) * 100

def plot_variance(var_u_p, var_u_g_p, tau_p, T_L, var_u_p_mean, var_u_g_p_mean): # Plots Tchen-Hinze result and obtained point
    x_TH = np.linspace(0.1, 1)
    y_TH = 1 / (1 + x_TH)

    y_sim = var_u_p / var_u_g_p
    x_sim = (tau_p / T_L) * np.ones_like(y_sim) 

    y_sim_mean = var_u_p_mean / var_u_g_p_mean
    x_sim_mean = np.mean(x_sim)
    err = percentage_error(y_sim_mean, (1 / (1 + tau_p / T_L)))
    print(f'Simulation mean for variance {y_sim_mean:.5f}')
    print(f'Tchen-Hinze result for variance {1/(1 + tau_p / T_L):.5f}')
    print(f'Error to Tchen-Hinze: {err:.5f}%')
    plt.figure(1, figsize=(10, 6))
    plt.plot(x_TH, y_TH, 'k', label='Tchen-Hinze', linewidth=1)
    plt.plot(x_sim, y_sim, '.r', label='Simulation', markersize=1)
    plt.plot(x_sim_mean, y_sim_mean, 'bo', label='Simulation Mean')
    plt.legend()
    plt.xlabel(r'$\tau_p$ / $T_L$')
    plt.ylabel(r'$var(u_p)$ / $var(u_{g@p})$')
    plt.grid(False)
    plt.title('Comparison to Tchen-Hinze result for velocity variance')
    plt.xlim([0, 1])
    plt.show()

def calc_q_(u_p, N_p):                              # Calculates the agitation or u²
    return 0.5*sum(u_p*u_p)/N_p

def calc_autocorrelation_function(u_g_p, q_g_p,     # Calculates fluid vel autocorrelation function 
                                  time_points, lag, N_p):     # lag in terms of dt
    
    R_g_p = np.zeros(len(time_points))
    
    for i in range(0,len(time_points)):
        t_idx = time_points[i]
        if t_idx + lag < u_g_p.shape[0]:
            denominator = 2*q_g_p[i]
            if denominator != 0 and not np.isnan(denominator):
                R_g_p[i] = np.mean(u_g_p[t_idx] * u_g_p[t_idx + lag]) / denominator
            else:
                R_g_p[i] = np.nan  # Skip division if variance is 0 or undefined
        else: 
            R_g_p[i] = np.nan

    return R_g_p

def calc_covariance(v1, v2, N_p):                   # Calculates velocity covariance between v1 qnd v2
    return sum(v1*v2)/N_p

#endregion 
      
t, x_p, u_p, u_g_p, T_L, k, tau_p = import_data("Results_T=10s_Np=1000.csv.npz")

t, x_p, u_p, u_g_p, dt = trim_transient_data(t, x_p, u_p, u_g_p, T_L, 10)

print(f'\nSimulation length = {t[-1]} s\n'
      f'Lagrangian timescale T_L = {T_L:.5f} s\n'
      f'Particle relaxation time tau_p = {tau_p:.5f} s\n')

N_p = len(u_p[0])
print(f'Number of droplets $N_p$: {N_p}')
time_points = np.round(len(t) / 1).astype(int) # Number of time locations to check ensemble results
time_points = np.round(np.linspace(0, len(t), time_points, endpoint=False)).astype(int)
var_u_p     = np.zeros(len(time_points))    # Variance of u_p
var_u_g_p   = np.zeros(len(time_points))    # Variance of u_g_p
q_p         = np.zeros(len(time_points))    # Particle agitation 
q_g_p       = np.zeros(len(time_points))    # Fluid agitation seen by particle
q_gp        = np.zeros(len(time_points))    # Particle and fluid velocity covariance
#   1. Velocity variance

for i in range(0,len(time_points)):
    j = time_points[i]
    var_u_p[i], var_u_g_p[i] = calc_variance(u_p[j], u_g_p[j])      # Calculates velocity variances
    q_p[i] = calc_q_(u_p[j], N_p)                                   # Calculates particle agitation
    q_g_p[i] = calc_q_(u_g_p[j], N_p)                               # Calculates fluid agitation seen by particle
    q_gp[i] = calc_covariance(u_p[j], u_g_p[j], N_p)                # Particle and fluid velocity covariance

var_u_p_mean = np.nanmean(var_u_p)
var_u_g_p_mean = np.nanmean(var_u_g_p)

plot_variance(var_u_p, var_u_g_p, tau_p, T_L, var_u_p_mean, var_u_g_p_mean)

#   2. Autocorrelation function decay
#   2.a First, autocorrelation function is computed for single value of tau
#       Particle agitation is also computed to check if system is in stationary regime

# Plot particle and fluid agitation to see stationary regime
plt.figure(2, figsize=(10, 6))
plt.plot(time_points*dt, q_p, 'r', label='Particle agitation', linewidth=1)
plt.plot(time_points*dt, q_g_p, 'b', label='Fluid agitation', linewidth=1)
plt.plot(time_points*dt, np.ones_like(time_points)*np.mean(q_p), '--r', label='Particle agitation', linewidth=1)
plt.plot(time_points*dt, np.ones_like(time_points)*np.mean(q_g_p), '--b', label='Fluid agitation', linewidth=1)
plt.legend()
plt.xlabel(r'Time [s]')
plt.ylabel(r'Agitation [m/s]²')
plt.grid(False)
plt.title('Particle and fluid agitation with time')
plt.xlim([0, 10])
plt.show()

# Calculate fluid velocity autocorrelation function for given lag 
R_g_p = calc_autocorrelation_function(u_g_p, q_g_p, time_points, 5, N_p)
print(f'Time average of autocorrelation function: {np.nanmean(R_g_p):.5f}')

# Plot autocorrelation function R_g_p with time
plt.figure(3, figsize=(10, 6))
plt.plot(time_points*dt, R_g_p, '.r', label='Fluid velocity autocorrelation', markersize=1)
plt.plot(time_points*dt, np.ones_like(time_points)*np.nanmean(R_g_p), '--k', label='Mean fluid velocity autocorrelation', linewidth=1)
plt.legend()
plt.xlabel(r'Time [s]')
plt.ylabel(r'Autocorrelation function [m/s]²')
plt.grid(False)
plt.title('Fluid velocity autocorrelation function or lag 5dt')
plt.xlim([0, 10])
plt.ylim([0, 1])
plt.show()

#   2.b Calculate exponential decay of autocorrelation function

lag = np.array([2, 5, 10, 50, 100, 500])
R_g_p_tau = np.zeros(len(lag))

for i in range(0, len(lag)):
    R_i = calc_autocorrelation_function(u_g_p, q_g_p, time_points, lag[i], N_p)
    R_g_p_tau[i] = np.nanmean(R_i)

# Plot exponential decay of autocorrelation function
x_sim = lag*dt/T_L
x_TH = np.linspace(0, 50, 1000)
y_TH = np.exp(-x_TH)
plt.figure(4, figsize=(10, 6))
plt.plot(x_sim, R_g_p_tau, 'xr', label='Mean Autocorrelation Function', markersize=10)
plt.plot(x_TH, y_TH, 'k', label='Tchen-Hinze', linewidth=1)
plt.legend()
plt.xlabel(r'Lag / Lagrangian Timescale [$\tau/T_L$] [s]')
plt.ylabel(r'Autocorrelation function [m/s]²')
plt.grid(False)
plt.title('Exponential decay of velocity autocorrelation function with lag')
plt.xlim([0, 10])
plt.show()

#   3. Covariance analysis
    # q_p = particle agitation
    # q_g_p = fluid agitation seen by particle
    # q_gp = particle fluid velocity covariance

x_TH = np.logspace(-2, 2, 1000)
y_TH = x_TH/(1+x_TH)

x_sim = T_L/ tau_p
y1_sim = q_gp / (2*q_g_p)
y2_sim = q_p / q_g_p

plt.figure(5, figsize=(10, 6))
plt.plot(np.ones_like(y1_sim)*x_sim, y1_sim, '.r', label=r'$q_{gp} / 2q_{g@p}$', markersize=1)
plt.plot(x_sim, np.nanmean(y1_sim), 'r', label=r'mean $q_{gp} / 2q_{g@p}$', markersize=10, marker='d')
plt.plot(np.ones_like(y2_sim)*x_sim, y2_sim, '.b', label=r'$q_{p} / q_{g@p}$', markersize=1)
plt.plot(x_sim, np.nanmean(y2_sim), 'b', label=r'mean $q_{gp} / 2q_{g@p}$', markersize=10, marker='x')
plt.plot(x_TH, y_TH, 'k', label='Tchen-Hinze', linewidth=1)
plt.xscale('log')
plt.legend()
plt.xlabel(r'Inverse Stokes [$T_L/ \tau_p$]')
#plt.ylabel(r'')
plt.grid(False)
plt.title('Fluid particle velocity covariance')
plt.xlim([0.1, 10])
plt.show()