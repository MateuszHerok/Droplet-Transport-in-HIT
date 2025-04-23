# Droplet-Transport-in-HIT
Repository contains models and files for simulating turbulent dispersion of particles using a Langevin type model in HIT.

# 1D
Description of files:
1. 1D_droplet_transport_multiple.py - Calculates particle dispersion based on parameters defined in data.json. Plots time response and saves raw data.
2. data.json - Contains droplet, gas, and turbulent parameters.
3. 1D_postprocessing_multiple.py - Postprocessing script that takes a results file and compares against expected results (Tchen-Hinze).
4. Results_T=10s_Np=100.csv.npz - Sample results file for 100 droplets.
