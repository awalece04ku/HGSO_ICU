# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:29:50 2023

@author: uqmawal
"""

import numpy as np
from Metropolish_HGSO import HGSO_Metropolish_diverse_stable
from HGS import HGS      # regular HGS

import pandas as pd
from get_CEC_functions_for_mdpi import get_CEC_functions
import matplotlib.pyplot as plt

import os




def run_and_plot_updated_v2(function_name, N, FEs, dim_size):
    # Get the function details
    lb, ub, dim, fobj = get_CEC_functions(function_name, dim_size)
    
    # Run the regular HGS algorithm
    fitness_HGS, best_positions_HGS, convergence_HGS = HGS(N, FEs, lb, ub, dim, fobj)
    
    # Run the HGSO_Metropolish_diverse_stable algorithm
    fitness_HGSO_MH, best_positions_HGSO_MH, convergence_HGSO_MH = HGSO_Metropolish_diverse_stable(N, FEs, lb, ub, dim, fobj)
    
    # Plot the convergence curves for both algorithms on the same plot
    plt.figure(figsize=(6, 5))
    plt.semilogy(convergence_HGS, label="Original HGSO", color="blue")
    plt.semilogy(convergence_HGSO_MH, label="Proposed HGSO", color="red")
    
    # Adjust y-axis limits to ensure all data points are visible
    min_val = min(np.min(convergence_HGS), np.min(convergence_HGSO_MH))
    max_val = max(np.max(convergence_HGS), np.max(convergence_HGSO_MH))
    plt.ylim(min_val, max_val)
    
    #plt.title(f"Convergence Curve for {function_name}")
    plt.xlabel("Generations")
    plt.ylabel("Best fitness obtained so far")
    plt.legend()
    plt.grid(True)
    # Adjusting the layout
    plt.tight_layout()
    # Saving the figure in high resolution .eps format
    plt.savefig(f"Convergence Curve for {function_name}.eps", format='eps', dpi=600)
    plt.savefig(f"Convergence Curve for {function_name}.png", format='png', dpi=600)

    plt.show()
    
    return {
        "HGS": {
            "Best Location": best_positions_HGS,
            "Best Fitness": fitness_HGS
        },
        "HGSO_Metropolish_diverse_stable": {
            "Best Location": best_positions_HGSO_MH,
            "Best Fitness": fitness_HGSO_MH
        }
    }



def run_and_save_to_csv(function_name, N, FEs, dim_size):
    # Get the function details
    lb, ub, dim, fobj = get_CEC_functions(function_name, dim_size)
    
    # Run the regular HGS algorithm
    fitness_HGS, _, convergence_HGS = HGS(N, FEs, lb, ub, dim, fobj)
    
    # Run the HGSO_Metropolish_diverse_stable algorithm
    fitness_HGSO_MH, _, convergence_HGSO_MH = HGSO_Metropolish_diverse_stable(N, FEs, lb, ub, dim, fobj)
    
    # Save to CSV files
    # For fitness values, we'll append to existing data
    if not os.path.exists('fitness_HGS.csv'):
        pd.DataFrame(columns=[function_name]).to_csv('fitness_HGS.csv', index=False)
    if not os.path.exists('fitness_HGSO_MH.csv'):
        pd.DataFrame(columns=[function_name]).to_csv('fitness_HGSO_MH.csv', index=False)
    
    df_fitness_HGS = pd.read_csv('fitness_HGS.csv')
    df_fitness_HGS[function_name] = [fitness_HGS]
    df_fitness_HGS.to_csv('fitness_HGS.csv', index=False)
    
    df_fitness_HGSO_MH = pd.read_csv('fitness_HGSO_MH.csv')
    df_fitness_HGSO_MH[function_name] = [fitness_HGSO_MH]
    df_fitness_HGSO_MH.to_csv('fitness_HGSO_MH.csv', index=False)
    
    # For convergence data, since it's a series of values, we'll create new files for each function
    pd.DataFrame({function_name: convergence_HGS}).to_csv(f'convergence_HGS_{function_name}.csv', index=False)
    pd.DataFrame({function_name: convergence_HGSO_MH}).to_csv(f'convergence_HGSO_MH_{function_name}.csv', index=False)
    
    return {
        "HGS": {
            "Best Fitness": fitness_HGS
        },
        "HGSO_Metropolish_diverse_stable": {
            "Best Fitness": fitness_HGSO_MH
        }
    }



# List of functions to test
function_names = ["F1", "F2","F3","F4"]  # Can be extended for F6-F13 as needed

# Store the results for each function
results_all_functions = {}
Max_iter=300
N=30
dim=30
# Run the script for each function
for fname in function_names:
    results_all_functions[fname] = run_and_plot_updated_v2(fname, N, Max_iter, dim)
    #run_and_save_to_csv(fname, N, Max_iter, dim)

results_all_functions