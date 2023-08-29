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

import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")


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


# Parameters for the runs
N_runs = 100

Max_iter=300
N=30
dim=30


function_name = "F1"
# Containers to collect data over the runs
all_fitness_HGS = []
all_convergence_HGS = []
all_fitness_HGSO_MH = []
all_convergence_HGSO_MH = []

# Execute the function N_runs times and collect the results
for run in range(N_runs):
    results = run_and_save_to_csv(function_name, N, Max_iter, dim)
    all_fitness_HGS.append(results["HGS"]["Best Fitness"])
    all_fitness_HGSO_MH.append(results["HGSO_Metropolish_diverse_stable"]["Best Fitness"])
    
    # Read the convergence data we just saved and append to the container lists
    df_convergence_HGS = pd.read_csv(f'convergence_HGS_{function_name}.csv')
    all_convergence_HGS.append(df_convergence_HGS[function_name].tolist())
    
    df_convergence_HGSO_MH = pd.read_csv(f'convergence_HGSO_MH_{function_name}.csv')
    all_convergence_HGSO_MH.append(df_convergence_HGSO_MH[function_name].tolist())


# Convert fitness results into DataFrames and save
pd.DataFrame({'Fitness_HGS': all_fitness_HGS}).to_csv(f'all_fitness_HGS_{function_name}.csv', index=False)
pd.DataFrame({'Fitness_HGSO_MH': all_fitness_HGSO_MH}).to_csv(f'all_fitness_HGSO_MH_{function_name}.csv', index=False)

# Convert convergence results into DataFrames and save
pd.DataFrame(data=list(zip(*all_convergence_HGS)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGS_{function_name}.csv', index=False)
pd.DataFrame(data=list(zip(*all_convergence_HGSO_MH)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGSO_MH_{function_name}.csv', index=False)




# Convert the lists into a DataFrame
df = pd.DataFrame({
    'Algorithms': ['Original HGSO'] * len(all_fitness_HGS) + ['Proposed HGSO'] * len(all_fitness_HGSO_MH),
    'Fitness': all_fitness_HGS + all_fitness_HGSO_MH
})





# Generate the boxplot without outliers
plt.figure(figsize=(6, 6))
sns.boxplot(x="Algorithms", y="Fitness", data=df, palette="muted", showfliers=False)
#plt.title(f"Function {function_name}")

plt.savefig(f"Boxplot for {function_name}.eps", format='eps', dpi=600)
plt.savefig(f"Boxplot for {function_name}.png", format='png', dpi=600)
plt.show()
# Mann-Whitney U test )
stat_f1, p_value_f1 = mannwhitneyu(all_fitness_HGS, all_fitness_HGSO_MH)

stat_f1, p_value_f1



## For Function F2
function_name = "F2"
# Containers to collect data over the runs
all_fitness_HGS = []
all_convergence_HGS = []
all_fitness_HGSO_MH = []
all_convergence_HGSO_MH = []

# Execute the function N_runs times and collect the results
for run in range(N_runs):
    results = run_and_save_to_csv(function_name, N, Max_iter, dim)
    all_fitness_HGS.append(results["HGS"]["Best Fitness"])
    all_fitness_HGSO_MH.append(results["HGSO_Metropolish_diverse_stable"]["Best Fitness"])
    
    # Read the convergence data we just saved and append to the container lists
    df_convergence_HGS = pd.read_csv(f'convergence_HGS_{function_name}.csv')
    all_convergence_HGS.append(df_convergence_HGS[function_name].tolist())
    
    df_convergence_HGSO_MH = pd.read_csv(f'convergence_HGSO_MH_{function_name}.csv')
    all_convergence_HGSO_MH.append(df_convergence_HGSO_MH[function_name].tolist())


# Convert fitness results into DataFrames and save
pd.DataFrame({'Fitness_HGS': all_fitness_HGS}).to_csv(f'all_fitness_HGS_{function_name}.csv', index=False)
pd.DataFrame({'Fitness_HGSO_MH': all_fitness_HGSO_MH}).to_csv(f'all_fitness_HGSO_MH_{function_name}.csv', index=False)

# Convert convergence results into DataFrames and save
pd.DataFrame(data=list(zip(*all_convergence_HGS)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGS_{function_name}.csv', index=False)
pd.DataFrame(data=list(zip(*all_convergence_HGSO_MH)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGSO_MH_{function_name}.csv', index=False)




# Convert the lists into a DataFrame
df = pd.DataFrame({
    'Algorithms': ['Original HGSO'] * len(all_fitness_HGS) + ['Proposed HGSO'] * len(all_fitness_HGSO_MH),
    'Fitness': all_fitness_HGS + all_fitness_HGSO_MH
})





# Generate the boxplot without outliers
plt.figure(figsize=(6, 6))
sns.boxplot(x="Algorithms", y="Fitness", data=df, palette="muted", showfliers=False)
#plt.title(f"Function {function_name}")

plt.savefig(f"Boxplot for {function_name}.eps", format='eps', dpi=600)
plt.savefig(f"Boxplot for {function_name}.png", format='png', dpi=600)
plt.show()
# Mann-Whitney U test )
stat_f2, p_value_f2 = mannwhitneyu(all_fitness_HGS, all_fitness_HGSO_MH)

stat_f2, p_value_f2


## For Function F3
function_name = "F3"
# Containers to collect data over the runs
all_fitness_HGS = []
all_convergence_HGS = []
all_fitness_HGSO_MH = []
all_convergence_HGSO_MH = []

# Execute the function N_runs times and collect the results
for run in range(N_runs):
    results = run_and_save_to_csv(function_name, N, Max_iter, dim)
    all_fitness_HGS.append(results["HGS"]["Best Fitness"])
    all_fitness_HGSO_MH.append(results["HGSO_Metropolish_diverse_stable"]["Best Fitness"])
    
    # Read the convergence data we just saved and append to the container lists
    df_convergence_HGS = pd.read_csv(f'convergence_HGS_{function_name}.csv')
    all_convergence_HGS.append(df_convergence_HGS[function_name].tolist())
    
    df_convergence_HGSO_MH = pd.read_csv(f'convergence_HGSO_MH_{function_name}.csv')
    all_convergence_HGSO_MH.append(df_convergence_HGSO_MH[function_name].tolist())


# Convert fitness results into DataFrames and save
pd.DataFrame({'Fitness_HGS': all_fitness_HGS}).to_csv(f'all_fitness_HGS_{function_name}.csv', index=False)
pd.DataFrame({'Fitness_HGSO_MH': all_fitness_HGSO_MH}).to_csv(f'all_fitness_HGSO_MH_{function_name}.csv', index=False)

# Convert convergence results into DataFrames and save
pd.DataFrame(data=list(zip(*all_convergence_HGS)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGS_{function_name}.csv', index=False)
pd.DataFrame(data=list(zip(*all_convergence_HGSO_MH)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGSO_MH_{function_name}.csv', index=False)




# Convert the lists into a DataFrame
df = pd.DataFrame({
    'Algorithms': ['Original HGSO'] * len(all_fitness_HGS) + ['Proposed HGSO'] * len(all_fitness_HGSO_MH),
    'Fitness': all_fitness_HGS + all_fitness_HGSO_MH
})





# Generate the boxplot without outliers
plt.figure(figsize=(6, 6))
sns.boxplot(x="Algorithms", y="Fitness", data=df, palette="muted", showfliers=False)
#plt.title(f"Function {function_name}")

plt.savefig(f"Boxplot for {function_name}.eps", format='eps', dpi=600)
plt.savefig(f"Boxplot for {function_name}.png", format='png', dpi=600)
plt.show()
# Mann-Whitney U test )
stat_f3, p_value_f3 = mannwhitneyu(all_fitness_HGS, all_fitness_HGSO_MH)

stat_f3, p_value_f3


## ## For Function F4
function_name = "F4"
# Containers to collect data over the runs
all_fitness_HGS = []
all_convergence_HGS = []
all_fitness_HGSO_MH = []
all_convergence_HGSO_MH = []

# Execute the function N_runs times and collect the results
for run in range(N_runs):
    results = run_and_save_to_csv(function_name, N, Max_iter, dim)
    all_fitness_HGS.append(results["HGS"]["Best Fitness"])
    all_fitness_HGSO_MH.append(results["HGSO_Metropolish_diverse_stable"]["Best Fitness"])
    
    # Read the convergence data we just saved and append to the container lists
    df_convergence_HGS = pd.read_csv(f'convergence_HGS_{function_name}.csv')
    all_convergence_HGS.append(df_convergence_HGS[function_name].tolist())
    
    df_convergence_HGSO_MH = pd.read_csv(f'convergence_HGSO_MH_{function_name}.csv')
    all_convergence_HGSO_MH.append(df_convergence_HGSO_MH[function_name].tolist())


# Convert fitness results into DataFrames and save
pd.DataFrame({'Fitness_HGS': all_fitness_HGS}).to_csv(f'all_fitness_HGS_{function_name}.csv', index=False)
pd.DataFrame({'Fitness_HGSO_MH': all_fitness_HGSO_MH}).to_csv(f'all_fitness_HGSO_MH_{function_name}.csv', index=False)

# Convert convergence results into DataFrames and save
pd.DataFrame(data=list(zip(*all_convergence_HGS)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGS_{function_name}.csv', index=False)
pd.DataFrame(data=list(zip(*all_convergence_HGSO_MH)), columns=[f"Run_{i+1}" for i in range(N_runs)]).to_csv(f'all_convergence_HGSO_MH_{function_name}.csv', index=False)




# Convert the lists into a DataFrame
df = pd.DataFrame({
    'Algorithms': ['Original HGSO'] * len(all_fitness_HGS) + ['Proposed HGSO'] * len(all_fitness_HGSO_MH),
    'Fitness': all_fitness_HGS + all_fitness_HGSO_MH
})





# Generate the boxplot without outliers
plt.figure(figsize=(6, 6))
sns.boxplot(x="Algorithms", y="Fitness", data=df, palette="muted", showfliers=False)
#plt.title(f"Function {function_name}")

plt.savefig(f"Boxplot for {function_name}.eps", format='eps', dpi=600)
plt.savefig(f"Boxplot for {function_name}.png", format='png', dpi=600)
plt.show()
# Mann-Whitney U test )
stat_f4, p_value_f4 = mannwhitneyu(all_fitness_HGS, all_fitness_HGSO_MH)

stat_f4, p_value_f4
