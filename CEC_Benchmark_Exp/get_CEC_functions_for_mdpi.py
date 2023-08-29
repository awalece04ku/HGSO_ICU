# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:35:03 2023

@author: uqmawal
"""
import numpy as np
# Define the benchmark functions

def sphere_function(x):
    """Sphere Function"""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin Function"""
    n = len(x)
    A = 10
    return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x))

def ackley_function(x):
    """Ackley Function"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
# Define the Rosenbrock function
def rosenbrock_function(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def griewank_function(x):
    """Griewank Function"""
    sum_part = np.sum(x**2)
    prod_part = np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    return 1 + sum_part/4000 - prod_part

def beale_function(x):
    """Beale Function"""
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def drop_wave_function(x):
    """Drop Wave Function"""
    return -(1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))) / (0.5 * (x[0]**2 + x[1]**2) + 2)

def bohachevsky_function(x):
    """Bohachevsky Function"""
    return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7

def michalewicz_function(x, m=10):
    """Michalewicz Function"""
    return -np.sum(np.sin(x) * (np.sin((np.arange(len(x)) + 1) * x**2 / np.pi))**(2*m))

def levy_function(x):
    """Levy Function"""
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[0])**2 + np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2)) + (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
def schwefel_function(x):
    """Schwefel Function"""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def schaffer_function_4(x):
    """Schaffer Function N. 4"""
    return 0.5 + (np.cos(np.sin(np.abs(x[0]**2 - x[1]**2)))**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2

def brown_function(x):
    """Brown Function"""
    return np.sum(x[:-1]**(x[1:] + 1) + x[1:]**(x[:-1] + 1))

def bartels_conn_function(x):
    """Bartels Conn Function"""
    return np.abs(x[0]**2 + x[1]**2 + x[0]*x[1]) + np.abs(np.sin(x[0])) + np.abs(np.cos(x[1]))


def get_CEC_functions(function_name, dim_size):
    """
    Returns details of the selected benchmark function based on the provided function name.
    """
    # Define the lower and upper bounds and the objective function for each function name
    if function_name == "F1":  # Sphere function
        lb = -100
        ub = 100
        fobj = sphere_function
        
      
    elif function_name == "F2":  # Ackley function
        lb = -32
        ub = 32
        fobj = ackley_function
        
   
    elif function_name == "F3":  # levy_function function
        lb = -10
        ub = 10
        fobj = levy_function      
    
    elif function_name == "F4":  # schaffer_function_4 function
        lb = -100
        ub = 100
        fobj = schaffer_function_4      
    
    else:
        lb = None
        ub = None
        fobj = None
        
    return lb, ub, dim_size, fobj

# Test the function
#get_CEC_functions("F1", 30)
# List of functions to test
