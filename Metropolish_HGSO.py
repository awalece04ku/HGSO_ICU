


import numpy as np
#import matplotlib as plt

def initialize_population_MH(n, d, lb, ub, fitness):
    """
    Initialize the population using Metropolis-Hastings method.
    
    Parameters:
    - n: number of individuals in the population
    - d: number of dimensions (variables) of each individual
    - lb: lower bound of each dimension
    - ub: upper bound of each dimension
    - fitness: objective function to evaluate the fitness
    
    Returns:
    - population: Initialized population
    """
    population = np.zeros((n, d))
    population[0, :] = lb + (ub - lb) * np.random.rand(d)

    for i in range(1, n):
        candidate = lb + (ub - lb) * np.random.rand(d)
        acceptance_prob = min(1, fitness(candidate) / fitness(population[i-1, :]))

        if np.random.rand() <= acceptance_prob:
            population[i, :] = candidate
        else:
            population[i, :] = population[i-1, :]

    return population



def HGSO_Metropolish_diverse_stable(N, Max_iter, lb, ub, dim, fobj):
    bestPositions = np.zeros(dim)
    tempPosition = np.zeros((N, dim))
    Destination_fitness = np.inf
    Worstest_fitness = -np.inf
    AllFitness = np.inf * np.ones(N)
    VC1 = np.ones(N)
    weight3 = np.ones((N, dim))
    weight4 = np.ones((N, dim))
    
    # Initialize using Metropolis-Hastings
    X = initialize_population_MH(N, dim, ub, lb, fobj)
    Convergence_curve = np.zeros(Max_iter)
    it = 1
    hungry = np.zeros(N)
    count = 0

    while it <= Max_iter:
        # Adaptive VC2 parameter
        VC2 = 0.03 + (0.3 - 0.03) * (1 - it/Max_iter)
        
        sumHungry = 0
        
        # Calculate and sort the fitness values
        for i in range(N):
            X[i, :] = np.clip(X[i, :], lb, ub)
            AllFitness[i] = fobj(X[i, :])
        
        sorted_indices = np.argsort(AllFitness)
        bestFitness = AllFitness[sorted_indices[0]]
        worstFitness = AllFitness[sorted_indices[-1]]

        # Update the best and worst fitness values
        if bestFitness < Destination_fitness:
            bestPositions = X[sorted_indices[0], :]
            Destination_fitness = bestFitness
            count = 0
        
        if worstFitness > Worstest_fitness:
            Worstest_fitness = worstFitness
        
        # Update positions and related values
        for i in range(N):
            VC1[i] = 1 / np.cosh(abs(AllFitness[i] - Destination_fitness))
            
            if Destination_fitness == AllFitness[i]:
                hungry[i] = 0
                count += 1
            else:
                hungry_value = np.exp(-(AllFitness[i] - Destination_fitness) / (AllFitness[i] - Worstest_fitness + 1e-5))
                hungry[i] = np.clip(hungry_value, -50, 50)  # Constrain the hungry values to prevent overflow
                sumHungry += hungry[i]
        
        tempPosition[0:count, :] = bestPositions
        
        for i in range(N):
            for j in range(dim):
                if hungry[i] > 0:
                    weight3[i, j] = hungry[i] * N / sumHungry * np.random.rand()
                    weight4[i, j] = hungry[i] * N / sumHungry * np.random.rand()
                else:
                    weight4[i, j] = 1
        
        # Update the Position of search agents
        shrink = 2 * (1 - it / Max_iter)
        for i in range(N):
            if np.random.rand() < VC2:
                X[i, :] = X[i, :] * (1 + np.random.randn())
            else:
                A = np.random.randint(0, max(min(count, N-1), 1))
                for j in range(dim):
                    r = np.random.rand()
                    vb = 2 * shrink * r - shrink
                    if r > VC1[i]:
                        X[i, j] = weight4[i, j] * tempPosition[A, j] + vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    else:
                        X[i, j] = weight4[i, j] * tempPosition[A, j] - vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
        
        # Reintroduce diversity by reinitializing a portion of the population using Metropolis-Hastings
        if it % 100 == 0:
            portion_to_reinitialize = int(0.2 * N)  # reinitialize 20% of the population
            X[:portion_to_reinitialize, :] = initialize_population_MH(portion_to_reinitialize, dim, ub, lb, fobj)
        
        # Check for NaN and reinitialize if any
        nan_indices = np.any(np.isnan(X), axis=1)
        num_nan_indices = nan_indices.sum()
        if num_nan_indices > 0:
            X[nan_indices, :] = initialize_population_MH(num_nan_indices, dim, ub, lb, fobj)
        
        Convergence_curve[it-1] = Destination_fitness
        it += 1

    return Destination_fitness, bestPositions, Convergence_curve