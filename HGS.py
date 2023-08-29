
import numpy as np
def random_initialization(N, dim, ub, lb):
    """Randomly initialize a set of solutions."""
    return np.random.rand(N, dim) * (ub - lb) + lb

def HGS(N, Max_iter, lb, ub, dim, fobj):
    bestPositions = np.zeros(dim)
    tempPosition = np.zeros((N, dim))
    Destination_fitness = float('inf')
    Worstest_fitness = float('-inf')
    AllFitness = np.full(N, float('inf'))
    VC1 = np.ones(N)
    weight3 = np.ones((N, dim))
    weight4 = np.ones((N, dim))
    X = random_initialization(N, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)
    it = 0  # Starting from 0 for Pythonic indexing
    hungry = np.zeros(N)  # Initializing the hungry variable
    
    while it < Max_iter:
        VC2 = 0.03
        sumHungry = 0
        
        # Adjust positions outside bounds and compute fitness
        X = np.clip(X, lb, ub)
        for i in range(N):
            AllFitness[i] = fobj(X[i])
        
        # Sort fitness values
        IndexSorted = np.argsort(AllFitness)
        AllFitnessSorted = AllFitness[IndexSorted]
        bestFitness = AllFitnessSorted[0]
        worstFitness = AllFitnessSorted[-1]
        
        # Update best fitness and position
        if bestFitness < Destination_fitness:
            bestPositions = X[IndexSorted[0]]
            Destination_fitness = bestFitness
            count = 0
        
        if worstFitness > Worstest_fitness:
            Worstest_fitness = worstFitness
        
        # Calculate the hunger for each position
        for i in range(N):
            VC1[i] = 1 / np.cosh(abs(AllFitness[i] - Destination_fitness))
            if Destination_fitness != AllFitness[i]:
                temprand = np.random.rand()
                c = (AllFitness[i]-Destination_fitness)/(Worstest_fitness-Destination_fitness)*temprand*2*(ub-lb)
                b = 100*(1+temprand) if c < 100 else c
                hungry[i] += b
                sumHungry += hungry[i]
        
        # Calculate the hungry weight for each position
        for i in range(N):
            for j in range(1, dim):
                weight3[i, j] = (1-np.exp(-abs(hungry[i]-sumHungry)))*np.random.rand()*2
                weight4[i, j] = hungry[i]*N/sumHungry*np.random.rand() if np.random.rand() < VC2 else 1
        
        # Update the position of search agents
        shrink = 2 * (1 - it/Max_iter)
        for i in range(N):
            if np.random.rand() < VC2:
                X[i] *= (1 + np.random.randn())
            elif count > 0:  # Ensure that count is greater than 0 before using tempPosition
                A = np.random.randint(0, count)
                for j in range(dim):
                    r = np.random.rand()
                    vb = 2 * shrink * r - shrink
                    if r > VC1[i]:
                        X[i, j] = weight4[i, j] * tempPosition[A, j] + vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    else:
                        X[i, j] = weight4[i, j] * tempPosition[A, j] - vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
        
        Convergence_curve[it] = Destination_fitness
        it += 1
    
    return Destination_fitness, bestPositions, Convergence_curve
