# Particle Swarm Optimization (PSO) 

This project provides an implementation of the Particle Swarm Optimization (PSO) algorithm. The PSO algorithm is a population-based stochastic optimization technique inspired by the social behavior of bird flocking or fish schooling. It is used to find the global minimum of a problem, given an objective function.

# Getting Started

The project is implemented in Python and uses numpy for handling arrays and tqdm for showing progress. To run the code, you would need a Python environment with numpy and tqdm installed. To run the algorithm, simply import the PSO class from the script and create a new PSO instance with your objective function and the necessary parameters.

from pso import PSO

def objective_function(position):
    # Define your objective function here
    return score

pso = PSO(n_particles, dimensions, c1, c2, w_max, w_min, lb, ub, objective_function)
best_position = pso.run(iterations)

