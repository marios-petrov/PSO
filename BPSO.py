# Author: Marios Petrov

# Necessary libraries
import numpy as np

# Define a Particle class for use in Particle Swarm Optimization (PSO)
class Particle:
    def __init__(self, n_features):
        # Initialize particle's position randomly with binary values
        self.position = np.random.randint(2, size=n_features)
        # Initialize particle's velocity with values in the range [-1, 1]
        self.velocity = np.random.uniform(-1, 1, size=n_features)
        # Set the particle's current best known position to its initial position
        self.best_position = self.position
        # Initialize the particle's best score to negative infinity
        self.best_score = float('-inf')

# Define the BPSO class to encapsulate the optimization algorithm
class BPSO:
    def __init__(self, objective_func, n_particles, n_iterations, w_max, w_min, c1, c2):
        # Objective function to be minimized or maximized
        self.objective_func = objective_func
        # Number of particles in the swarm
        self.n_particles = n_particles
        # Number of iterations to perform
        self.n_iterations = n_iterations
        # Maximum and minimum inertia weight
        self.w_max = w_max
        self.w_min = w_min
        # Cognitive and social coefficients
        self.c1 = c1
        self.c2 = c2

    # Method to execute the optimization process
    def optimize(self, features, labels):
        # Determine the number of features
        n_features = features.shape[1]
        # Initialize particles
        particles = [Particle(n_features) for _ in range(self.n_particles)]
        # Initialize global best position and score
        global_best_position = None
        global_best_score = float('-inf')

        # Main optimization loop
        for iteration in range(self.n_iterations):
            for particle in particles:
                # Convert particle's position to boolean mask for feature selection
                mask = particle.position.astype(bool)

                # Ensure at least one feature is selected
                if not np.any(mask):
                    mask[np.argmax(particle.velocity)] = True

                # Apply mask to select features
                reduced_features = features[:, mask]

                # Evaluate current position using the objective function
                score = self.objective_func(reduced_features, labels)

                # Update particle's best position and score if current score is better
                if score > particle.best_score:
                    particle.best_position = particle.position
                    particle.best_score = score

                    # Update global best position and score if necessary
                    if score > global_best_score:
                        global_best_position = particle.position
                        global_best_score = score

            # Update inertia weight linearly from max to min
            w = self.w_max - iteration * (self.w_max - self.w_min) / self.n_iterations

            # Update velocity and position for each particle
            for particle in particles:
                # Update velocity
                particle.velocity = w * particle.velocity + self.c1 * np.random.uniform() * (particle.best_position - particle.position) + self.c2 * np.random.uniform() * (global_best_position - particle.position)
                # Update position with a logistic function threshold
                particle.position = np.where(np.random.uniform(size=n_features) < 1/(1 + np.exp(-particle.velocity)), 1, 0)

                # Ensure new position selects at least one feature
                if not np.any(particle.position):
                    particle.position[np.argmax(particle.velocity)] = 1

        # Return the global best position and score after all iterations
        return global_best_position, global_best_score
