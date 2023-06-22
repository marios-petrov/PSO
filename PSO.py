import numpy as np
from tqdm import tqdm

class Particle:
    def __init__(self, dimensions):
        self.position = np.random.choice([0, 1], size=(dimensions,), p=[0.5, 0.5])
        self.prob_position = self.position.astype(float)  # This is the "probabilistic" position
        self.velocity = np.random.rand(dimensions)
        self.pbest_position = self.position
        self.pbest_prob_position = self.prob_position
        self.pbest_value = float('inf')


class PSO:
    def __init__(self, n_particles, dimensions, c1, c2, w_max, w_min, iterations, objective_function):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.iterations = iterations
        self.objective_function = objective_function
        self.particles = [Particle(dimensions) for _ in range(n_particles)]
        self.global_best_value = float('inf')
        self.global_best_position = np.zeros((dimensions,))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_velocity(self, particle, iteration):
        w = self.w_max - iteration * ((self.w_max - self.w_min) / self.iterations)
        r1 = np.random.uniform(size=self.dimensions)
        r2 = np.random.uniform(size=self.dimensions)
        cognitive_velocity = self.c1 * r1 * (particle.pbest_prob_position - particle.prob_position)
        social_velocity = self.c2 * r2 * (self.global_best_position - particle.prob_position)
        particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity

    def update_position(self, particle):
        particle.prob_position = 1 / (1 + np.exp(-particle.velocity))
        particle.position = particle.prob_position > np.random.uniform(size=self.dimensions)

    def run(self, iterations):
        for iteration in tqdm(range(iterations)):
            for particle in self.particles:
                score = self.objective_function(particle.position)
                if score < particle.pbest_value:
                    particle.pbest_value = score
                    particle.pbest_position = particle.position
                    particle.pbest_prob_position = particle.prob_position
                if score < self.global_best_value:
                    self.global_best_value = score
                    self.global_best_position = particle.prob_position

            for particle in self.particles:
                self.update_velocity(particle, iteration)
                self.update_position(particle)

        return self.global_best_position

