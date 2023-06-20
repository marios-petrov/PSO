import numpy as np
from tqdm import tqdm

class Particle:
    def __init__(self, lb, ub):
        self.position = np.random.randint(low=lb, high=ub)
        self.velocity = np.random.rand()
        self.best_position = self.position
        self.best_score = float('inf')

class PSO:
    def __init__(self, n_particles, dimensions, c1, c2, w_max, w_min, lb, ub, objective_function):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.lb = lb
        self.ub = ub
        self.particles = [Particle(lb, ub) for _ in range(n_particles)]
        self.global_best_position = self.particles[0].position
        self.global_best_score = float('inf')
        self.objective_function = objective_function

    def run(self, iterations):
        for iteration in tqdm(range(iterations)):
            for particle in self.particles:
                score = self.objective_function(particle.position)
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position

            w = self.w_max - iteration * ((self.w_max - self.w_min) / iterations)
            for particle in self.particles:
                new_velocity = (w * particle.velocity) + (self.c1 * np.random.rand() * (particle.best_position - particle.position)) + (self.c2 * np.random.rand() * (self.global_best_position - particle.position))
                new_position = particle.position + new_velocity

                if new_position < self.lb:
                    new_position = self.lb
                elif new_position > self.ub:
                    new_position = self.ub

                particle.position = new_position
                particle.velocity = new_velocity
        return self.global_best_position
