import numpy as np
from tqdm import tqdm

class Particle:
    def __init__(self, lb, ub, dim):
        self.lb = lb
        self.ub = ub
        self.position = np.random.uniform(low=lb, high=ub, size=dim)
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position
        self.best_score = -np.inf

class PSO:
    def __init__(self, n_particles, dimensions, c1, c2, w_max, w_min, lb, ub):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.lb = lb
        self.ub = ub
        self.particles = [Particle(self.lb, self.ub, self.dimensions) for _ in range(self.n_particles)]
        self.gbest_position = self.particles[0].position
        self.gbest_score = self.particles[0].best_score

    def run(self, iterations):
        for i in tqdm(range(iterations), desc='Progress'):
            w = self.w_max - i * ((self.w_max - self.w_min) / iterations)
            for particle in self.particles:
                particle.velocity = (w * particle.velocity) + \
                                    (self.c1 * np.random.rand() * (particle.best_position - particle.position)) + \
                                    (self.c2 * np.random.rand() * (self.gbest_position - particle.position))
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, self.lb, self.ub)
                score = self.objective_function(particle.position)
                if score > particle.best_score:
                    particle.best_position = particle.position.copy()
                    particle.best_score = score
                if score > self.gbest_score:
                    self.gbest_position = particle.position.copy()
                    self.gbest_score = score
        return self.gbest_position
