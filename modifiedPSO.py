import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

class Particle:
    def __init__(self, n_features, initialization_prob):
        self.position = np.random.choice([0, 1], size=n_features, p=[1 - initialization_prob, initialization_prob])
        self.velocity = np.random.uniform(-1, 1, size=n_features)  # Initialize velocity
        self.best_position = self.position  # Best position found by this particle
        self.best_score = float('-inf')  # Initialize best score to negative infinity

def PSO(features, labels, n_particles, n_iterations, w_max, w_min, c1, c2, initialization_prob):
    n_features = features.shape[1]
    particles = [Particle(n_features, initialization_prob) for _ in range(n_particles)]
    global_best_position = None
    global_best_score = float('-inf')

    for iteration in range(n_iterations):
        for particle in particles:
            mask = particle.position.astype(bool)

            # Ensure mask selects at least one feature
            if not np.any(mask):
                mask[np.argmax(particle.velocity)] = True

            reduced_features = features[:, mask]

            X_train, X_test, y_train, y_test = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)
            clf = svm.SVC()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            if score > particle.best_score:
                particle.best_position = particle.position
                particle.best_score = score

                if score > global_best_score:
                    global_best_position = particle.position
                    global_best_score = score

        w = w_max - iteration * (w_max - w_min) / n_iterations

        for particle in particles:
            particle.velocity = w * particle.velocity + c1 * np.random.uniform() * (particle.best_position - particle.position) + c2 * np.random.uniform() * (global_best_position - particle.position)
            particle.position = np.where(np.random.uniform(size=n_features) < 1/(1 + np.exp(-particle.velocity)), 1, 0)

            # Ensure new position selects at least one feature
            if not np.any(particle.position):
                particle.position[np.argmax(particle.velocity)] = 1

            # Ensure 90% reduction
            if np.sum(particle.position) > 0.1 * n_features:
                zero_indices = np.random.choice(np.where(particle.position == 1)[0], size=int(0.9 * np.sum(particle.position)), replace=False)
                particle.position[zero_indices] = 0

    return global_best_position, global_best_score
