import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def PSO(X, y, n_particles, n_iterations, w_max, w_min, c1, c2):
    dim = X.shape[1]

    # Initialize particles
    particles = np.random.uniform(low=-1, high=1, size=(n_particles, dim))
    velocities = np.random.uniform(low=-1, high=1, size=(n_particles, dim))

    pbest_positions = particles.copy()
    pbest_scores = np.full(shape=(n_particles,), fill_value=float('inf'))
    gbest_position = None
    gbest_score = float('inf')

    for iteration in tqdm(range(n_iterations), desc='PSO progress'):
        w = w_max - (w_max - w_min) * iteration / n_iterations

        # Binary mask
        masks = sigmoid(particles) > 0.5

        for i, mask in enumerate(masks):
            # Skip if all features are masked out
            if np.sum(mask) == 0:
                continue

            X_train, X_test, y_train, y_test = train_test_split(X[:, mask], y, test_size=0.2, random_state=42)

            # Train SVM
            clf = svm.SVC()
            clf.fit(X_train, y_train)

            # Evaluate accuracy
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Update personal best
            if accuracy < pbest_scores[i]:
                pbest_positions[i] = particles[i]
                pbest_scores[i] = accuracy

            # Update global best
            if accuracy < gbest_score:
                gbest_position = particles[i]
                gbest_score = accuracy

        # Update velocity and position
        for i in range(n_particles):
            velocities[i] = w * velocities[i] + c1 * np.random.uniform() * (
                        pbest_positions[i] - particles[i]) + c2 * np.random.uniform() * (gbest_position - particles[i])
            particles[i] += velocities[i]

    return gbest_position, gbest_score
