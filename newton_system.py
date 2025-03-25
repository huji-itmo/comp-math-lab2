import numpy as np


def newton_system():
    # Initial guess
    x = 3.0
    y = 0.9
    tolerance = 1e-6
    max_iterations = 100

    for i in range(max_iterations):
        # Compute function values
        f1 = np.cos(x - 1) + y - 0.5
        f2 = x - np.cos(y) - 3
        F = np.array([f1, f2])

        # Compute Jacobian matrix
        df1dx = -np.sin(x - 1)
        df1dy = 1.0
        df2dx = 1.0
        df2dy = np.sin(y)
        J = np.array([[df1dx, df1dy], [df2dx, df2dy]])

        try:
            # Solve for the delta update
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("Singular Jacobian encountered. No solution found.")
            return None, None

        # Update x and y
        x += delta[0]
        y += delta[1]

        # Check for convergence
        if np.linalg.norm(delta) < tolerance:
            print(f"Converged in {i+1} iterations.")
            return x, y

    print("Did not converge within the maximum number of iterations.")
    return x, y


# Solve the system
solution_x, solution_y = newton_system()
print(f"Solution: x = {solution_x:.6f}, y = {solution_y:.6f}")
