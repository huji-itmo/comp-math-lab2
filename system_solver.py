import numpy as np


class SystemSolver:
    def __init__(self, F, J, initial_guess, output_dir="output/"):
        """
        Initialize the system solver for Newton's method.

        Parameters:
        - F: Function that returns the vector of residuals
        - J: Function that returns the Jacobian matrix
        - initial_guess: Initial guess for the solution vector
        - output_dir: Directory to save output files
        """
        self.F = F
        self.J = J
        self.initial_guess = np.array(initial_guess, dtype=float)
        self.output_dir = output_dir
        self.iterations = []
        self.root = None
        self.converged = False

    def newton_method(self, tolerance=1e-6, max_iterations=100, debug=False):
        """
        Perform Newton-Raphson iterations to solve the system.

        Parameters:
        - tolerance: Convergence threshold
        - max_iterations: Maximum number of iterations
        - debug: Whether to record iteration data

        Returns:
        - x: Final solution vector
        """
        x = self.initial_guess.copy()
        for i in range(max_iterations):
            F_val = self.F(x)
            J_val = self.J(x)

            try:
                delta = np.linalg.solve(J_val, -F_val)
            except np.linalg.LinAlgError:
                raise RuntimeError("Jacobian is singular and cannot be inverted.")

            if debug:
                iteration_data = {
                    "iteration": i + 1,
                    "x": x.copy(),
                    "delta_norm": np.linalg.norm(delta),
                    "f_norm": np.linalg.norm(F_val),
                }
                self.iterations.append(iteration_data)

            x += delta

            if np.linalg.norm(delta) < tolerance:

                if debug:
                    iteration_data = {
                        "iteration": i + 1,
                        "x": x.copy(),
                        "delta_norm": np.linalg.norm(delta),
                        "f_norm": np.linalg.norm(F_val),
                    }
                    self.iterations.append(iteration_data)

                self.root = x
                self.converged = True
                return x

        raise RuntimeError("Maximum number of iterations reached without convergence.")
