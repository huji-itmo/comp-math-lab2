from latex import generate_newton_system_latex_table
from plotter import plot_newton_system, plot_system
from system_solver import SystemSolver
import numpy as np

systems = [
    {
        "id": 1,
        "name": "sin(x+1) - y = 1.2; 2x + cos(y) = 2",
        "F": lambda x: np.array(
            [np.sin(x[0] + 1) - x[1] - 1.2, 2 * x[0] + np.cos(x[1]) - 2]
        ),
        "J": lambda x: np.array([[np.cos(x[0] + 1), -1], [2, -np.sin(x[1])]]),
    },
    {
        "id": 2,
        "name": "sin(x) + 2y = 2; x + cos(y-1) = 0.7",
        "F": lambda x: np.array(
            [np.sin(x[0]) + 2 * x[1] - 2, x[0] + np.cos(x[1] - 1) - 0.7]
        ),
        "J": lambda x: np.array([[np.cos(x[0]), 2], [1, -np.sin(x[1] - 1)]]),
    },
    {
        "id": 3,
        "name": "sin(x+y) = 1.5x -0.1; x² + 2y² =1",
        "F": lambda x: np.array(
            [np.sin(x[0] + x[1]) - 1.5 * x[0] + 0.1, x[0] ** 2 + 2 * x[1] ** 2 - 1]
        ),
        "J": lambda x: np.array(
            [[np.cos(x[0] + x[1]) - 1.5, np.cos(x[0] + x[1])], [2 * x[0], 4 * x[1]]]
        ),
    },
    {
        "id": 4,
        "name": "sin(x+y) = 1.5x -0.1; x² + 2y² =1",
        "F": lambda x: np.array(
            [np.cos(x[0] - 1) + x[1] - 0.5, x[0] - np.cos(x[1]) - 3]
        ),
        "J": lambda x: np.array([[-np.sin(x[0] - 1), 1], [1, np.sin(x[1])]]),
    },
]


def get_initial_guess():
    while True:
        try:
            x0 = float(input("Enter initial guess for x: "))
            y0 = float(input("Enter initial guess for y: "))
            return [x0, y0]
        except ValueError:
            print("Please enter valid numbers.")


def get_epsilon():
    while True:
        try:
            eps = float(input("Enter tolerance (epsilon): "))
            if eps <= 0:
                print("Epsilon must be positive.")
                continue
            return eps
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    # Вывод списка доступных систем
    print("Available systems:")
    for system in systems:
        print(f"{system['id']}. {system['name']}")

    # Выбор системы пользователем
    while True:
        try:
            system_id = int(input("Select system by ID: "))
            selected_system = next((s for s in systems if s["id"] == system_id), None)
            if selected_system is None:
                print("Invalid ID. Please select again.")
            else:
                break
        except ValueError:
            print("Please enter a valid integer.")

    initial_guess = get_initial_guess()
    epsilon = get_epsilon()
    max_iterations = 100  # Could also ask user for this

    # Initialize the system solver
    system_solver = SystemSolver(
        F=selected_system["F"],
        J=selected_system["J"],
        initial_guess=initial_guess,
        output_dir="output/",
    )

    try:
        print("\nRunning Newton's method for system of equations:")
        root = system_solver.newton_method(
            tolerance=epsilon, max_iterations=max_iterations, debug=True
        )
        print(f"Root found: {root}")
        print(f"value=", system_solver.F(root))
        print(f"iterations=", len(system_solver.iterations))

        # Generate LaTeX table
        latex_table = generate_newton_system_latex_table(system_solver)
        with open("output/newton_system.tex", "w") as f:
            f.write(latex_table)

        # Plot convergence
        plot_newton_system(system_solver)
        # plot_system(system_solver)

    except RuntimeError as e:
        print(f"Error during Newton's method: {e}")
