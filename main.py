import math
from latex import (
    generate_bisection_latex_table,
    generate_newton_latex_table,
    generate_simple_iter_latex_table,
)
from plotter import plot_bisection, plot_graph, plot_newton, plot_simple_iteration
from zero_finder import ZeroFinder


# Predefined equations with f, df, and d2f
equations = [
    {
        "id": 1,
        "name": "-2.4x³ + 1.27x² + 8.36x + 2.31",
        "f": lambda x: -2.4 * x**3 + 1.27 * x**2 + 8.36 * x + 2.31,
        "df": lambda x: -7.2 * x**2 + 2.54 * x + 8.36,
    },
    {
        "id": 2,
        "name": "5.74x³ - 2.95x² - 10.28x - 3.23",
        "f": lambda x: 5.74 * x**3 - 2.95 * x**2 - 10.28 * x - 3.23,
        "df": lambda x: 17.22 * x**2 - 5.9 * x - 10.28,
    },
    {
        "id": 3,
        "name": "x³ + 2.64x² - 5.41x - 11.76",
        "f": lambda x: x**3 + 2.64 * x**2 - 5.41 * x - 11.76,
        "df": lambda x: 3 * x**2 + 5.28 * x - 5.41,
    },
    {
        "id": 4,
        "name": "sin(x) - e^(-x)",
        "f": lambda x: math.sin(x) - math.exp(-x),
        "df": lambda x: math.cos(x) + math.exp(-x),
    },
    {
        "id": 5,
        "name": "x³ + 2.84x² - 5.606x - 14.766",
        "f": lambda x: x**3 + 2.84 * x**2 - 5.606 * x - 14.766,
        "df": lambda x: 3 * x**2 + 5.68 * x - 5.606,
    },
]


# Function to let the user select a function from the list
def select_function():
    print("\nAvailable Functions:")
    for eq in equations:
        print(f"{eq['id']}. {eq['name']}")
    while True:
        try:
            choice = int(input("Select function by ID (1-5): "))
            for eq in equations:
                if eq["id"] == choice:
                    print(f"Selected function: {eq['name']}")
                    return eq["f"], eq["df"], eq["name"]
            print("Invalid ID. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_interval():
    while True:
        try:
            a = float(input("Enter left endpoint of interval (a): "))
            b = float(input("Enter right endpoint of interval (b): "))
            if a >= b:
                print("Error: a must be less than b")
                continue
            return (a, b)
        except ValueError:
            print("Please enter valid numbers")


def get_epsilon():
    while True:
        try:
            eps = float(input("Enter tolerance (epsilon): "))
            if eps <= 0:
                print("Error: epsilon must be positive")
                continue
            return eps
        except ValueError:
            print("Please enter a valid number")


if __name__ == "__main__":
    f, df, name = select_function()
    interval = get_interval()
    epsilon = get_epsilon()
    zero_finder = ZeroFinder(f, df, interval, "output/")
    plot_graph(zero_finder, "output/graph_plot.png")

    try:
        print("\nRunning Bisection method:")
        root = zero_finder.bisection_method(tolerance=epsilon, debug=True)
        print(f"Bisection root: {root:.6f}")
        print(f"Bisection value: {f(root)}")
        print(f"Bisection iterations: {len(zero_finder.bisection_data)}")

        latex_str = generate_bisection_latex_table(zero_finder)
        with open("output/bisection.tex", "w") as file:
            file.write(latex_str)
        plot_bisection(zero_finder)

    except ValueError as e:
        print(f"Bisection error: {e}")
    except OverflowError as e:
        print(f"Bisection error: {e}")

    try:
        print("\nRunning Newton method:")
        root = zero_finder.newton_method(tolerance=epsilon, debug=True)
        print(f"Newton root: {root:.6f}")
        print(f"Newton value: {f(root)}")
        print(f"Newton iterations: {len(zero_finder.newton_data)}")

        latex_str = generate_newton_latex_table(zero_finder)
        with open("output/newton.tex", "w") as file:
            file.write(latex_str)
        plot_newton(zero_finder)

    except ValueError as e:
        print(f"Newton error: {e}")
    except OverflowError as e:
        print(f"Newton error: {e}")

    try:
        print("\nRunning Iterative method:")
        root = zero_finder.simple_iteration_method(tolerance=epsilon, debug=True)
        print(f"Iterative root: {root:.6f}")
        print(f"Iterative value: {f(root)}")
        print(f"Iterative iterations: {len(zero_finder.simple_iter_data)}")

        latex_str = generate_simple_iter_latex_table(zero_finder)
        with open("output/simple_iteration.tex", "w") as file:
            file.write(latex_str)
        plot_simple_iteration(zero_finder)

    except ValueError as e:
        print(f"Iterative error: {e}")
    except OverflowError as e:
        print(f"Iterative error: {e}")
