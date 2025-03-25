from latex import (
    generate_bisection_latex_table,
    generate_newton_latex_table,
    generate_simple_iter_latex_table,
)
from plotter import plot_bisection, plot_newton, plot_simple_iteration
from zero_finder import ZeroFinder


def f(x):
    return x**3 + 2.84 * x**2 - 5.606 * x - 14.766


def df(x):
    return 3 * x**2 + 5.68 * x - 5.606


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
    interval = get_interval()
    epsilon = get_epsilon()
    zero_finder = ZeroFinder(f, df, interval, "output/")

    try:
        print("\nRunning Bisection method:")
        root = zero_finder.bisection_method(tolerance=epsilon, debug=True)
        print(f"Bisection root: {root:.6f}")
        latex_str = generate_bisection_latex_table(zero_finder)
        with open("output/bisection.tex", "w") as file:
            file.write(latex_str)
        plot_bisection(zero_finder)

    except ValueError as e:
        print(f"Bisection error: {e}")

    try:
        print("\nRunning Newton method:")
        root = zero_finder.newton_method(tolerance=epsilon, debug=True)
        print(f"Newton root: {root:.6f}")
        latex_str = generate_newton_latex_table(zero_finder)
        with open("output/newton.tex", "w") as file:
            file.write(latex_str)
        plot_newton(zero_finder)

    except ValueError as e:
        print(f"Newton error: {e}")

    try:
        print("\nRunning Iterative method:")
        root = zero_finder.simple_iteration_method(tolerance=epsilon, debug=True)
        print(f"Iterative root: {root:.6f}")
        latex_str = generate_simple_iter_latex_table(zero_finder)
        with open("output/simple_iteration.tex", "w") as file:
            file.write(latex_str)
        plot_simple_iteration(zero_finder)

    except ValueError as e:
        print(f"Iterative error: {e}")
