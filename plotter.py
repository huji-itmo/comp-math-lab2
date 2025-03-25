from matplotlib import pyplot as plt
import numpy as np

from zero_finder import ZeroFinder


def plot_bisection(zero_finder: ZeroFinder):
    if not zero_finder.bisection_data:
        print("Run iterative_method with debug=True first")
        return

    left, right = zero_finder.a, zero_finder.b
    x = np.linspace(left, right, 1000)
    y = [zero_finder.func(xi) for xi in x]

    plt.figure(figsize=(12, 7))
    plt.plot(x, y, label="Function", color="navy")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)

    # Plot intervals and midpoints
    colors = plt.cm.viridis(np.linspace(0, 1, len(zero_finder.bisection_data)))
    for i, (data, color) in enumerate(zip(zero_finder.bisection_data, colors)):
        plt.axvspan(data["left"], data["right"], alpha=0.1, color=color)
        plt.scatter(
            data["mid"],
            0,
            color=color,
            s=50,
            zorder=2,
            label=f"Iter {i+1}" if i < 3 else None,
        )

    # Final result
    final_x = zero_finder.bisection_data[-1]["mid"]
    plt.scatter(
        final_x, 0, color="red", marker="*", s=200, zorder=3, label="Final Result"
    )

    plt.title("Bisection Method Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)

    if zero_finder.plot_path:
        plt.savefig(zero_finder.plot_path + "bisection.pdf", bbox_inches="tight")
        plt.savefig(zero_finder.plot_path + "bisection.png", bbox_inches="tight")
    plt.close()


def plot_newton(zero_finder: ZeroFinder):
    if not zero_finder.newton_data:
        print("Run newton_method with debug=True first")
        return

    plt.figure(figsize=(12, 7))
    x_vals = np.linspace(zero_finder.a, zero_finder.b, 1000)
    f_vals = [zero_finder.func(x) for x in x_vals]

    # Create single plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot function and iterations
    ax.plot(x_vals, f_vals, label="f(x)", color="blue")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)

    colors = plt.cm.plasma(np.linspace(0, 1, len(zero_finder.newton_data)))
    for i, (data, color) in enumerate(zip(zero_finder.newton_data, colors)):
        # Function plot annotations
        ax.scatter(data["x"], data["fx"], color=color, s=80, zorder=3)
        ax.plot(
            [data["x"], data["x_new"]],
            [data["fx"], 0],
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"Iter {i+1}" if i == 0 else "",
        )

    # Final result marker
    final_x = zero_finder.newton_data[-1]["x_new"]
    ax.scatter(final_x, 0, color="red", marker="*", s=200, zorder=4, label="Root")

    ax.set_title("Newton-Raphson Method Convergence")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if zero_finder.plot_path:
        plt.savefig(zero_finder.plot_path + "newton.pdf", bbox_inches="tight")
        plt.savefig(zero_finder.plot_path + "newton.png", bbox_inches="tight")
    plt.close()


def plot_simple_iteration(zero_finder: ZeroFinder):
    """Visualize the simple iteration method convergence steps"""
    if not zero_finder.simple_iter_data:
        print("Run simple_iteration_method with debug=True first")
        return

    plt.figure(figsize=(12, 7))
    x_vals = np.linspace(zero_finder.a, zero_finder.b, 1000)
    f_vals = [zero_finder.func(x) for x in x_vals]

    # Main function plot
    plt.plot(x_vals, f_vals, label="f(x)", color="blue")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # Iteration visualization
    colors = plt.cm.plasma(np.linspace(0, 1, len(zero_finder.simple_iter_data)))

    for i, (entry, color) in enumerate(zip(zero_finder.simple_iter_data, colors)):
        x_prev = entry["x_prev"]
        x_next = entry["x_next"]
        f_x_prev = zero_finder.func(x_prev)

        # Plot iteration step components
        plt.plot([x_prev, x_prev], [f_x_prev, 0], color=color, linestyle=":", alpha=0.5)
        plt.plot([x_prev, x_next], [0, 0], color=color, linestyle="-", alpha=0.7)
        plt.scatter(
            x_prev,
            f_x_prev,
            color=color,
            s=80,
            zorder=3,
            label=f"Iter {i+1}" if i == 0 else "",
        )
        plt.scatter(x_next, 0, color=color, marker="X", s=100, zorder=3)

    # Final root marker
    final_x = zero_finder.simple_iter_data[-1]["x_next"]
    plt.scatter(final_x, 0, color="red", marker="*", s=200, zorder=4, label="Root")

    plt.title("Simple Iteration Method Convergence")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    # Save plots if path specified
    if zero_finder.plot_path:
        plt.savefig(f"{zero_finder.plot_path}simple_iteration.pdf", bbox_inches="tight")
        plt.savefig(f"{zero_finder.plot_path}simple_iteration.png", bbox_inches="tight")
    plt.close()
