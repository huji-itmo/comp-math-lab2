from zero_finder import ZeroFinder


def generate_bisection_latex_table(zero_finder: ZeroFinder):
    if not zero_finder.bisection_data:
        return ""

    headers = ["Iteration", "$a$", "$b$", "$c$", "$f(c)$"]
    latex = []
    latex.append(r"\begin{tabular}{|c|c|c|c|c|}")
    latex.append(r"\hline")
    latex.append(" & ".join(headers) + r" \\")
    latex.append(r"\hline")

    for i, entry in enumerate(zero_finder.bisection_data):
        row = [
            str(i + 1),
            f"{entry['left']:.6f}",
            f"{entry['right']:.6f}",
            f"{entry['mid']:.6f}",
            f"{entry['f_mid']:.6f}",
        ]
        latex.append(" & ".join(row) + r" \\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")
    return "\n".join(latex)


def generate_newton_latex_table(zero_finder: ZeroFinder):
    if not zero_finder.newton_data:
        return ""

    headers = ["Iteration", "$x_n$", "$f(x_n)$", "$f'(x_n)$", "$x_{n+1}$"]
    latex = []
    latex.append(r"\begin{tabular}{|c|c|c|c|c|}")
    latex.append(r"\hline")
    latex.append(" & ".join(headers) + r" \\")
    latex.append(r"\hline")

    for i, entry in enumerate(zero_finder.newton_data):
        row = [
            str(i + 1),
            f"{entry['x']:.6f}",
            f"{entry['fx']:.6f}",
            f"{entry['dfx']:.6f}",
            f"{entry['x_new']:.6f}",
        ]
        latex.append(" & ".join(row) + r" \\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")
    return "\n".join(latex)


def generate_simple_iter_latex_table(zero_finder: ZeroFinder):
    if not zero_finder.simple_iter_data:
        return ""

    headers = ["Iteration", "$x_n$", "$x_{n+1}$", "$f(x_{n+1})$", "Error"]
    latex = [
        r"\begin{tabular}{|c|c|c|c|c|}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]

    for entry in zero_finder.simple_iter_data:
        row = [
            str(entry["iteration"]),
            f"{entry['x_prev']:.6f}",
            f"{entry['x_next']:.6f}",
            f"{entry['f_x_next']:.6f}",
            f"{entry['error']:.2e}",
        ]
        latex.append(" & ".join(row) + r" \\")
        latex.append(r"\hline")

    latex.append(r"\end{tabular}")
    return "\n".join(latex)


def generate_newton_system_latex_table(solver):
    """
    Generate a LaTeX table for Newton's method iterations.

    Parameters:
    - solver: Instance of SystemSolver after running Newton's method

    Returns:
    - LaTeX string representation of the table
    """
    iterations = solver.iterations
    n_vars = len(solver.initial_guess)

    # Start LaTeX table
    latex = (
        "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|"
        + "c|" * n_vars
        + "c|c|}\n\\hline\n"
    )
    latex += "Iteration"

    for var_idx in range(n_vars):
        latex += f" & x_{var_idx + 1}"

    latex += " & $\\|\\Delta x\\|$ & $\\|F(x)\\|$ \\\\ \\hline\n"

    # Add rows
    for iter_data in iterations:
        latex += f"{iter_data['iteration']} & "
        x_values = " & ".join(f"{xi:.6f}" for xi in iter_data["x"])
        latex += f"{x_values} & {iter_data['delta_norm']:.3e} & {iter_data['f_norm']:.3e} \\\\ \\hline\n"

    latex += "\\end{tabular}\n\\caption{Newton's Method Iterations for System of Equations}\n\\end{table}"
    return latex
