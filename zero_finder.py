class ZeroFinder:
    def __init__(self, func, derivative, interval, plot_path=""):
        self.func = func
        self.derivative = derivative
        self.a, self.b = interval
        self.plot_path = plot_path
        self.bisection_data = []
        self.newton_data = []
        self.simple_iter_data = []

        if self.a >= self.b:
            raise ValueError("Interval must be in the form [a, b] where a < b")

    def bisection_method(self, tolerance=1e-6, max_iterations=1000, debug=False):
        self.bisection_data = []
        a, b = self.a, self.b
        fa = self.func(a)
        fb = self.func(b)

        if fa * fb >= 0:
            raise ValueError("Function must have opposite signs at endpoints")

        for _ in range(max_iterations):
            c = (a + b) / 2
            fc = self.func(c)

            if debug:
                self.bisection_data.append(
                    {"left": a, "right": b, "mid": c, "f_mid": fc}
                )

            if abs(fc) < tolerance and (b - a) / 2 < tolerance:
                print("function value return:", abs(fc) < tolerance)
                print("function argument return:", (b - a) / 2 < tolerance)

                return c

            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

        return (a + b) / 2

    def newton_method(
        self, initial_guess=None, tolerance=1e-6, max_iterations=1000, debug=False
    ):
        self.newton_data = []
        x = initial_guess if initial_guess else (self.a + self.b) / 2

        for _ in range(max_iterations):
            fx = self.func(x)
            dfx = self.derivative(x)
            if dfx == 0:
                raise ValueError("Zero derivative encountered")
            x_new = x - fx / dfx

            if debug:
                self.newton_data.append({"x": x, "fx": fx, "dfx": dfx, "x_new": x_new})

            if abs(x_new - x) < tolerance:
                return x_new
            x = x_new

        return x

    def simple_iteration_method(
        self, initial_guess=None, tolerance=1e-6, max_iterations=1000, debug=False
    ):
        self.simple_iter_data = []
        x0 = initial_guess if initial_guess else (self.a + self.b) / 2

        # Validate contraction condition
        try:
            df_a = self.derivative(self.a)
            df_b = self.derivative(self.b)
            if abs(df_a) >= 1 or abs(df_b) >= 1:
                print("Derivative condition not satisfied (|phi'| < 1 required)")

            M = max(df_a, df_b)
            print("M =", M)
            if M < 0:
                lam = -1 / M
            else:
                lam = 1 / M
            print("lambda =", lam)
        except ZeroDivisionError:
            raise ValueError("Cannot compute λ - zero derivative at boundaries")

        phi = lambda x: x + lam * self.func(x)
        x_prev = x0

        phi_prime = lambda x: 1 + lam * self.derivative(x)

        print("phi'(a)=", phi_prime(self.a))
        print("phi'(b)=", phi_prime(self.b))

        for iter_count in range(max_iterations):
            x_next = phi(x_prev)
            error = abs(x_next - x_prev)

            if debug:
                self.simple_iter_data.append(
                    {
                        "iteration": iter_count + 1,
                        "x_prev": x_prev,
                        "x_next": x_next,
                        "f_x_next": self.func(x_next),
                        "error": error,
                    }
                )

            if error < tolerance and abs(self.func(x_next)) < tolerance:
                return x_next

            x_prev = x_next

        raise ValueError(f"No convergence in {max_iterations} iterations")
