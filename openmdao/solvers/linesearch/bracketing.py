"""
This linesearch brackets a minimum and then uses a variation of
Brent's algorithm (involving successive parabolic interpolations) to
home in on the minimum. It does not use gradients.
"""

import warnings
from copy import deepcopy, copy

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.solver import LinesearchSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.om_warnings import issue_warning, SolverWarning


class BracketingLS(LinesearchSolver):
    """
    This linesearch brackets a minimum and then uses a variation of
    Brent's algorithm (involving successive parabolic interpolations) to
    home in on the minimum. It does not use gradients.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _analysis_error_raised : bool
        Flag is set to True if a subsystem raises an AnalysisError.
    """

    SOLVER = "LS: BRKT"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self._analysis_error_raised = False
        self.mu_upper = None
        self.mu_lower = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        self.options.declare("alpha", default=1.0, lower=0.0, desc="Initial line search step.")
        self.options.declare(
            "retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised."
        )
        self.options.declare(
            "alpha_max", default=10.0, lower=1.0, desc="Initial max line search step length for formward tracking"
        )
        self.options.declare("beta", default=2.0, lower=1.0, desc="Bracketing expansion/contraction factor")
        self.options.declare("use_spi", default=True, desc="Use successive parabolic interpolation instead of Brent")
        self.options.declare(
            "spi_tol",
            default=1e-4,
            desc="Relative difference in alpha between minimum of previous and current iteration",
        )

    def _line_search_objective(self):
        """
        Calculate the objective function of the line search.

        Returns
        -------
        float
            Line search objective (residual norm).
        """

        # Compute the penalized residual norm
        if self._lower_finite_mask is not None and self._upper_finite_mask is not None:
            if self.mu_lower is not None and self.mu_upper is not None:
                system = self._system()
                u = system._outputs.asarray()
                resids = system._residuals.asarray()
                lb = self._lower_bounds
                ub = self._upper_bounds
                lb_mask = self._lower_finite_mask
                ub_mask = self._upper_finite_mask

                penalty = np.zeros(u.size)

                t_lower = u[lb_mask] - lb[lb_mask]
                t_upper = ub[ub_mask] - u[ub_mask]

                if t_lower.size > 0:
                    penalty[lb_mask] += np.sum(self.mu_lower * -np.log(t_lower + 1e-10))

                if t_upper.size > 0:
                    penalty[ub_mask] += np.sum(self.mu_upper * -np.log(t_upper + 1e-10))

                return np.linalg.norm(resids + penalty)

        # Compute the unpenalized residual norm
        return self._iter_get_norm()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        system = self._system()
        self.alpha = self.options["alpha"]
        self.alpha_max = self.options["alpha_max"]
        buffer = 1e-13  # Buffer by which to pull alpha max away from the bound (absolute magnitude in states)

        u = system._outputs
        du = system._vectors["output"]["linear"]

        ub = self._upper_bounds
        lb = self._lower_bounds

        # Exit codes
        fwd = 0
        bak = 1
        bnd = 2

        self._run_apply()
        phi0 = self._line_search_objective()
        if phi0 == 0.0:
            phi0 = 1.0
        self._phi0 = phi0

        # --- Limit alpha max to satisfy bounds ---
        # Limit alpha max to find the value that will prevent the line search
        # from exceeding bounds and from exceeding the specified alpha max
        d_alpha = 0  # required change in step size, relative to the du vector
        u.add_scal_vec(self.options["alpha_max"], du)

        # Find the largest amount a bound is violated
        # Where positive d_alpha means a bound is violated
        mask = du.asarray() != 0
        if mask.any():
            abs_du_mask = np.abs(du.asarray()[mask] * self.options["alpha_max"])
            u_mask = u.asarray()[mask]

            # Check for states violating the lower bound
            if lb is not None:
                max_d_alpha = np.amax((lb[mask] - u_mask) / abs_du_mask)
                if max_d_alpha > d_alpha:
                    d_alpha = max_d_alpha

            # Check for states violating the upper bound
            if ub is not None:
                max_d_alpha = np.amax((u_mask - ub[mask]) / abs_du_mask)
                if max_d_alpha > d_alpha:
                    d_alpha = max_d_alpha

        # Adjust alpha_max so that it goes right to the most restrictive bound,
        # but pull it away from the obund by the buffer amount so the penalty
        # doesn't return NaN.
        if d_alpha > 0:
            self.alpha_max *= 1 - d_alpha
            self.alpha_max -= buffer / np.linalg.norm(du.asarray())

        # Move the states back
        u.add_scal_vec(-self.options["alpha_max"], du)

        # --- Setup and evaluate the first point ---
        self.bracket_low = {"alpha": 0, "phi": self._phi0}
        self.bracket_mid = {"alpha": None, "phi": None}
        self.bracket_high = {"alpha": self.alpha, "phi": None}

        # Check that the initial step doesn't exceed alpha max
        bounds_enforced = False
        if self.alpha > self.alpha_max:
            bounds_enforced = True
            self.alpha = self.bracket_high["alpha"] = self.alpha_max

        # Move the states to the first alpha
        u.add_scal_vec(self.alpha, du)
        self._single_iteration()
        phi = self.bracket_high["phi"] = self._line_search_objective()
        self._iter_count += 1

        self._mpi_print(self._iter_count, phi, phi / self._phi0)

        # TODO: Record this iteration

        # If phi at the first alpha is greater than the original point,
        # we assume there's a minimum between alpha of 0 and the current
        # point.
        if self.bracket_high["phi"] >= self.bracket_low["phi"]:
            return bak

        # If phi is less that the original phi and it's not on a bound,
        # we want to continue the search forward.
        if not bounds_enforced:
            self.bracket_mid = deepcopy(self.bracket_high)
            self.bracket_high["alpha"] *= self.options["beta"]
            self.bracket_high["phi"] = None
            return fwd

        # Otherwise, report that the line search is on a bound and no
        # more bracketing operations can be done.
        return bnd

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        self._analysis_error_raised = False

        # Hybrid newton support.
        if self._do_subsolve and self._iter_count > 0:
            self._solver_info.append_solver()

            try:
                cache = self._solver_info.save_cache()
                self._gs_iter()
                self._run_apply()

            except AnalysisError as err:
                self._solver_info.restore_cache(cache)

                if self.options["retry_on_analysis_error"]:
                    self._analysis_error_raised = True

                else:
                    raise err

            finally:
                self._solver_info.pop()

        else:
            self._run_apply()

    def _fwd_bracketing(self):
        system = self._system()
        maxiter = self.options["maxiter"]
        u = system._outputs
        du = system._vectors["output"]["linear"]

        bound_hit = False  # Set to true if a bound or alpha max is reached

        if self.bracket_high["alpha"] > self.alpha_max:
            bound_hit = True
            self.bracket_high["alpha"] = self.alpha_max

        # Initialize the upper bracket phi
        u.add_scal_vec(self.bracket_high["alpha"] - self.alpha, du)
        self.alpha = self.bracket_high["alpha"]
        self._single_iteration()
        phi = self.bracket_high["phi"] = self._line_search_objective()
        self._iter_count += 1

        self._mpi_print(self._iter_count, phi, phi / self._phi0)

        # Keep forward tracking the bracket until a minimum has been bracketed
        while self.bracket_mid["phi"] > self.bracket_high["phi"] or self.bracket_mid["phi"] > self.bracket_low["phi"]:
            # If the max number of iterations has been reached, break out and return the value
            if self._iter_count >= maxiter:
                return True

            # If a bound has been hit and it makes it this far, it means it has not been bracketed
            if bound_hit:
                return True

            # Shift the brackets over and compute the alpha for the new high
            self.bracket_low = deepcopy(self.bracket_mid)
            self.bracket_mid = deepcopy(self.bracket_high)
            self.bracket_high["alpha"] *= self.options["beta"]

            # Limit the step if necessary
            if self.bracket_high["alpha"] > self.alpha_max:
                bound_hit = True
                self.bracket_high["alpha"] = self.alpha_max

            # Move the states to the new alpha
            u.add_scal_vec(self.bracket_high["alpha"] - self.alpha, du)
            self.alpha = self.bracket_high["alpha"]
            self._single_iteration()
            phi = self.bracket_high["phi"] = self._line_search_objective()
            self._iter_count += 1

            self._mpi_print(self._iter_count, phi, phi / self._phi0)

        return False

    def _brent(self):
        self.SOLVER = "LS: BRENT"
        tol = 1e-2

        system = self._system()
        states = system._outputs
        du = system._vectors["output"]["linear"]

        # Set the golden ratio
        maxiter = self.options["maxiter"]
        c = (3 - 5 ** (1 / 2)) / 2
        eps = np.finfo(float).eps
        e = 0
        d = 0

        # Set the upper and lower bracket step sizes
        a = min(self.bracket_low["alpha"], self.bracket_high["alpha"])
        b = max(self.bracket_low["alpha"], self.bracket_high["alpha"])

        # Set the midpoint step and objective value
        if self.bracket_mid["alpha"] is None:
            self.bracket_mid["alpha"] = x = a + c * (b - a)
            states.add_scal_vec(x - self.alpha, du)
            self.alpha = x
            self._single_iteration()
            phi = self.bracket_mid["phi"] = fx = self._line_search_objective()
            self._iter_count += 1

            self._mpi_print(self._iter_count, phi, phi / self._phi0)

            # If we are not guaranteed a minimum within the bracket,
            # just take the Newton step. As far as we can tell,
            # using the combination of the penalized residual in the line search
            # and the "unsteady" Newton linear system formulation does not guarantee
            # that the line search will be searching in a downhill direction;
            # d(phi)/d(alpha) at alpha = 0 is not necessarily negative
            if (
                self.bracket_mid["phi"] >= self.bracket_high["phi"]
                or self.bracket_mid["phi"] >= self.bracket_low["phi"]
            ):
                states.add_scal_vec(self.bracket_high["alpha"] - self.alpha, du)
                self.alpha = self.bracket_high["alpha"]
                self._single_iteration()
                phi = self._line_search_objective()
                self._iter_count += 1

                self._mpi_print(self._iter_count, phi, phi / self._phi0)

                return

        # Initialize v, w, x, fv, fw, and fx
        # Set x and fx to the point with the lowest phi value (mid point)
        x = copy(self.bracket_mid["alpha"])
        fx = copy(self.bracket_mid["phi"])

        # Set w and fw to the point with the next lowest phi and v and fv to the final one
        if self.bracket_low["phi"] < self.bracket_high["phi"]:
            w = copy(self.bracket_low["alpha"])
            fw = copy(self.bracket_low["phi"])
            v = copy(self.bracket_high["alpha"])
            fv = copy(self.bracket_high["phi"])
        else:
            v = copy(self.bracket_low["alpha"])
            fv = copy(self.bracket_low["phi"])
            w = copy(self.bracket_high["alpha"])
            fw = copy(self.bracket_high["phi"])

        # Loop until reaching the maximum number of iterations
        while self._iter_count < maxiter:
            m = 0.5 * (b + a)  # Start with bisection
            tol1 = tol + abs(x) * eps
            tol2 = 2 * tol1

            if abs(x - m) > tol2 - 0.5 * (b - a):
                p = q = r = 0
                if abs(e) > tol:
                    # Fit parabola
                    r = (x - w) * (fx - fv)
                    q = (x - v) * (fx - fw)
                    p = (x - v) * q - (x - w) * r
                    q = 2 * (q - r)

                    if q > 0:
                        p = -p
                    else:
                        q = -q

                    r = e
                    e = d

                if abs(p) < abs(0.5 * q * r) and p < q * (a - x) and p < q * (b - x):
                    # Parabolic inerpolation step
                    d = p / q
                    u = x + d
                    # f must not be evaluated too close to a or b
                    if u - a < tol2 or b - u < tol2:
                        d = tol if x < m else -tol

                else:
                    # Golden section step
                    e = (b - x) if x < m else (a - x)
                    d = c * e

                # f must not be evaluated too close to x
                if abs(d) >= tol:
                    u = x + d
                elif d > 0:
                    u = x + tol
                else:
                    u = x - tol

                # Move the states to u and evaluate f(u)
                states.add_scal_vec(u - self.alpha, du)
                self.alpha = u
                self._single_iteration()
                phi = fu = self._line_search_objective()
                self._iter_count += 1

                self._mpi_print(self._iter_count, phi, phi / self._phi0)

                # Update a, b, v, w, and x
                if fu <= fx:
                    if u < x:
                        b = x
                    else:
                        a = x

                    v, fv = w, fw
                    w, fw = x, fx
                    x, fx = u, fu

                else:
                    if u < x:
                        a = u
                    else:
                        b = u

                    if fu <= fw or w == x:
                        v, fv = w, fw
                        w, fw = u, fu
                    elif fu <= fv or v == x or v == w:
                        v, fv = u, fu
            else:
                return

        return

    def _spi(self):
        self.SOLVER = "LS: SPI"
        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        maxiter = self.options["maxiter"]

        # Set the midpoint step and objective value
        if self.bracket_mid["alpha"] is None:
            self.bracket_mid["alpha"] = 0.5 * (self.bracket_low["alpha"] + self.bracket_high["alpha"])
            u.add_scal_vec(self.bracket_mid["alpha"] - self.alpha, du)
            self.alpha = self.bracket_mid["alpha"]
            self._single_iteration()
            phi = self.bracket_mid["phi"] = self._line_search_objective()
            self._iter_count += 1

            self._mpi_print(self._iter_count, phi, phi / self._phi0)

            # If we are not guaranteed a minimum within the bracket,
            # just take the Newton step. As far as we can tell,
            # using the combination of the penalized residual in the line search
            # and the "unsteady" Newton linear system formulation does not guarantee
            # that the line search will be searching in a downhill direction;
            # d(phi)/d(alpha) at alpha = 0 is not necessarily negative
            if (
                self.bracket_mid["phi"] >= self.bracket_high["phi"]
                or self.bracket_mid["phi"] >= self.bracket_low["phi"]
            ):
                u.add_scal_vec(self.bracket_high["alpha"] - self.alpha, du)
                self.alpha = self.bracket_high["alpha"]
                self._single_iteration()
                phi = self._line_search_objective()
                self._iter_count += 1

                self._mpi_print(self._iter_count, phi, phi / self._phi0)

                return

        # Initialize value for parabola minimum to enter the while loop
        x_min = np.inf
        y = self.bracket_mid["alpha"]

        # Loop until reaching the maximum number of iterations
        while self._iter_count < maxiter and abs((x_min - y) / y) > self.options["spi_tol"]:
            x = self.bracket_low["alpha"]
            fx = self.bracket_low["phi"]
            y = self.bracket_mid["alpha"]
            fy = self.bracket_mid["phi"]
            z = self.bracket_high["alpha"]
            fz = self.bracket_high["phi"]

            # Find the minimum of the parabola and evaluate it
            x_min = y + 0.5 * ((y - z) ** 2 * (fy - fx) - (y - x) ** 2 * (fy - fz)) / (
                (y - x) * (fy - fz) - (y - z) * (fy - fx)
            )

            # Move the states to u and evaluate f(u)
            u.add_scal_vec(x_min - self.alpha, du)
            self.alpha = x_min
            self._single_iteration()
            phi = fx_min = self._line_search_objective()
            self._iter_count += 1

            self._mpi_print(self._iter_count, phi, phi / self._phi0)

            # Update the bracket based on the function value at x_min
            if x < x_min < y:
                # The new phi must be less than both fx and fy to guarantee a minimum within x and y
                if fx_min <= fx and fx_min <= fy:
                    self.bracket_mid["alpha"] = x_min
                    self.bracket_mid["phi"] = fx_min
                    self.bracket_high["alpha"] = y
                    self.bracket_high["phi"] = fy
                # Otherwise there is a minimum between x_min and z
                else:
                    self.bracket_low["alpha"] = x_min
                    self.bracket_low["phi"] = fx_min
            elif y < x_min < z:
                # The new phi must be less than both fx and fy to guarantee a minimum within y and z
                if fx_min <= fy and fx_min <= fz:
                    self.bracket_mid["alpha"] = x_min
                    self.bracket_mid["phi"] = fx_min
                    self.bracket_low["alpha"] = y
                    self.bracket_low["phi"] = fy
                # Otherwise there is a minimum between x and x_min
                else:
                    self.bracket_high["alpha"] = x_min
                    self.bracket_high["phi"] = fx_min
            # Somewhow the parabola minimum is outside the current bracket
            else:
                u.add_scal_vec(y - self.alpha, du)
                self.alpha = y
                self._single_iteration()
                phi = self._line_search_objective()
                self._iter_count += 1

                self._mpi_print(self._iter_count, phi, phi / self._phi0)
                return

        return

    def _solve(self):
        """
        Run the iterative solver.
        """
        self.SOLVER = "LS: BRKT"
        self._iter_count = 0
        brkt_dir = self._iter_initialize()

        fwd = 0
        bnd = 2

        if brkt_dir == bnd:
            return

        if brkt_dir == fwd:
            if self._fwd_bracketing():
                return

        if self.options["use_spi"]:
            self._spi()

        else:
            self._brent()