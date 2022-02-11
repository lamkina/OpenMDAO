"""
A bracketing and pinpointing linesearch that uses the inner product
between the residual and the Newton step vector.

InnerProductLS -- Enforces bounds using the vector method.
"""

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.solver import LinesearchSolver

# from openmdao.recorders.recording_iteration_stack import Recording


class InnerProductLS(LinesearchSolver):
    """
    Inner product linesearch
    Attributes
    ----------
    _analysis_error_raised : bool
        Flag is set to True if a subsystem raises an AnalysisError.
    """

    SOLVER = "LS: IP"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        Parameters
        ----------
        **kwargs : dict
            Options dictionary.
        """
        super().__init__(**kwargs)

        self._analysis_error_raised = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt.declare(
            "beta",
            default=2.0,
            types=float,
            desc="Bracketing expansion factor. "
            + "Beta is multiplied with the upper bracket step to expand the bracket.",
        )
        opt.declare(
            "rho",
            default=0.5,
            types=float,
            desc="Illinois algorithm contraction factor. "
            + "This factor is useful for increasing the convergence rate "
            + "when the euclidian norm of the residual vector is very "
            + "large at one of the bracketing points.",
        )
        opt.declare(
            "alpha_max",
            default=16.0,
            types=float,
            desc="Largest allowable bracketing step.  The bracketing "
            + "phase will terminate if the upper bracket step goes beyond this limit.",
        )
        opt.declare("alpha", default=1.0, types=float, desc="Initial upper bracket step.")
        opt.declare("maxiter", default=10, types=int, desc="Maximum iterations taken by the root finding algorithm.")
        opt.declare(
            "root_method",
            default="illinois",
            values=["illinois", "brent", "secant"],
            desc="Name of the root finding algorithm.",
        )
        opt.declare("retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised.")

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        for unused_option in ("rtol", "atol", "err_on_non_converge"):
            opt.undeclare(unused_option)

    def _evaluate_residuals(self):
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

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.
        Returns
        -------
        float
            residual norm after initial iteration
        float
            initial error.
        float
            error at the first iteration.
        """

        self.SOLVER = "LS: IP INIT"
        self.bracket = {
            "alpha": {"lower": 0.0, "upper": self.options["alpha"]},
            "phi": {"lower": 0.0, "upper": 0.0},
        }
        self.alpha_max = self.options["alpha_max"]
        system = self._system()
        buffer = 1e-13

        u = system._outputs
        du = system._vectors["output"]["linear"]

        ub = self._upper_bounds
        lb = self._lower_bounds

        is_bracketed = False

        self._run_apply()

        # Store the initial residual norm for recording
        self.bracket["phi"]["lower"], self._norm0 = self._linesearch_objective()
        self._phi0 = self.bracket["phi"]["lower"]

        # --- Limit alpha max to satisfy bounds ---
        # Limit alpha max to find the value that will prevent the line search
        # from exceeding bounds and from exceeding the specified alpha max
        d_alpha = 0  # required change in step size, relative to the du vector
        u.add_scal_vec(self.options["alpha_max"], du)
        self.alpha = self.options["alpha_max"]

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

        if self.bracket["alpha"]["upper"] < self.alpha_max:
            # If the upper bracket is less than the computed alpha max, move the
            # states to the upper bracket.
            u.add_scal_vec(self.bracket["alpha"]["upper"] - self.alpha, du)
            self.alpha = self.bracket["alpha"]["upper"]
        else:
            # Else, move the states to the computed alpha max and set
            # the upper bracket equal to alpha max.  If this condition is met,
            # then we set the is_bracketed flag to true because we can no
            # longer search further along this direction.
            u.add_scal_vec(self.alpha_max - self.alpha)
            self.bracket["alpha"]["upper"] = self.alpha_max
            self.alpha = self.bracket["alpha"]["upper"]
            is_bracketed = True

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            phi, norm = self._linesearch_objective()
            self._iter_count += 1

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options["retry_on_analysis_error"]:
                self._analysis_error_raised = True
            else:
                raise err

            norm = np.nan

        self.bracket["phi"]["upper"] = phi

        self._mpi_print(self._iter_count, norm, self.alpha)

        return norm, is_bracketed

    def _linesearch_objective(self):
        """
        Moves the state vector the difference between the new step length
        and old step length along the Newton step direction.  Then,
        evaluate the residuals and take the inner product between the
        Newton step and residual vectors.
        Parameters
        ----------
        alpha_new : float
            New step length
        alpha_old : float
            Old step length
        Returns
        -------
        float
            Inner product between the Newton step and residual vectors.
        """
        system = self._system()
        u = system._outputs.asarray()
        du = system._vectors["output"]["linear"].asarray()
        residuals = system._residuals.asarray().copy()

        if self._lower_finite_mask is not None and self._upper_finite_mask is not None:
            if self.mu_lower is not None and self.mu_upper is not None:
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

                residuals += penalty

        return np.dot(du, residuals), np.linalg.norm(residuals)

    def _bracketing(self):
        """Bracketing phase of the linesearch
        Parameters
        ----------
        phi : float
            Residual norm at current points
        rec : OpenMDAO recorder
            The recorder for this linesearch
        Returns
        -------
        float
            Function value at the upper bracket step
        """
        self.SOLVER = "LS: IP BRKT"
        system = self._system()
        beta = self.options["beta"]
        maxiter = self.options["maxiter"]

        u = system._outputs
        du = system._vectors["output"]["linear"]

        while (
            self.bracket["alpha"]["upper"] * beta < self.alpha_max
            and self._iter_count < maxiter
            and (
                np.sign(self.bracket["phi"]["upper"]) * np.sign(self.bracket["phi"]["lower"]) > 0
                or self._analysis_error_raised
            )
        ):

            # Move the lower bracket step and value
            self.bracket["alpha"]["lower"], self.bracket["phi"]["lower"] = (
                self.bracket["alpha"]["upper"],
                self.bracket["phi"]["upper"],
            )

            # Update the upper bracket step
            self.bracket["alpha"]["upper"] *= beta

            # Compute the relative step between the bracket endpoints
            # Move the relative step between the new step and previous step
            u.add_scal_vec(self.bracket["alpha"]["upper"] - self.alpha, du)
            self.alpha = self.bracket["alpha"]["upper"]

            cache = self._solver_info.save_cache()

            try:
                self._evaluate_residuals()
                self._iter_count += 1

                # Compute the new upper bracket value
                self.bracket["phi"]["upper"], norm = self._linesearch_objective()

                # rec.abs = norm
                # rec.rel = norm / self._norm0
                # rec.alpha = self.alpha

            except AnalysisError as err:
                self._solver_info.restore_cache(cache)
                self._iter_count += 1

                if self.options["retry_on_analysis_error"]:
                    self._analysis_error_raised = True
                    # rec.abs = np.nan
                    # rec.rel = np.nan

                else:
                    raise err

            self._mpi_print(self._iter_count, norm, self.alpha)

    def _inv_quad_interp(self, alpha_mid, phi_mid):
        """Calls the inverse quadratic interpolation helper function
        Parameters
        ----------
        s_c : float
            Step length at a midpoint of the bracket
        g_c : float
            Inner product at step length 'c'
        Returns
        -------
        float
            Approximate root for the inner product
        """
        alpha_upper = self.bracket["alpha"]["upper"]
        alpha_lower = self.bracket["alpha"]["lower"]
        phi_upper = self.bracket["phi"]["upper"]
        phi_lower = self.bracket["phi"]["lower"]
        return _inv_quad_interp(alpha_upper, phi_upper, alpha_lower, phi_lower, alpha_mid, phi_mid)

    def _swap_bracket(self):
        """Swaps upper and lower points in the bracket"""
        self.bracket["alpha"]["upper"], self.bracket["alpha"]["lower"] = (
            self.bracket["alpha"]["lower"],
            self.bracket["alpha"]["upper"],
        )
        self.bracket["phi"]["upper"], self.bracket["phi"]["lower"] = (
            self.bracket["phi"]["lower"],
            self.bracket["phi"]["upper"],
        )

    def _brentq(self):
        """
        Brent's method for finding the root of a function.

        Reference
        ---------
        Brent, Richard P. 1973. Algorithms for Minimization without
        Derivatives. Algorithms for Minimization without Derivatives.
        Englewood Cliffs, New Jersey: Prentice-Hall.
        """
        self.SOLVER = "LS: IP BRENT"
        options = self.options
        maxiter = options["maxiter"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        # Swap a and b
        if abs(self.bracket["phi"]["lower"]) < abs(self.bracket["phi"]["upper"]):
            # State vector is always at point 'a' after bracketing.
            # If a swap occurs, we know it simply switches to point 'b'.
            self.alpha = self.bracket["alpha"]["lower"]
            self._swap_bracket()

        alpha_mid, phi_mid = self.bracket["alpha"]["upper"], self.bracket["phi"]["upper"]
        alpha_temp = 0.0  # Initialize this to zero, but will not be used on the first iteration
        phi_new = self.bracket["phi"]["lower"]
        flag = True

        while self._iter_count < maxiter and self.bracket["phi"]["lower"] != 0.0 and phi_new != 0.0:
            if self.bracket["phi"]["upper"] != phi_mid and self.bracket["phi"]["lower"] != phi_mid:
                # inverse quadratic interpolation
                alpha_new = self._inv_quad_interp(alpha_mid, phi_mid)
            else:
                alpha_new = self.bracket["alpha"]["lower"] - self.bracket["phi"]["lower"] * (
                    (self.bracket["alpha"]["lower"] - self.bracket["alpha"]["upper"])
                    / (self.bracket["phi"]["lower"] - self.bracket["phi"]["upper"])
                )

            if (
                (
                    (3 * self.bracket["alpha"]["upper"] + self.bracket["alpha"]["lower"]) / 4
                    < alpha_new
                    < self.bracket["alpha"]["lower"]
                )
                or (
                    flag
                    and abs(alpha_new - self.bracket["alpha"]["lower"])
                    >= abs(self.bracket["alpha"]["lower"] - alpha_mid) / 2
                )
                or (not flag and abs(alpha_new - self.bracket["alpha"]["lower"]) >= abs(alpha_mid - alpha_temp) / 2)
                or (flag and abs(self.bracket["alpha"]["lower"] - alpha_mid) < 1e-4)
                or (not flag and abs(alpha_mid - alpha_temp) < 1e-4)
            ):
                alpha_new = (self.bracket["alpha"]["upper"] + self.bracket["alpha"]["lower"]) / 2  # bisect method
                flag = True

            else:
                flag = False

            # Evaluate the residuals
            u.add_scal_vec(alpha_new - self.alpha, du)
            self.alpha = alpha_new
            self._evaluate_residuals()
            phi_new, norm = self._linesearch_objective()
            self._iter_count += 1

            # rec.abs = norm
            # rec.rel = norm / self._norm0
            # rec.alpha = self.alpha

            self._mpi_print(self._iter_count, norm, self.alpha)

            alpha_temp = alpha_mid
            alpha_mid, phi_mid = self.bracket["alpha"]["lower"], self.bracket["phi"]["lower"]

            if np.sign(self.bracket["phi"]["upper"]) * np.sign(phi_new) < 0:
                self.bracket["alpha"]["lower"], self.bracket["phi"]["lower"] = alpha_new, phi_new
            else:
                self.bracket["alpha"]["upper"], self.bracket["phi"]["upper"] = alpha_new, phi_new

            # swap a and b
            if abs(self.bracket["phi"]["upper"]) < abs(self.bracket["phi"]["lower"]):
                self._swap_bracket()

    def _illinois(self):
        """Illinois root finding algorithm that is a variation of the
        secant method.

        Reference
        ---------
        G. Dahlquist and 8. Bjorck, Numerical Methods, Prentice-Hall,
        Englewood Cliffs, N.J., 1974.
        """
        self.SOLVER = "LS: IP ILLNS"
        options = self.options
        maxiter = options["maxiter"]
        rho = options["rho"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        # Initialize 'g' as the upper bracket step
        phi_new = self.bracket["phi"]["upper"]

        while self._iter_count < maxiter and (
            abs(phi_new) > 0.5 * abs(self._phi0)
            or abs(self.bracket["alpha"]["lower"] - self.bracket["alpha"]["upper"])
            > 0.25 * (self.bracket["alpha"]["lower"] + self.bracket["alpha"]["upper"])
        ):
            alpha_new = self.bracket["alpha"]["upper"] - self.bracket["phi"]["upper"] * (
                (self.bracket["alpha"]["upper"] - self.bracket["alpha"]["lower"])
                / (self.bracket["phi"]["upper"] - self.bracket["phi"]["lower"])
            )

            # Update the state vector using a relative step between
            # alpha and the upper bracket.
            u.add_scal_vec(alpha_new - self.alpha, du)
            self.alpha = alpha_new
            self._evaluate_residuals()
            phi_new, norm = self._linesearch_objective()
            self._iter_count += 1

            # rec.abs = norm
            # rec.rel = norm / self._norm0
            # rec.alpha = self.alpha

            if np.sign(phi_new) * np.sign(self.bracket["phi"]["upper"]) > 0:
                self.bracket["phi"]["lower"] *= rho

            else:
                self.bracket["alpha"]["lower"], self.bracket["phi"]["lower"] = (
                    self.bracket["alpha"]["upper"],
                    self.bracket["phi"]["upper"],
                )

            self.bracket["alpha"]["upper"], self.bracket["phi"]["upper"] = alpha_new, phi_new

            self._mpi_print(self._iter_count, norm, self.alpha)

    def _secant(self):
        self.SOLVER = "LS: SCNT"
        options = self.options
        maxiter = options["maxiter"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        while self._iter_count < maxiter:
            alpha_new = self.bracket["alpha"]["upper"] - self.bracket["phi"]["upper"] * (
                (self.bracket["alpha"]["upper"] - self.bracket["alpha"]["lower"])
                / (self.bracket["phi"]["upper"] - self.bracket["phi"]["lower"])
            )

            u.add_scal_vec(alpha_new - self.alpha, du)
            self.alpha = alpha_new
            self._evaluate_residuals()
            phi_new, norm = self._linesearch_objective()
            self._iter_count += 1

            # rec.abs = norm
            # rec.rel = norm / self._norm0
            # rec.alpha = self.alpha

            self._mpi_print(self._iter_count, norm, self.alpha)

            if np.sign(phi_new) * np.sign(self.bracket["phi"]["upper"]) > 0:
                self.bracket["alpha"]["upper"], self.bracket["phi"]["upper"] = alpha_new, phi_new

            else:
                self.bracket["alpha"]["lower"], self.bracket["phi"]["lower"] = alpha_new, phi_new

    def _solve(self):
        """
        Run the iterative solver.
        """
        self._iter_count = 0
        options = self.options
        method = options["root_method"]

        norm, is_bracketed = self._iter_initialize()
        if not is_bracketed:
            self._bracketing()

        if self.bracket["phi"]["lower"] > self.bracket["phi"]["upper"]:
            self._swap_bracket()

        # Only rootfind/pinpoint if a bracket exists
        if not np.sign(self.bracket["phi"]["upper"]) * np.sign(self.bracket["phi"]["lower"]) >= 0:
            if method == "illinois":
                self._illinois()
            elif method == "brent":
                self._brentq()
            elif method == "secant":
                self._secant()


def _inv_quad_interp(s_a, g_a, s_b, g_b, s_c, g_c):
    """Inverse quadratic interpolation/extrapolation
    Parameters
    ----------
    s_a : float
        Upper bracket step
    g_a : float
        Upper bracket objective function value
    s_b : float
        Lower bracket step
    g_b : float
        Lower bracket function value
    s_c : float
        Step at point 'c'
    g_c : float
        Function value at point 'c'
    Returns
    -------
    float
        Step along search direction du at point 'k' that
        approximates the root
    """
    term_1 = (s_a * g_b * g_c) / ((g_a - g_b) * (g_a - g_c))
    term_2 = (s_b * g_a * g_c) / ((g_b - g_a) * (g_b - g_c))
    term_3 = (s_c * g_a * g_b) / ((g_c - g_a) * (g_c - g_b))
    return term_1 + term_2 + term_3
