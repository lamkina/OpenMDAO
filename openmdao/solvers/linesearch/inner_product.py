"""
A bracketing and pinpointing linesearch that uses the inner product
between the residual and the Newton step vector.

InnerProductLS -- Enforces bounds using the vector method.
"""

import warnings

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.solver import LinesearchSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.om_warnings import issue_warning, SolverWarning


def _print_violations(outputs, lower, upper):
    """
    Print out which variables exceed their bounds.

    Parameters
    ----------
    outputs : <Vector>
        Vector containing the outputs.
    lower : <Vector>
        Vector containing the lower bounds.
    upper : <Vector>
        Vector containing the upper bounds.
    """
    start = end = 0
    for name, val in outputs._abs_item_iter():
        end += val.size
        if upper is not None and any(val > upper[start:end]):
            msg = f"'{name}' exceeds upper bounds\n  Val: {val}\n  Upper: {upper[start:end]}\n"
            issue_warning(msg, category=SolverWarning)

        if lower is not None and any(val < lower[start:end]):
            msg = f"'{name}' exceeds lower bounds\n  Val: {val}\n  Lower: {lower[start:end]}\n"
            issue_warning(msg, category=SolverWarning)

        start = end


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
        self.mu_upper = None
        self.mu_lower = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        self.options.declare(
            "penalty_residual",
            default=False,
            types=bool,
            desc="Whether or not to use the penalty terms in the line search residual.",
        )
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

    def _enforce_bounds(self, step, alpha):
        """
        Enforce lower/upper bounds.
        Modifies the vector of outputs and the step.
        Parameters
        ----------
        step : <Vector>
            Newton step; the backtracking is applied to this vector in-place.
        alpha : float
            Step size parameter.
        """
        system = self._system()
        if not system._has_bounds:
            return

        options = self.options
        method = options["bound_enforcement"]
        lower = self._lower_bounds
        upper = self._upper_bounds

        if options["print_bound_enforce"]:
            _print_violations(system._outputs, lower, upper)

        if method == "vector":
            _enforce_bounds_vector(system._outputs, step, alpha, lower, upper)
        elif method == "scalar":
            _enforce_bounds_scalar(system._outputs, step, alpha, lower, upper)

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

        self.SOLVER = "LS: IP BRKT"
        system = self._system()
        s_b = 0.0  # Lower bracket step length
        s_a = self.alpha = self.options["alpha"]  # Upper brackt step length

        u = system._outputs
        du = system._vectors["output"]["linear"]

        self._run_apply()

        # Store the initial residual norm for recording
        self.g_0, self._phi0 = self._linesearch_objective()

        # Move the states to the upper bracket point
        u.add_scal_vec(s_a, du)

        # Enforce bounds at the upper bracket step length
        self._enforce_bounds(step=du, alpha=s_a)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            g_a, phi = self._linesearch_objective()

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options["retry_on_analysis_error"]:
                self._analysis_error_raised = True
            else:
                raise err

            phi = np.nan
            # Make g_a a value that is slightly larger and has the same
            # sign as g_b so we can enter restoration mode in the
            # bracketing phase if 'retry_on_anlysis_error' is True
            g_a = self.g_0 * 1.01

        # Set initial brackets
        self.s_ab = [s_b, s_a]
        self.g_ab = [self.g_0, g_a]

        self._mpi_print(self._iter_count, phi, s_a)

        return phi

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
        residuals = system._residuals.asarray()

        if self.options["penalty_residual"]:
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

    def _bracketing(self, phi):
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
        options = self.options
        alpha_max = options["alpha_max"]
        beta = options["beta"]
        maxiter = options["maxiter"]

        u = system._outputs
        du = system._vectors["output"]["linear"]

        # First, we need to make sure alpha_max doesn't violate a bound.
        # If alpha_max violates a bound, we need to set it to a value that will cut
        # off right at the bound.
        du_arr = du.asarray()
        u_arr = u.asarray()
        u_max = u_arr + alpha_max * du_arr

        # This is the required change in step size, relative to the du vector.
        d_alpha = 0

        # Find the largest amount a bound is violated
        # where positive means a bound is violated - i.e. the required d_alpha.
        du_arr = du.asarray()
        mask = du_arr != 0
        if mask.any():
            abs_du_mask = np.abs(du_arr[mask])
            u_mask = u_max[mask]

            # Check lower bound
            if self._lower_bounds is not None:
                max_d_alpha = np.amax((self._lower_bounds[mask] - u_mask) / abs_du_mask)
                if max_d_alpha > d_alpha:
                    d_alpha = max_d_alpha

            # Check upper bound
            if self._upper_bounds is not None:
                max_d_alpha = np.amax((u_mask - self._upper_bounds[mask]) / abs_du_mask)
                if max_d_alpha > d_alpha:
                    d_alpha = max_d_alpha

        # Adjust alpha_max so that it goes right to the most restrictive
        # bound.
        if d_alpha > 0:
            alpha_max = alpha_max - d_alpha

        while (
            self.s_ab[1] * beta < alpha_max
            and self._iter_count < maxiter
            and (np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) > 0 or self._analysis_error_raised)
        ):

            with Recording("InnerProductLS", self._iter_count, self) as rec:
                # Move the lower bracket step and value
                self.s_ab[0], self.g_ab[0] = self.s_ab[1], self.g_ab[1]

                # Update the upper bracket step
                self.s_ab[1] *= beta

                # Compute the relative step between the bracket endpoints
                s_rel = self.s_ab[1] - self.s_ab[0]

                # Move the relative step between the new step and previous step
                u.add_scal_vec(s_rel, du)

                cache = self._solver_info.save_cache()

                try:

                    self._evaluate_residuals()
                    self._iter_count += 1

                    # Compute the new upper bracket value
                    self.g_ab[1], phi = self._linesearch_objective()

                    rec.abs = phi
                    rec.rel = phi / self._phi0

                except AnalysisError as err:
                    self._solver_info.restore_cache(cache)
                    self._iter_count += 1

                    if self.options["retry_on_analysis_error"]:
                        self._analysis_error_raised = True
                        rec.abs = np.nan
                        rec.rel = np.nan

                    else:
                        raise err

            self._mpi_print(self._iter_count, phi, self.s_ab[1])

    def _inv_quad_interp(self, s_c, g_c):
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
        return _inv_quad_interp(self.s_ab[1], self.g_ab[1], self.s_ab[0], self.g_ab[0], s_c, g_c)

    def _swap_bracket(self):
        """Swaps points 'a' and 'b' in the bracket"""
        self.s_ab[1], self.s_ab[0] = self.s_ab[0], self.s_ab[1]
        self.g_ab[1], self.g_ab[0] = self.g_ab[0], self.g_ab[1]

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

        # State vector is always at point 'a' after bracketing.
        uidx = 1

        # Swap a and b
        if abs(self.g_ab[0]) < abs(self.g_ab[1]):
            # State vector is always at point 'a' after bracketing.
            # If a swap occurs, we know it simply switches to point 'b'.
            uidx = 0
            self._swap_bracket()

        s_c, g_c = self.s_ab[1], self.g_ab[1]
        s_d = 0.0  # Initialize this to zero, but will not be used on the first iteration
        g_k = self.g_ab[0]
        flag = True

        while self._iter_count < maxiter and self.g_ab[0] != 0.0 and g_k != 0.0:

            with Recording("InnerProductLS", self._iter_count, self) as rec:
                if self.g_ab[1] != g_c and self.g_ab[0] != g_c:
                    # inverse quadratic interpolation
                    s_k = self._inv_quad_interp(s_c, g_c)
                else:
                    s_k = self.s_ab[0] - self.g_ab[0] * ((self.s_ab[0] - self.s_ab[1]) / (self.g_ab[0] - self.g_ab[1]))

                if (
                    ((3 * self.s_ab[1] + self.s_ab[0]) / 4 < s_k < self.s_ab[0])
                    or (flag and abs(s_k - self.s_ab[0]) >= abs(self.s_ab[0] - s_c) / 2)
                    or (not flag and abs(s_k - self.s_ab[0]) >= abs(s_c - s_d) / 2)
                    or (flag and abs(self.s_ab[0] - s_c) < 1e-4)
                    or (not flag and abs(s_c - s_d) < 1e-4)
                ):
                    s_k = (self.s_ab[1] + self.s_ab[0]) / 2  # bisect method
                    flag = True

                else:
                    flag = False

                # Evaluate the residuals
                u.add_scal_vec(s_k - self.s_ab[uidx], du)
                self._evaluate_residuals()
                g_k, phi = self._linesearch_objective()
                self._iter_count += 1

                phi = self._iter_get_norm()
                rec.abs = phi
                rec.rel = phi / self._phi0

                self._mpi_print(self._iter_count, phi, s_k)

                s_d = s_c
                s_c, g_c = self.s_ab[0], self.g_ab[0]

                if np.sign(self.g_ab[1]) * np.sign(g_k) < 0:
                    self.s_ab[0], self.g_ab[0] = s_k, g_k

                    # The state vector just moved to point 'k', so now
                    # we set the location depending on which point, 'a' or 'b'
                    # is set as point 'k'.
                    uidx = 0
                else:
                    self.s_ab[1], self.g_ab[1] = s_k, g_k
                    uidx = 1

                # swap a and b
                if abs(self.g_ab[1]) < abs(self.g_ab[0]):
                    self._swap_bracket()
                    # If the state vector was moved to point 'a' before the swap,
                    # set the location to 'b' after the swap, and vice versa.
                    # This ensures the state vector is scaled with reference to
                    # its previous position and not the other bracket point.
                    if uidx == 1:
                        uidx = 0
                    else:
                        uidx = 1

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
        g_k = self.g_ab[1]

        while self._iter_count < maxiter and (
            abs(g_k) > 0.5 * abs(self.g_0) or abs(self.s_ab[0] - self.s_ab[1]) > 0.25 * np.sum(self.s_ab)
        ):
            with Recording("InnerProductLS", self._iter_count, self) as rec:
                s_k = self.s_ab[1] - self.g_ab[1] * ((self.s_ab[1] - self.s_ab[0]) / (self.g_ab[1] - self.g_ab[0]))

                # Update the state vector using a relative step between
                # alpha and the upper bracket.
                u.add_scal_vec(s_k - self.s_ab[1], du)
                self._evaluate_residuals()
                g_k, phi = self._linesearch_objective()
                self._iter_count += 1

                rec.abs = phi
                rec.rel = phi / self._phi0

                if np.sign(g_k) * np.sign(self.g_ab[1]) > 0:
                    self.g_ab[0] *= rho

                else:
                    self.s_ab[0], self.g_ab[0] = self.s_ab[1], self.g_ab[1]

                self.s_ab[1], self.g_ab[1] = s_k, g_k

                self._mpi_print(self._iter_count, phi, s_k)

    def _secant(self):
        self.SOLVER = "LS: IP SECANT"
        options = self.options
        maxiter = options["maxiter"]
        uidx = 1

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        while self._iter_count < maxiter:
            with Recording("InnerProductLS", self._iter_count, self) as rec:
                s_k = self.s_ab[1] - self.g_ab[1] * ((self.s_ab[1] - self.s_ab[0]) / (self.g_ab[1] - self.g_ab[0]))

                u.add_scal_vec(s_k - self.s_ab[uidx], du)
                self._evaluate_residuals()
                g_k, phi = self._linesearch_objective()
                self._iter_count += 1

                rec.abs = phi
                rec.rel = phi / self._phi0

                self._mpi_print(self._iter_count, phi, s_k)

                if np.sign(g_k) * np.sign(self.g_ab[1]) > 0:
                    self.s_ab[1], self.g_ab[1] = s_k, g_k
                    uidx = 1

                else:
                    self.s_ab[0], self.g_ab[0] = s_k, g_k
                    uidx = 0

    def _solve(self):
        """
        Run the iterative solver.
        """
        self._iter_count = 0
        options = self.options
        method = options["root_method"]

        phi = self._iter_initialize()

        # Find the interval that brackets the root.  Analysis errors
        # are caught within the bracketing phase and bounds are enforced
        # at the upper step of the bracket.  If the bracketing phase
        # exits without error, we should be able to find a step length
        # within the bracket which drives the inner product to zero.
        self._bracketing(phi)

        # Only rootfind/pinpoint if a bracket exists
        if not np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) >= 0:
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


def _enforce_bounds_vector(u, du, alpha, lower_bounds, upper_bounds):
    """
    Enforce lower/upper bounds, backtracking the entire vector together.
    This method modifies both self (u) and step (du) in-place.
    Parameters
    ----------
    u :<Vector>
        Output vector.
    du : <Vector>
        Newton step; the backtracking is applied to this vector in-place.
    alpha : float
        step size.
    lower_bounds : ndarray
        Lower bounds array.
    upper_bounds : ndarray
        Upper bounds array.
    """
    # The assumption is that alpha * du has been added to self (i.e., u)
    # just prior to this method being called. We are currently in the
    # initialization of a line search, and we're trying to ensure that
    # the u does not violate bounds in the first iteration. If it does,
    # we modify the du vector directly.

    # This is the required change in step size, relative to the du vector.
    d_alpha = 0

    # Find the largest amount a bound is violated
    # where positive means a bound is violated - i.e. the required d_alpha.
    du_arr = du.asarray()
    mask = du_arr != 0
    if mask.any():
        abs_du_mask = np.abs(du_arr[mask])
        u_mask = u.asarray()[mask]

        # Check lower bound
        if lower_bounds is not None:
            max_d_alpha = np.amax((lower_bounds[mask] - u_mask) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

        # Check upper bound
        if upper_bounds is not None:
            max_d_alpha = np.amax((u_mask - upper_bounds[mask]) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

    if d_alpha > 0:
        # d_alpha will not be negative because it was initialized to be 0
        # and we've only done max operations.
        # d_alpha will not be greater than alpha because the assumption is that
        # the original point was valid - i.e., no bounds were violated.
        # Therefore 0 <= d_alpha <= alpha.

        # We first update u to reflect the required change to du.
        u.add_scal_vec(-d_alpha, du)

        # At this point, we normalize d_alpha by alpha to figure out the relative
        # amount that the du vector has to be reduced, then apply the reduction.
        du *= 1 - d_alpha / alpha


def _enforce_bounds_scalar(u, du, alpha, lower_bounds, upper_bounds):
    """
    Enforce lower/upper bounds on each scalar separately, then backtrack as a vector.
    This method modifies both self (u) and step (du) in-place.
    Parameters
    ----------
    u :<Vector>
        Output vector.
    du : <Vector>
        Newton step; the backtracking is applied to this vector in-place.
    alpha : float
        step size.
    lower_bounds : ndarray
        Lower bounds array.
    upper_bounds : ndarray
        Upper bounds array.
    """
    # The assumption is that alpha * step has been added to this vector
    # just prior to this method being called. We are currently in the
    # initialization of a line search, and we're trying to ensure that
    # the initial step does not violate bounds. If it does, we modify
    # the step vector directly.

    # enforce bounds on step in-place.
    u_data = u.asarray()

    # If u > lower, we're just adding zero. Otherwise, we're adding
    # the step required to get up to the lower bound.
    # For du, we normalize by alpha since du eventually gets
    # multiplied by alpha.
    change_lower = 0.0 if lower_bounds is None else np.maximum(u_data, lower_bounds) - u_data

    # If u < upper, we're just adding zero. Otherwise, we're adding
    # the step required to get down to the upper bound, but normalized
    # by alpha since du eventually gets multiplied by alpha.
    change_upper = 0.0 if upper_bounds is None else np.minimum(u_data, upper_bounds) - u_data

    change = change_lower + change_upper
    u_data += change
    du += change / alpha
