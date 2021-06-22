"""
Inner Product line search

InnerProductLS - The inner product line search seeks to find the
point along the direction of the Newton Step for which the inner
product of the Newton Step vector and residuals vector equals zero.
The linesearch uses a bracketing and pinpointing algorithm to solve
for the zero of the dot product.

Reference
---------
Matthies, Hermann, and Gilbert Strang. 1979. “The Solution of Nonlinear
Finite Element Equations.” International Journal for Numerical Methods
in Engineering 14 (11): 1613–26.
"""

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.warnings import issue_warning, SolverWarning


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


class LinesearchSolver(NonlinearSolver):
    """
    Base class for line search solvers.

    Attributes
    ----------
    _do_subsolve : bool
        Flag used by parent solver to tell the line search whether to solve subsystems while
        backtracking.
    _lower_bounds : ndarray or None
        Lower bounds array.
    _upper_bounds : ndarray or None
        Upper bounds array.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            Options dictionary.
        """
        super().__init__(**kwargs)
        # Parent solver sets this to control whether to solve subsystems.
        self._do_subsolve = False
        self._lower_bounds = None
        self._upper_bounds = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt.declare(
            "bound_enforcement",
            default="scalar",
            values=["vector", "scalar", "wall"],
            desc="If this is set to 'vector', the entire vector is backtracked together "
            + "when a bound is violated. If this is set to 'scalar', only the violating "
            + "entries are set to the bound and then the backtracking occurs on the vector "
            + "as a whole. If this is set to 'wall', only the violating entries are set "
            + "to the bound, and then the backtracking follows the wall - i.e., the "
            + "violating entries do not change during the line search.",
        )
        opt.declare(
            "print_bound_enforce",
            default=False,
            desc="Set to True to print out names and values of variables that are pulled " "back to their bounds.",
        )

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)
        if system._has_bounds:
            abs2meta_out = system._var_abs2meta["output"]
            start = end = 0
            for abs_name, val in system._outputs._abs_item_iter():
                end += val.size
                meta = abs2meta_out[abs_name]
                var_lower = meta["lower"]
                var_upper = meta["upper"]

                if var_lower is None and var_upper is None:
                    start = end
                    continue

                ref0 = meta["ref0"]
                ref = meta["ref"]

                if not np.isscalar(ref0):
                    ref0 = ref0.ravel()
                if not np.isscalar(ref):
                    ref = ref.ravel()

                if var_lower is not None:
                    if self._lower_bounds is None:
                        self._lower_bounds = np.full(len(system._outputs), -np.inf)
                    if not np.isscalar(var_lower):
                        var_lower = var_lower.ravel()
                    self._lower_bounds[start:end] = (var_lower - ref0) / (ref - ref0)

                if var_upper is not None:
                    if self._upper_bounds is None:
                        self._upper_bounds = np.full(len(system._outputs), np.inf)
                    if not np.isscalar(var_upper):
                        var_upper = var_upper.ravel()
                    self._upper_bounds[start:end] = (var_upper - ref0) / (ref - ref0)

                start = end
        else:
            self._lower_bounds = self._upper_bounds = None

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

    def _iter_initialize(self, u, du):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        self.SOLVER = "LS: IP BRKT"
        system = self._system()
        self.s_b = 0.0
        self.s_a = self.options["s_a"]

        self._run_apply()

        # Store the initial residual norm for recording
        self._phi0 = self._iter_get_norm()

        # The lower bracket value is the inner product of the newton
        # step and the residuals vector
        self.g_0 = self.g_b = np.inner(du.asarray(), system._residuals.asarray())

        # Initial step length based on the upper bracket step length
        u.add_scal_vec(self.s_a, du)

        # The bounds are enforced at the upper bracket step length
        self._enforce_bounds(step=du, alpha=self.s_a)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            self.g_a = np.inner(du.asarray(), system._residuals.asarray())
            phi = self._iter_get_norm()

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options["retry_on_analysis_error"]:
                self._analysis_error_raised = True
            else:
                raise err

            phi = np.nan
            self.g_a = self.g_b * 2.0

        self._mpi_print(self._iter_count, phi, self.s_a)

        return phi

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt.declare("gamma", default=2.0, desc="Bracketing expansion factor")
        opt.declare("rho", default=0.5, desc="Root finding contraction factor")
        opt.declare("s_max", default=16.0, desc="Largest allowable bracketing step")
        opt.declare("s_a", default=1.0, desc="Initial upper bracket step")
        opt.declare("stol", default=0.5, desc="Root finding tolerance for the Illinois algorithm")
        opt.declare("maxiter_bracketing", default=10, desc="Maximum bracketing iterations")
        opt.declare("maxiter_rootfinder", default=10, desc="Maximum root finding iterations")
        opt.declare("retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised.")

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        for unused_option in ("rtol", "maxiter", "err_on_non_converge"):
            opt.undeclare(unused_option)

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

    def _bracketing(self, u, du, phi, rec):
        """Bracketing phase of the linesearch

        Parameters
        ----------
        u : <vecotor>
            State Vector
        du : <vector>
            Newton step
        phi : float
            Residual norm at current points
        rec : OpenMDAO recorder
            The recorder for this linesearch

        Returns
        -------
        tuple(float, float)
            A tuple containing the step found by bracketing and the
            inner product of the Newton step and the residual vectors
            at that step.
        """
        self.SOLVER = "LS: IP BRKT"
        system = self._system()
        options = self.options
        s_max = options["s_max"]
        gamma = options["gamma"]
        maxiter = options["maxiter_bracketing"]
        restoration = False

        while (
            self.s_a < s_max
            and self._iter_count < maxiter
            and (np.sign(self.g_a) * np.sign(self.g_b) > 0 or self._analysis_error_raised)
        ):

            if self._analysis_error_raised and self._iter_count > 0:
                restoration = True

                self.s_a *= 0.5

                s_rel = self.s_a - self.s_b

                # Move the relative step between the new step and previous step
                u.add_scal_vec(s_rel, du)

            else:
                # Move the lower bracket step and value
                self.s_b = self.s_a
                self.g_b = self.g_a

                # Update the upper bracket step
                self.s_a = gamma * self.s_a

                s_rel = self.s_a - self.s_b

                # Move the relative step between the new step and previous step
                u.add_scal_vec(s_rel, du)

                self._enforce_bounds(step=du, alpha=self.s_a)

            cache = self._solver_info.save_cache()

            try:

                self._single_iteration()

                self._iter_count += 1

                # Compute the new upper bracket value
                self.g_a = np.inner(du.asarray(), system._residuals.asarray())

                phi = self._iter_get_norm()
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

            self._mpi_print(self._iter_count, phi, self.s_a)

            # If we find a successful step in restoration mode,
            # exit the bracketing loop
            if restoration and not self._analysis_error_raised:
                break

        # Set the current step and value to the upper bracket value
        alpha = self.s_a
        g = self.g_a

        return alpha, g

    def _inv_quad_interp(self, g_c, s_c):
        s = (
            ((self.s_a * self.g_b * g_c) / ((self.g_a - self.g_b) * (self.g_a - g_c)))
            + ((self.s_b * self.g_a * g_c) / ((self.g_b - self.g_a) * (self.g_b - g_c)))
            + ((s_c * self.g_a * self.g_b) / ((g_c - self.g_a) * (g_c - self.g_b)))
        )

        return s

    def _brentq(self, alpha, g, u, du, rec):
        system = self._system()
        options = self.options
        maxiter = options["maxiter_rootfinder"]
        atol = options["atol"]

        # Exit if there is no bracket
        if np.sign(self.g_a) * np.sign(self.g_b) >= 0:
            return

        # Swap bracket
        if abs(self.g_a) < abs(self.g_b):
            self.s_a, self.s_b, self.g_a, self.g_b = self.s_b, self.s_a, self.g_b, self.g_a

        s_c, g_c = self.s_a, self.g_a
        flag = True

        while self.g_b != 0.0 and self.g_k != 0.0 and self._iter_count < maxiter and abs(self.s_b - self.s_a) > 1e-4:
            if self.g_a != g_c and self.g_b != g_c:
                s_k = self._inv_quad_interp(g_c, s_c)  # inverse quadratic interpolation
            else:
                s_k = self.s_b - self.g_b * ((self.s_b - self.s_a) / (self.g_b - self.g_a))  # secant method

            if (
                ((3 * self.s_a + self.s_b) / 4 < self.s_k < self.s_b)
                or (flag and abs(s_k - self.s_b) >= abs(self.s_b - s_c) / 2)
                or (not flag and abs(s_k - self.s_b) >= abs(s_c - s_d) / 2)
                or (flag and abs(self.s_b - s_c) < atol)
                or (not flag and abs(s_c - s_d) < atol)
            ):
                s_k = (self.s_a + self.s_b) / 2  # bisect method
                flag = True

            else:
                flag = False

            # Update the state vector using a relative step between
            # alpha and the upper bracket.
            u.add_scal_vec(s_k - self.s_a, du)

            # Evaluate the residuals
            self._single_iteration()
            self._iter_count += 1

            g_k = np.inner(du.asarray(), system._residuals.asarray())

            s_d, g_d = s_c, g_c
            s_c, g_c = self.s_b, self.g_b

            if np.sign(self.g_a) * np.sign(self.g_b) < 0:
                self.s_b, self.g_b = s_k, g_k
            else:
                self.s_a, self.g_b = s_k, g_k

            if abs(self.g_a) < abs(self.g_b):
                self.s_a, self.s_b, self.g_a, self.g_b = self.s_b, self.s_a, self.g_b, self.g_a

        return s_k

    def _illinois(self, alpha, g, u, du, rec):
        """Illinois root finding algorithm

        Parameters
        ----------
        alpha : float
            Step from the bracketing algorithm
        g : float
            Inner product at step alpha from the bracketing algorithm
        u : <vector>
            State vector
        du : <vector>
            Newton step
        rec : OpenMDAO recorder
            The recorder for the linesearch

        Reference
        ---------
        G. Dahlquist and 8. Bjorck, Numerical Methods, Prentice-Hall,
        Englewood Cliffs, N.J., 1974.
        """
        self.SOLVER = "LS: IP PNPT"
        system = self._system()
        options = self.options
        maxiter = options["maxiter_rootfinder"]
        stol = options["stol"]
        rho = options["rho"]

        while (
            np.sign(self.g_a) * np.sign(self.g_b) < 0
            and self._iter_count < maxiter
            and (abs(g) > stol * abs(self.g_0) or abs(self.s_b - self.s_a) > stol * 0.5 * (self.s_b + self.s_a))
        ):

            # Interpolate a new guess for alpha based on the bracket
            alpha = self.s_a - self.g_a * ((self.s_a - self.s_b) / (self.g_a - self.g_b))

            # Update the state vector using a relative step between
            # alpha and the upper bracket.
            u.add_scal_vec(alpha - self.s_a, du)

            # Evaluate the residuals
            self._single_iteration()
            self._iter_count += 1

            g = np.inner(du.asarray(), system._residuals.asarray())

            phi = self._iter_get_norm()
            rec.abs = phi
            rec.rel = phi / self._phi0

            if np.sign(g) * np.sign(self.g_a) > 0:
                self.g_b = rho * self.g_b

            else:
                self.s_b = self.s_a
                self.g_b = self.g_a

            self.s_a = alpha
            self.g_a = g

            self._mpi_print(self._iter_count, phi, alpha)

    def _solve(self):
        """
        Run the iterative solver.
        """
        options = self.options
        atol = options["atol"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        self._iter_count = 0

        phi = self._iter_initialize(u, du)  # Initialize the line search

        with Recording("InnerProductLS", self._iter_count, self) as rec:  # NOQA

            # Find the interval that brackets the root.  Analysis errors
            # are caught within the bracketing phase and bounds are enforced
            # at the upper step of the bracket.  If the bracketing phase
            # exits without error, we should be able to find a step length
            # within the bracket which drives the inner product to zero.
            alpha, g = self._bracketing(u, du, phi, rec)

            # The upper bracket step is the largest step we can take
            # within the bounds.  If the iter norm is less than the
            # requested absolute tolerance, we skip the root finding
            # and take the full upper bracket step.
            if not self._iter_get_norm() < atol:
                # Find the zero of the inner product between the Newton step
                # and residual vectors within the bracketing interval
                # using the Illinois algorithm.
                self._iter_count = 0
                self._illinois(alpha, g, u, du, rec)


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
        du *= d_alpha / alpha


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
    du -= change / alpha
