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
from scipy.optimize import root_scalar

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
        elif method == "wall":
            _enforce_bounds_wall(system._outputs, step, alpha, lower, upper)


class InnerProductLS(LinesearchSolver):
    """
    Bracketing and pinpointing linesearch whith several root
    finding methods available.  Default is the Illinois method with
    others available through SciPy.

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
        self.s_b = 0.0
        self.s_a = self.options["s_a"]

        u = system._outputs
        du = system._vectors["output"]["linear"]

        self._run_apply()

        # The lower bracket value is the inner product of the newton
        # step and the residuals vector
        self.g_0 = self.g_b = np.inner(du, system._vectors["residual"]["linear"])

        # Initial step length based on the upper bracket step length
        u.add_scal_vec(self.s_a, du)

        # TODO: Figure out bounds enforcement

        # The bounds are enforced at the upper bracket step length
        # self._enforce_bounds(step=du, alpha=self.s_a)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            self.g_a = np.inner(du, system._vectors["residual"]["linear"])

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options["retry_on_analysis_error"]:
                self._analysis_error_raised = True
            else:
                raise err

            self.g_a = np.nan

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt["maxiter"] = 10
        opt.declare("s_max", default=16, desc="Largest allowable bracketing step")
        opt.declare("s_a", default=1.0, desc="Initial upper bracket step")
        opt.declare("stol", default=0.5, desc="Root finding tolerance for the Illinois algorithm")
        opt.declare("atol", default=1e-6, desc="Absolute error tolerance for scipy root_scalar methods")
        opt.declare("rtol", default=1e-6, desc="Relative error tolerance for scipy root_scalar methods")
        opt.declare("retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised.")
        opt.declare(
            "method",
            defualt="illinois",
            values=["illinois", "secant", "brentq", "brenth", "toms748", "ridder"],
            desc="Root finding method for the pinointing phase of the linesearch",
        )

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

    def _bracketing(self, u, du):
        system = self._system
        du = system._vectors["output"]["linear"]
        options = self.options
        s_max = options["s_max"]

        while np.sign(self.g_a) * np.sign(self.g_b) > 0 and self.s_a < s_max and not self._analysis_error_raised:

            cache = self._solver_info.save_cache()

            try:
                # Move the lower bracket step and value
                self.s_b = self.s_a
                self.g_b = self.g_a

                # Update the upper bracket step
                self.s_a = 2 * self.s_a

                # TODO: Figure out bounds enforcement

                # Enforce bounds after upper bracket expands
                # self._enforce_bounds(step=du, alpha=self.s_a)

                # Compute the new upper bracket value
                self.g_a = self._line_search_objective(self.s_a, u, du)

                # Set the current step and value to the upper bracket value
                alpha = self.s_a
                g = self.g_a

            except AnalysisError as err:
                self._solver_info.restore_cache(cache)

                if self.options["retry_on_analysis_error"]:
                    self._analysis_error_raised = True

                else:
                    raise err

        return alpha, g

    def _illinois(self, alpha, g, u, du):
        """
        Illinois root finding algorithm

        Reference
        ---------
        G. Dahlquist and 8. Bjorck, Numerical Methods, Prentice-Hall, Englewood Cliffs, N.J., 1974.
        """

        options = self.options
        maxiter = options["maxiter"]
        stol = options["stol"]

        while (
            np.sign(self.g_a) * np.sign(self.g_b) < 0
            and self._iter_count < maxiter
            and (abs(g) > stol * abs(self.g_0) or abs(self.s_b - self.s_a) > stol * 0.5 * (self.s_b + self.s_a))
        ):
            alpha = self.s_a - self.g_a * (self.s_a - self.s_b) / (self.g_a - self.g_b)

            g = self._line_search_objective(alpha, u, du)

            self._iter_count += 1

            if np.sign(g) * np.sign(self.g_a) > 0:
                self.g_b = 0.5 * self.g_b

            else:
                self.s_b = self.s_a
                self.g_b = self.g_a
            self.s_a = alpha
            self.g_a = g

        return alpha

    def _line_search_objective(self, alpha, u, du):
        # Update the state vector
        u.add_scal_vec(-alpha, du)

        # Evaluate the residuals
        self._single_iteration()

        # Return the inner product of the Newton step and the residuals
        return np.inner(du, self._system._vectors["residual"]["linear"])

    def _solve(self):
        """
        Run the iterative solver.
        """
        options = self.options
        maxiter = options["maxiter"]
        atol = options["atol"]
        rtol = options["rtol"]
        method = options["method"]

        system = self._system
        u = system._outputs
        du = system._vectors["output"]["linear"]

        self._iter_count = 0

        self._iter_initialize()  # Initialize the line search

        # TODO: Record the abs and rel residual norms
        with Recording("InnerProductLS", self._iter_count, self) as rec:

            # Find the interval that brackets the root.  Analysis errors
            # are caught within the bracketing phase and bounds are enforced
            # at the upper step of the bracket.  If the bracketing phase
            # exits without error, we should be able to find a step length
            # which drives the inner product to zero.
            alpha, g = self._bracketing(u, du)

            self._mpi_print(self._iter_count, self._iter_get_norm(), alpha)

            # Find the zero of the inner product between the Newton step
            # and residual vectors within the bracketing interval
            # using declared method.
            if method == "illinois":
                alpha = self._illinois(alpha, g, u, du)
            elif method in ["secant", "brentq", "brenth", "toms748", "ridder"]:
                res = root_scalar(
                    self._line_search_objective,
                    args=(u, du),
                    method=method,
                    bracket=[self.s_b, self.s_a],
                    x0=self.s_a,
                    x1=self.s_b,
                    xtol=atol,
                    rtol=rtol,
                    maxiter=maxiter,
                )
                alpha = res.root
                if not res.converged:
                    raise Warning(f"Root finding method {method} failed to converge in {res.iterations} iterations.")
            else:
                raise AnalysisError("Invalid root solver option declared for InnerBracketLS")

            self._mpi_print(self._iter_count, self._iter_get_norm(), alpha)


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


def _enforce_bounds_wall(u, du, alpha, lower_bounds, upper_bounds):
    """
    Enforce lower/upper bounds on each scalar separately, then backtrack along the wall.

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
    du_data = du.asarray()

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
    du_data += change / alpha

    # Now we ensure that we will backtrack along the wall during the
    # line search by setting the entries of du at the bounds to zero.
    changed_either = change.astype(bool)
    du_data[changed_either] = 0.0
