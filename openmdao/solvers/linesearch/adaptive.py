"""
A forward and back tracking adaptive line search.

AdaptiveLS -- Enforces bounds using the vector method.
"""

# TODO: Create a local options dictionary for adaptive line search specific settings
# TODO: Create a good way to switch between an unstead and steady line search formulation.
# TODO: Add the forward tracking logic and algorithm
# TODO: Make sure alpha is properly set as a class attribute so that it can be used in the Newton solver class.

import warnings

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.solver import LinesearchSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.om_warnings import issue_warning, SolverWarning


class AdaptiveLS(LinesearchSolver):
    """
    Forward and back tracking line search that terminates using the
    Armijo-Goldstein condition.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _analysis_error_raised : bool
        Flag is set to True if a subsystem raises an AnalysisError.
    """

    SOLVER = "LS: ADAPT"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self._analysis_error_raised = False
        self.mu_upper = None
        self.mu_lower = None

        self._lower_finite_mask = None
        self._upper_finite_mask = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        self.options.declare(
            "penalty_residual",
            default=False,
            types=bool,
            desc="Whether or not to use the penalty terms in the line search residual.",
        )
        self.options.declare(
            "c",
            default=0.1,
            lower=0.0,
            upper=1.0,
            desc="Slope parameter for line of "
            "sufficient decrease. The larger the step, the more decrease is required to "
            "terminate the line search.",
        )
        self.options.declare("bt_factor", default=0.5, lower=0.0, upper=1.0, desc="Back tracking contraction factor.")
        self.options.declare("ft_factor", default=2.0, lower=1.0, desc="Forward tracking multiplicative factor")
        self.options.declare("alpha", default=1.0, lower=0.0, desc="Initial line search step.")
        self.options.declare(
            "retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised."
        )
        self.options.declare(
            "method", default="Armijo", values=["Armijo", "Goldstein"], desc="Method to calculate stopping condition."
        )
        self.options.declare(
            "alpha_max", default=10.0, lower=1.0, desc="Initial max line search step length for formward tracking"
        )

    def _line_search_objective(self):
        """
        Calculate the objective function of the line search.

        Returns
        -------
        float
            Line search objective (residual norm).
        """
        # TODO: Need a better way to handle the penalty_residual check
        # to ensure the user doesn't attempt to use that option with
        # an incorrect solver type.

        # Compute the penalized residual norm
        if self.options["penalty_residual"]:
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
                else:
                    raise warnings.warn(
                        "Cannot use the penalty residual options without the interior penalty solver.  Defaulting to the unpenalized residual."
                    )
            else:
                raise warnings.warn(
                    "Cannot use the penalty residual options without the interior penalty solver.  Defaulting to the unpenalized residual."
                )

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
        self.alpha = alpha = self.options["alpha"]
        use_fwd_track = False

        u = system._outputs
        du = system._vectors["output"]["linear"]

        self._run_apply()
        phi0 = self._line_search_objective()
        if phi0 == 0.0:
            phi0 = 1.0
        self._phi0 = phi0
        # From definition of Newton's method one full step should drive the linearized residuals
        # to 0, hence the directional derivative is equal to the initial function value.
        self._dir_derivative = -phi0

        # Initial step length based on the input step length parameter
        u.add_scal_vec(alpha, du)

        self._enforce_bounds(step=du, alpha=alpha)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            phi = self._line_search_objective()
            if phi < phi0:
                use_fwd_track = True

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options["retry_on_analysis_error"]:
                self._analysis_error_raised = True
            else:
                raise err

            phi = np.nan

        return phi, use_fwd_track

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        self._analysis_error_raised = False
        system = self._system()

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

    def _stopping_criteria(self, fval, method):
        """
        Sufficient decrease criteria for the line search.

        The initial line search objective and the step length parameter are stored in the class
        instance.

        Parameters
        ----------
        fval : float
            Current line search objective value.
        method : str, optional
            Method to calculate stopping condition. Can be "Armijo" or "Goldstein".

        Returns
        -------
        bool
            Stopping condition is satisfied.
        """
        method = method.lower()
        fval0 = self._phi0
        df_dalpha = self._dir_derivative
        c1 = self.options["c"]
        alpha = self.alpha
        if method == "armijo":
            return fval <= fval0 + c1 * alpha * df_dalpha
        elif method == "goldstein":
            return fval0 + (1 - c1) * alpha * df_dalpha <= fval <= fval0 + c1 * alpha * df_dalpha

    def _update_step_length_parameter(self, bt_factor):
        """
        Update the step length parameter by multiplying with the contraction factor.

        Parameters
        ----------
        bt_factor : float
            Contraction factor
        """
        self.alpha *= bt_factor  # update alpha

    def _backtracking(self, phi0):
        options = self.options
        maxiter = options["maxiter"]
        bt_factor = options["bt_factor"]
        method = options["method"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]  # Newton step

        # Further backtracking if needed.
        while self._iter_count < maxiter and (not self._stopping_criteria(phi, method) or self._analysis_error_raised):

            with Recording("AdaptiveLS", self._iter_count, self) as rec:

                if self._iter_count > 0:
                    alpha_old = self.alpha
                    self._update_step_length_parameter(bt_factor)
                    # Moving on the line search with the difference of the old and new step length.
                    u.add_scal_vec(self.alpha - alpha_old, du)
                cache = self._solver_info.save_cache()

                try:
                    self._single_iteration()
                    self._iter_count += 1

                    phi = self._line_search_objective()

                    # Save the norm values in the context manager so they can also be recorded.
                    rec.abs = phi
                    rec.rel = phi / phi0

                except AnalysisError as err:
                    self._solver_info.restore_cache(cache)
                    self._iter_count += 1

                    if self.options["retry_on_analysis_error"]:
                        self._analysis_error_raised = True
                        rec.abs = np.nan
                        rec.rel = np.nan

                    else:
                        raise err

            # self._mpi_print(self._iter_count, norm, norm / norm0)
            self._mpi_print(self._iter_count, phi, self.alpha)

    def _quad_interp_min(self, x, f):
        x_min = 0.5 * (x[0] + x[2]) + 0.5 * ((f[0] - f[1]) * (f[1] - f[2]) * (f[3] - f[0])) / (
            (x[1] - x[2]) * f[0] + (x[2] - x[0]) * f[1] + (x[0] - x[1]) * f[0]
        )
        return x_min

    def _forwardtracking(self, phi):
        # TODO: Add print statements after each model evaultion
        # TODO: Clean up and refactor this function
        # TODO: Add ability to run more interpolation iterations to better converge on the optimal alpha
        options = self.options
        ft_factor = options["ft_factor"]
        alpha_max = options["alpha_max"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]  # Newton step

        # We want to store the last three alpha and phi values to do
        # a quadratic interpolation for the best alpha value
        alpha_hist = []
        phi_hist = []

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
            abs_du_mask = np.abs(du[mask])
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

        # TODO: Add bracketing algorithm
        # TODO: Add interpolation logic

    def _solve(self):
        """
        Run the iterative solver.
        """

        self._iter_count = 0
        phi, use_fwd_track = self._iter_initialize()
        phi0 = self._phi0

        if use_fwd_track:
            self._forwardtracking(phi)
        else:
            self._backtracking(phi0)


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
