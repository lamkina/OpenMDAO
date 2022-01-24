"""
A forward and back tracking adaptive line search.

AdaptiveLS -- Enforces bounds using the vector method.
"""

# TODO: Create a local options dictionary for adaptive line search specific settings
# TODO: Create a good way to switch between an unstead and steady line search formulation.
# TODO: Add the forward tracking logic and algorithm
# TODO: Make sure alpha is properly set as a class attribute so that it can be used in the Newton solver class.

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

    def _line_search_objective(self):
        """
        Calculate the objective function of the line search.

        Returns
        -------
        float
            Line search objective (residual norm).
        """
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

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options["retry_on_analysis_error"]:
                self._analysis_error_raised = True
            else:
                raise err

            phi = np.nan

        return phi

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt["maxiter"] = 5
        opt.declare(
            "c",
            default=0.1,
            lower=0.0,
            upper=1.0,
            desc="Slope parameter for line of "
            "sufficient decrease. The larger the step, the more decrease is required to "
            "terminate the line search.",
        )
        opt.declare("rho", default=0.5, lower=0.0, upper=1.0, desc="Contraction factor.")
        opt.declare("alpha", default=1.0, lower=0.0, desc="Initial line search step.")
        opt.declare("retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised.")
        opt.declare(
            "method", default="Armijo", values=["Armijo", "Goldstein"], desc="Method to calculate stopping condition."
        )

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

    def _update_step_length_parameter(self, rho):
        """
        Update the step length parameter by multiplying with the contraction factor.

        Parameters
        ----------
        rho : float
            Contraction factor
        """
        self.alpha *= rho  # update alpha

    def _solve(self):
        """
        Run the iterative solver.
        """
        options = self.options
        maxiter = options["maxiter"]
        rho = options["rho"]
        method = options["method"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]  # Newton step

        self._iter_count = 0
        phi = self._iter_initialize()
        phi0 = self._phi0

        # Further backtracking if needed.
        while self._iter_count < maxiter and (not self._stopping_criteria(phi, method) or self._analysis_error_raised):

            with Recording("ArmijoGoldsteinLS", self._iter_count, self) as rec:

                if self._iter_count > 0:
                    alpha_old = self.alpha
                    self._update_step_length_parameter(rho)
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
