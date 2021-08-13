"""
A few different backtracking line search subsolvers.

BoundsEnforceLS - Only checks bounds and enforces them by one of three methods.
ArmijoGoldsteinLS -- Like above, but terminates with the ArmijoGoldsteinLS condition.

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


class InteriorPointLS(NonlinearSolver):
    """
    Backtracking line search that terminates using the Armijo-Goldstein condition.

    Attributes
    ----------
    _analysis_error_raised : bool
        Flag is set to True if a subsystem raises an AnalysisError.
    """

    SOLVER = "LS: IntPnt"

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
        self._w_bounds = None
        self._v_bounds = None

        self._analysis_error_raised = False

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
        self._v_bounds = np.zeros(len(system._outputs))
        self._w_bounds = np.zeros(len(system._outputs))

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
                        self._lower_bounds = np.full(len(system._outputs), -1e10)
                    if not np.isscalar(var_lower):
                        var_lower = var_lower.ravel()
                    self._lower_bounds[start:end] = (var_lower - ref0) / (ref - ref0)
                else:
                    self._lower_bounds = np.full(len(system._outputs), -1e10)

                if var_upper is not None:
                    if self._upper_bounds is None:
                        self._upper_bounds = np.full(len(system._outputs), 1e10)
                    if not np.isscalar(var_upper):
                        var_upper = var_upper.ravel()
                    self._upper_bounds[start:end] = (var_upper - ref0) / (ref - ref0)
                else:
                    self._upper_bounds = np.full(len(system._outputs), 1e10)
                start = end
        else:
            self._lower_bounds = np.full(len(system._outputs), -1e10)
            self._upper_bounds = np.full(len(system._outputs), 1e10)

    def _enforce_bounds(self, du, dw, dv, w, v, alpha):
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
        lower = self._lower_bounds
        upper = self._upper_bounds
        w_bounds = self._w_bounds
        v_bounds = self._v_bounds

        if options["print_bound_enforce"]:
            _print_violations(system._outputs, lower, upper)

        _enforce_bounds_vector(system._outputs, w, v, du, dw, dv, alpha, lower, upper, w_bounds, v_bounds)

    def _iter_initialize(self, w, v, dw, dv):
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
        phi0 = self._line_search_objective(w, v)
        if phi0 == 0.0:
            phi0 = 1.0
        self._phi0 = phi0

        # Initial step length based on the input step length parameter
        u.add_scal_vec(alpha, du)
        v[:] = v + alpha * dv
        w[:] = w + alpha * dw

        self._enforce_bounds(du, dw, dv, w, v, alpha)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            phi = self._line_search_objective(w, v)

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
        opt.declare("beta", default=0.5, desc="Curvature parameter for controlling linesearch termination")
        opt.declare("retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised.")
        opt.declare(
            "print_bound_enforce",
            default=False,
            desc="Set to True to print out names and values of variables that are pulled " "back to their bounds.",
        )
        opt.declare("rho", default=0.5, lower=0.0, upper=1.0, desc="Contraction factor.")

        opt.declare("alpha", default=1.0, lower=0.0, desc="Line search step length")

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

    def _line_search_objective(self, w, v):
        """
        Calculate the objective function of the line search.

        Returns
        -------
        float
            Line search objective (residual norm).
        """
        system = self._system()
        u = system._vectors["output"]["linear"].asarray()
        resid = system._vectors["residual"]["linear"].asarray()
        X = np.diag(u)
        W = np.diag(w)
        V = np.diag(v)
        U = np.diag(self._upper_bounds)
        L = np.diag(self._lower_bounds)

        n = len(u)

        # Compute L2 norm squared terms
        L2_norm_1 = np.linalg.norm((X - L).dot(W).dot(np.ones(n).reshape((n, 1)))) ** 2
        L2_norm_2 = np.linalg.norm((U - X).dot(V).dot(np.ones(n).reshape((n, 1)))) ** 2
        L2_norm_3 = np.linalg.norm(resid - w + v) ** 2

        # Compose merit function
        phi = L2_norm_1 + L2_norm_2 + L2_norm_3

        return phi

    def _stopping_criteria(self, fval, w, v, mu):
        """
        Sufficient decrease criteria for the line search.

        The initial line search objective and the step length parameter are stored in the class
        instance.

        Parameters
        ----------
        fval : float
            Current line search objective value.

        Returns
        -------
        bool
            Stopping condition is satisfied.
        """
        beta = self.options["beta"]
        system = self._system()
        u = system._outputs.asarray()
        alpha = self.alpha

        phi_dot_z = -2 * (self._phi0 - mu * ((u - self._lower_bounds).dot(w) + (self._upper_bounds - u).dot(v)))

        return fval <= self._phi0 + alpha * beta * phi_dot_z

    def _update_step_length_parameter(self, rho):
        """
        Update the step length parameter by multiplying with the contraction factor.

        Parameters
        ----------
        rho : float
            Contraction factor
        """
        self.alpha *= rho  # update alpha

    def _solve(self, w, v, dw, dv, mu):
        """
        Run the iterative solver.
        """
        options = self.options
        maxiter = options["maxiter"]
        rho = options["rho"]

        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]  # Newton step

        self._iter_count = 0

        phi = self._iter_initialize(w, v, dw, dv)

        phi0 = self._phi0

        # Further backtracking if needed.
        while self._iter_count < maxiter and (
            not self._stopping_criteria(phi, w, v, mu) or self._analysis_error_raised
        ):

            with Recording("InteriorPointLS", self._iter_count, self) as rec:

                if self._iter_count > 0:
                    alpha_old = self.alpha
                    self._update_step_length_parameter(rho)
                    # Moving on the line search with the difference of the old and new step length.
                    u.add_scal_vec(self.alpha - alpha_old, du)
                    v = v + (self.alpha - alpha_old) * dv
                    w = w + (self.alpha - alpha_old) * dw
                cache = self._solver_info.save_cache()

                try:
                    self._single_iteration()
                    self._iter_count += 1

                    phi = self._line_search_objective(w, v)

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

    def solve(self, w, v, dw, dv, mu):
        self._solve(w, v, dw, dv, mu)


def _enforce_bounds_vector(u, w, v, du, dw, dv, alpha, lower_bounds, upper_bounds, w_bounds, v_bounds):
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

    du_arr = du.asarray()

    # Find the largest amount a bound is violated
    # where positive means a bound is violated - i.e. the required d_alpha.
    du_mask = du_arr != 0
    dw_mask = dw != 0
    dv_mask = dv != 0

    # Find the alpha change for the states
    if du_mask.any():
        abs_du_mask = np.abs(du_arr[du_mask])
        u_mask = u.asarray()[du_mask]

        # Check lower bound
        if lower_bounds is not None:
            max_d_alpha = np.amax((lower_bounds[du_mask] - u_mask) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

        # Check upper bound
        if upper_bounds is not None:
            max_d_alpha = np.amax((u_mask - upper_bounds[du_mask]) / abs_du_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

    # Find the alpha change for the w pseudo-state
    if dw_mask.any():
        abs_dw_mask = np.abs(dw[dw_mask])
        w_mask = w[dw_mask]

        if w_bounds is not None:
            max_d_alpha = np.amax((w_bounds[dw_mask] - w_mask) / abs_dw_mask)
            if max_d_alpha > d_alpha:
                d_alpha = max_d_alpha

    # Find the alpha change for the v pseudo-state
    if dv_mask.any():
        abs_dv_mask = np.abs(dv[dv_mask])
        v_mask = v[dv_mask]

        if v_bounds is not None:
            max_d_alpha = np.amax((v_bounds[dv_mask] - v_mask) / abs_dv_mask)
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
        v[:] = v - d_alpha * dv
        w[:] = w - d_alpha * dw

        # At this point, we normalize d_alpha by alpha to figure out the relative
        # amount that the du vector has to be reduced, then apply the reduction.
        du *= 1 - d_alpha / alpha
        dw[:] *= 1 - d_alpha / alpha
        dv[:] *= 1 - d_alpha / alpha
