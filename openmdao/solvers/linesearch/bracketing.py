"""
This linesearch brackets a minimum (either forward or backward)
and then uses successive parabolic interpolation to
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
        self.options.declare(
            "spi_tol",
            default=5e-2,
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
            # TODO: doesn't seem obvious to me that these are guaranteed to be initialized.
            #       Should we check that the attributes exist before this if?  -EA
            if self._mu_lower is not None and self._mu_upper is not None:
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
                    penalty[lb_mask] += np.sum(self._mu_lower * -np.log(t_lower + 1e-10))

                if t_upper.size > 0:
                    penalty[ub_mask] += np.sum(self._mu_upper * -np.log(t_upper + 1e-10))

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
        # doesn't return NaN and the bound is never actually hit. We want
        # d_alpha >= 0 instead of strictly greater because then the buffer
        # will be applied even in the case where alpha_max brings the
        # line search exactly to the bound.
        if d_alpha >= 0:
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

        # Cache the state in case it needs to be recovered later
        self._cache_best_point = self._solver_info.save_cache()

        self._mpi_print(self._iter_count, phi, self.alpha)

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
        self.SOLVER = "LS: BRKT final"
        self._mpi_print(self._iter_count, phi, self.alpha)
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

    def _bwd_bracketing(self):
        """
        Returns
        -------
        bool
            True if hits maxiter iterations, false otherwise
        """
        self.SOLVER = "LS: BRKT BWD"
        system = self._system()
        maxiter = self.options["maxiter"]
        u = system._outputs
        du = system._vectors["output"]["linear"]
        best_phi = self.bracket_high["phi"]
        best_alpha = self.bracket_high["alpha"]

        # Initialize the mid bracket phi
        u.add_scal_vec(self.bracket_mid["alpha"] - self.alpha, du)
        self.alpha = self.bracket_mid["alpha"]
        self._single_iteration()
        phi = self.bracket_mid["phi"] = self._line_search_objective()
        self._iter_count += 1

        # Cache the best step
        if self.bracket_mid["phi"] < self.bracket_high["phi"]:
            best_phi = self.bracket_mid["phi"]
            best_alpha = self.bracket_mid["alpha"]
            self._cache_best_point = self._solver_info.save_cache()

        self._mpi_print(self._iter_count, phi, self.alpha)

        # Keep forward tracking the bracket until a minimum has been bracketed
        while self.bracket_mid["phi"] > self.bracket_high["phi"] or self.bracket_mid["phi"] > self.bracket_low["phi"]:
            # If the max number of iterations has been reached, break out and return the value
            if self._iter_count >= maxiter:
                self._solver_info.restore_cache(self._cache_best_point)
                u.add_scal_vec(best_alpha - self.alpha, du)
                self.SOLVER = "LS: BRKT final"
                self._mpi_print(self._iter_count, best_phi, best_alpha)
                return True

            # Shift the brackets over and compute the alpha for the new mid
            self.bracket_high = deepcopy(self.bracket_mid)
            self.bracket_mid["alpha"] /= self.options["beta"]

            # Move the states to the new alpha
            u.add_scal_vec(self.bracket_mid["alpha"] - self.alpha, du)
            self.alpha = self.bracket_mid["alpha"]
            self._single_iteration()
            phi = self.bracket_mid["phi"] = self._line_search_objective()
            self._iter_count += 1

            # Cache the mid point if it's better than the current best
            if self.bracket_mid["phi"] < best_phi:
                best_phi = self.bracket_mid["phi"]
                best_alpha = self.bracket_mid["alpha"]
                self._cache_best_point = self._solver_info.save_cache()

            self._mpi_print(self._iter_count, phi, self.alpha)

        return False

    def _fwd_bracketing(self):
        """
        Returns
        -------
        bool
            True if bound is hit without bracketing or hits maxiter iterations, false otherwise
        """
        self.SOLVER = "LS: BRKT FWD"
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

        # Cache the best step
        if self.bracket_high["phi"] < self.bracket_mid["phi"]:
            self._cache_best_point = self._solver_info.save_cache()

        self._mpi_print(self._iter_count, phi, self.alpha)

        # Keep forward tracking the bracket until a minimum has been bracketed
        while self.bracket_mid["phi"] > self.bracket_high["phi"]:
            # Cache the best step. Since the forward tracking is continuing to look forward,
            # the previous model state is an improvement over the one before it
            self._cache_best_point = self._solver_info.save_cache()

            # If the max number of iterations has been reached or a bound has been hit (thus, not bracketed),
            # break out and return the value
            if self._iter_count >= maxiter or bound_hit:
                self.SOLVER = "LS: BRKT final"
                self._mpi_print(self._iter_count, phi, self.alpha)
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

            self._mpi_print(self._iter_count, phi, self.alpha)

        return False

    def _spi(self):
        """
        This function assumes that the step with the best phi
        so far is already cached in self._cache_best_point. If
        self.bracket_mid["alpha"] is None (bracketed at the first
        step), it assumes that the cached state is at alpha = 1.
        """
        self.SOLVER = "LS: BRKT SPI"
        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]

        maxiter = self.options["maxiter"]

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

            # If the minimum hasn't moved (the residual is very likely parabolic),
            # there's no point in reevaluating the objective function, so just break
            if x_min == y:
                break

            # Move the states to u and evaluate f(u)
            u.add_scal_vec(x_min - self.alpha, du)
            self.alpha = x_min
            self._single_iteration()
            phi = fx_min = self._line_search_objective()
            self._iter_count += 1

            # Update the bracket based on the function value at x_min
            if x < x_min < y:
                # The new phi must be less than both fx and fy to guarantee a minimum within x and y
                if fx_min <= fx and fx_min <= fy:
                    self.bracket_mid["alpha"] = x_min
                    self.bracket_mid["phi"] = fx_min
                    self.bracket_high["alpha"] = y
                    self.bracket_high["phi"] = fy

                    # The new point is better than anything so far, so cache it
                    self._cache_best_point = self._solver_info.save_cache()
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

                    # The new point is better than anything so far, so cache it
                    self._cache_best_point = self._solver_info.save_cache()
                # Otherwise there is a minimum between x and x_min
                else:
                    self.bracket_high["alpha"] = x_min
                    self.bracket_high["phi"] = fx_min
            # Somewhow the parabola minimum is outside the current bracket, so
            # restore from the best point so far. This should never happen,
            # but this provides a safeguard to handle the case if it does.
            else:
                break

            self._mpi_print(self._iter_count, phi, self.alpha)

        # Take the best point
        self._solver_info.restore_cache(self._cache_best_point)
        u.add_scal_vec(y - self.alpha, du)
        self.alpha = y
        phi = self._line_search_objective()
        self.SOLVER = "LS: BRKT final"
        self._mpi_print(self._iter_count, self.bracket_mid["phi"], self.bracket_mid["alpha"])

    def _solve(self):
        """
        Run the iterative solver.
        """
        self.SOLVER = "LS: BRKT"
        self._iter_count = 0
        brkt_dir = self._iter_initialize()

        fwd = 0
        bak = 1
        bnd = 2

        if brkt_dir == bnd:
            return

        if brkt_dir == fwd:
            # Bracket it forward and return without running SPI
            # if it hits a bound without bracketing a minimum
            if self._fwd_bracketing():
                return
        elif brkt_dir == bak:
            # If maxiters is hit, return
            if self._bwd_bracketing():
                return

        self._spi()
