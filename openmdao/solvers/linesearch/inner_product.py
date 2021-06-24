"""
Inner Product line search

InnerProductLS - The inner product line search seeks to find the
point along the direction of the Newton Step for which the inner
product of the Newton Step vector and residuals vector equals zero.
The linesearch uses bracketing and root finding algorithms to solve
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

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Parameters
        ----------
        u : <vector>
            State vector
        du : <vector>
            Newton Step

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """

        self.SOLVER = "LS: IP BRKT"
        system = self._system()
        s_b = 0.0
        s_a = self.options["alpha"]

        u = system._outputs
        du = system._vectors["output"]["linear"]

        self._run_apply()

        # Store the initial residual norm for recording
        self._phi0 = self._iter_get_norm()

        # The lower bracket value is the inner product of the newton
        # step and the residuals vector
        self.g_0 = g_b = np.inner(du.asarray(), system._residuals.asarray())

        # Initial step length based on the upper bracket step length
        u.add_scal_vec(s_a, du)

        # The bounds are enforced at the upper bracket step length
        self._enforce_bounds(step=du, alpha=s_a)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            g_a = np.inner(du.asarray(), system._residuals.asarray())
            phi = self._iter_get_norm()

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
            g_a = g_b * 1.01

        # Set initial brackets
        self.s_ab = [s_b, s_a]
        self.g_ab = [g_b, g_a]

        self._mpi_print(self._iter_count, phi, s_a)

        return phi

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        opt = self.options
        opt.declare(
            "beta",
            default=2.0,
            desc="Bracketing expansion factor."
            + "Beta is multiplied with the upper bracket step to expand the bracket.",
        )
        opt.declare(
            "rho",
            default=0.5,
            desc="Illinois algorithm contraction factor."
            + "This factor is useful for increasing the convergence rate"
            + "when the euclidian norm of the residual vector is very"
            + "large at one of the bracketing points.",
        )
        opt.declare(
            "alpha_max",
            default=16.0,
            desc="Largest allowable bracketing step.  The bracketing"
            + "phase will terminate if the upper bracket step goes beyond this limit.",
        )
        opt.declare("alpha", default=1.0, desc="Initial upper bracket step.")
        opt.declare("maxiter_root", default=10, desc="Maximum iterations taken by the root finding algorithm.")
        opt.declare(
            "root_method",
            default="illinois",
            values=["illinois", "brent", "ridder", "toms748"],
            desc="Name of the root finding algorithm.",
        )
        opt.declare("retry_on_analysis_error", default=True, desc="Backtrack and retry if an AnalysisError is raised.")
        opt.declare("k", default=2, values=[1, 2])

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        for unused_option in ("rtol", "maxiter", "err_on_non_converge"):
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

    def _linesearch_objective(self, alpha_new, alpha_old):
        system = self._system()
        u = system._outputs
        du = system._vectors["output"]["linear"]
        resid = system._residuals

        u.add_scal_vec(alpha_new - alpha_old, du)
        self._evaluate_residuals()
        return np.inner(du.asarray(), resid.asarray())

    def _bracketing(self, phi, rec):
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
        float
            Function value at the upper bracket step
        """
        self.SOLVER = "LS: IP BRKT"
        system = self._system()
        options = self.options
        s_max = options["alpha_max"]
        beta = options["beta"]

        u = system._outputs
        du = system._vectors["output"]["linear"]

        # Max bracketing iterations to prevent the backtracking restoration
        # search from continuing indefinitely.
        # The maximum step termination criteria will limit iterations
        # when not in restoration mode.
        maxiter = 10

        # Disable the restoration algorithm
        restoration = False

        while (
            self.s_ab[1] < s_max
            and self._iter_count < maxiter
            and (np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) > 0 or self._analysis_error_raised)
        ):

            # Enable the restoration algorithm and backtrack to recover
            # from an analysis error.
            if self._analysis_error_raised and self._iter_count > 0:
                restoration = True

                self.s_ab[1] *= 0.5

                s_rel = self.s_ab[1] - self.s_ab[0]

                # Move the relative step between the new step and previous step
                u.add_scal_vec(s_rel, du)

            else:
                # Move the lower bracket step and value
                self.s_ab[0] = self.s_ab[1]
                self.g_ab[0] = self.g_ab[0]

                # Update the upper bracket step
                self.s_ab[1] = beta * self.s_ab[1]

                s_rel = self.s_ab[1] - self.s_ab[0]

                # Move the relative step between the new step and previous step
                u.add_scal_vec(s_rel, du)

                self._enforce_bounds(step=du, alpha=self.s_ab[1])

            cache = self._solver_info.save_cache()

            try:

                self._evaluate_residuals()
                self._iter_count += 1

                # Compute the new upper bracket value
                self.g_ab[1] = np.inner(du.asarray(), system._residuals.asarray())

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

            self._mpi_print(self._iter_count, phi, self.s_ab[1])

            # If we find a successful step in restoration mode,
            # exit the bracketing loop
            if restoration and not self._analysis_error_raised:
                break

    def _inv_quad_interp(self, s_c, g_c):
        return _inv_quad_interp(self.s_ab[1], self.g_ab[1], self.s_ab[0], self.g_ab[0], s_c, g_c)

    def _swap_bracket(self):
        self.s_ab[1], self.s_ab[0] = self.s_ab[0], self.s_ab[1]
        self.g_ab[1], self.g_ab[0] = self.g_ab[0], self.g_ab[1]

    def _brentq(self, rec):
        """
        Brent's method for finding the root of a function applied to
        the inner product of the Newton step (du) and the residual
        vector (R).

        Parameters
        ----------
        u : <vector>
            State vector
        du : <vector>
            Newton Step
        rec : OpenMDAO recorder
            The recorder object for this line search solver

        Reference
        ---------
        Brent, Richard P. 1973. Algorithms for Minimization without
        Derivatives. Algorithms for Minimization without Derivatives.
        Englewood Cliffs, New Jersey: Prentice-Hall.
        """
        self.SOLVER = "LS: IP BRENT"
        options = self.options
        maxiter = options["maxiter_root"]

        # Exit if there is no bracket
        if np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) >= 0:
            return

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
            if self.g_ab[1] != g_c and self.g_ab[0] != g_c:
                # inverse quadratic interpolation
                s_k = self._inv_quad_interp(s_c, g_c)
            else:
                s_k = self.s_ab[0] - self.g_ab[0] * (
                    (self.s_ab[0] - self.s_ab[1]) / (self.g_ab[0] - self.g_ab[1])
                )  # secant method

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
            g_k = self._linesearch_objective(s_k, self.s_ab[uidx])
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

    def _illinois(self, rec):
        """Illinois root finding algorithm

        Parameters
        ----------
        g : float
            Function value at step alpha from the bracketing algorithm
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
        self.SOLVER = "LS: IP ILLNS"
        options = self.options
        maxiter = options["maxiter_root"]
        rho = options["rho"]

        # Initialize 'g' as the upper bracket step
        g_k = self.g_ab[1]

        while (
            np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) < 0
            and self._iter_count < maxiter
            and (
                abs(g_k) > 0.5 * abs(self.g_0)
                or abs(self.s_ab[0] - self.s_ab[1]) > 0.25 * (self.s_ab[0] + self.s_ab[1])
            )
        ):

            # Interpolate a new guess for alpha based on the bracket using the secant method
            s_k = self.s_ab[1] - self.g_ab[1] * ((self.s_ab[1] - self.s_ab[0]) / (self.g_ab[1] - self.g_ab[0]))

            # Update the state vector using a relative step between
            # alpha and the upper bracket.
            g_k = self._linesearch_objective(s_k, self.s_ab[1])
            self._iter_count += 1

            phi = self._iter_get_norm()
            rec.abs = phi
            rec.rel = phi / self._phi0

            if np.sign(g_k) * np.sign(self.g_ab[1]) > 0:
                self.g_ab[0] = rho * self.g_ab[0]

            else:
                self.s_ab[0] = self.s_ab[1]
                self.g_ab[0] = self.g_ab[1]

            self.s_ab[1] = s_k
            self.g_ab[1] = g_k

            self._mpi_print(self._iter_count, phi, s_k)

    def _false_position(self, s_c, g_c):
        return _false_position(self.s_ab[0], self.g_ab[1], self.g_ab[0], s_c, g_c)

    def _ridder(self, rec):
        """Ridder's algorithm for finding rootfinding adapted to work
        with the inner product of the Newton step (du) and the residual
        vector (R).

        Parameters
        ----------
        u : <vector>
            State Vector
        du : <vector>
            Newton Step
        rec : OpenMDAO recorder
            The recorder for the linesearch

        Reference
        ---------
        C. Ridders, "A new algorithm for computing a single root of a
        real continuous function," in IEEE Transactions on Circuits
        and Systems, vol. 26, no. 11, pp. 979-980, November 1979,
        doi: 10.1109/TCS.1979.1084580.
        """
        self.SOLVER = "LS: IP RIDDR"
        options = self.options
        maxiter = options["maxiter_root"]

        # Exit if there is no bracket
        if np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) >= 0:
            return

        uidx = 1

        while self._iter_count < maxiter:
            # Get a third point with a simple bisection
            s_c = (self.s_ab[1] + self.s_ab[0]) / 2

            # Evaluate g_c
            # State vector will be at either 'a' or 'b' depending on
            # which point was set as point 'k' in the prior iteraton.
            # On the first iteration, the state vector will be at the
            # upper bracket.
            g_c = self._linesearch_objective(s_c, self.s_ab[uidx])

            # Solve the quadratic function and apply the false position method
            s_k = self._false_position(s_c, g_c)

            # State vector will be at point 'c', so move the relative
            # step from point 'c' to point 'k'.
            g_k = self._linesearch_objective(s_k, s_c)
            self._iter_count += 1

            phi = self._iter_get_norm()
            rec.abs = phi
            rec.rel = phi / self._phi0

            self._mpi_print(self._iter_count, phi, s_k)

            if min(abs(s_k - self.s_ab[0]), abs(s_k - self.s_ab[1])) < 1e-12:
                return

            if abs(g_k) < 1e-6:
                return

            # Adjust the bracket to converge on the root and adjust
            # the state vector location depending on which point, 'a' or 'b',
            # is set equal to the current location 'k'.
            if np.sign(g_c) * np.sign(g_k) < 0:
                self.s_ab[0], self.g_ab[0], self.s_ab[1], self.g_ab[1] = s_c, g_c, s_k, g_k
                uidx = 1

            elif np.sign(self.g_ab[1]) * np.sign(g_k) < 0:
                self.s_ab[0], self.g_ab[0] = s_k, g_k
                uidx = 0

            else:
                self.s_ab[1], self.g_ab[1] = s_k, g_k
                uidx = 1

    def _toms748(self, rec):
        """TOMS 748 root finding algorithm that uses cubic interpolation
        and Newton-quadratic steps to find the root of the function.py

        Parameters
        ----------
        u : <vector>
            The state vector
        du : <vector>
            The Newton step
        rec : OpenMDAO recorder
            The recorder for the linesearch solver

        Reference
        ---------
        G. E. Alefeld, F. A. Potra, and Yixun Shi. 1995.
        Algorithm 748: enclosing zeros of continuous functions.
        ACM Trans. Math. Softw. 21, 3 (Sept. 1995), 327–344.
        DOI:https://doi.org/10.1145/210089.210111
        """
        self.SOLVER = "LS: IP TOMS"
        options = self.options
        self.k = options["k"]
        self._uidx = 1

        # Exit if there is no bracket
        if np.sign(self.g_ab[1]) * np.sign(self.g_ab[0]) >= 0:
            return

        # Initialize point 'c' using the secant method
        s_c = self.s_ab[0] - self.g_ab[0] * ((self.s_ab[0] - self.s_ab[1]) / (self.g_ab[0] - self.g_ab[1]))

        if not self.s_ab[0] < s_c < self.s_ab[1]:
            # Simple bisection to ensure point 'c' is within the bracket
            s_c = (self.s_ab[1] + self.s_ab[0]) / 2

        # Evaluate the residuals at point 'c'
        # The state vector is at 'a' so we move relative to 'a'
        g_c = self._linesearch_objective(s_c, self.s_ab[self._uidx])
        phi = self._iter_get_norm()

        # If g_c equals zero we found the root, exit
        if g_c == 0:
            return

        # Convert brackets to lists to match scipy implementation
        self.s_ab = [self.s_ab[0], self.s_ab[1]]
        self.g_ab = [self.g_ab[0], self.g_ab[1]]

        self.s_d, self.g_d, self._uidx = self._update_bracket(s_c, g_c)

        self.s_e, self.g_e = None, None

        # Increase the iteration count, record the residual norms, and print
        self._iter_count += 1
        rec.abs = phi
        rec.rel = phi / self._phi0

        self._mpi_print(self._iter_count, phi, s_c)

        while True:
            status = self._toms748_single_iteration(rec)
            if status:
                return

    def _toms748_single_iteration(self, rec):
        self._iter_count += 1
        eps = np.finfo(float).eps
        s_d, g_d, s_e, g_e = self.s_d, self.g_d, self.s_e, self.g_e
        s_ab_width = self.s_ab[1] - self.s_ab[0]
        s_c = None

        for nsteps in range(2, self.k + 2):
            # If the f-values are sufficiently separated, perform an inverse
            # polynomial interpolation step. Otherwise, nsteps repeats of
            # an approximate Newton-Raphson step.
            if _notclose(self.g_ab + [g_d, g_e], rtol=0, atol=32 * eps):
                s_c0 = _inverse_poly_zero(self.s_ab[0], self.s_ab[1], s_d, s_e, self.g_ab[0], self.g_ab[1], g_d, g_e)

                if self.s_ab[0] < s_c0 < self.s_ab[1]:
                    s_c = s_c0

            if s_c is None:
                s_c = _newton_quadratic(self.s_ab, self.g_ab, s_d, g_d, nsteps)

            g_c = self._linesearch_objective(s_c, self.s_ab[self._uidx])

            if g_c == 0:
                return True

            # re-bracket
            s_e, g_e = s_d, g_d
            s_d, g_d, self._uidx = self._update_bracket(s_c, g_c)

            # u is the endpoint with the smallest f-value
            uix = 0 if np.abs(self.g_ab[0]) < np.abs(self.g_ab[1]) else 1
            s_u, g_u = self.s_ab[uix], self.g_ab[uix]

            _, A = _compute_divided_differences(self.s_ab, self.g_ab, forward=(uix == 0), full=False)

            s_c = s_u - 2 * g_u / A

            if np.abs(s_c - s_u) > 0.5 * (self.s_ab[1] - self.s_ab[0]):
                s_c = sum(self.s_ab) / 2.0
            else:
                if np.isclose(s_c, s_u, rtol=eps, atol=0):
                    # c didn't change (much).
                    # Either because the f-values at the endpoints have vastly
                    # differing magnitudes, or because the root is very close to
                    # that endpoint
                    frs = np.frexp(self.g_ab)[1]
                    if frs[uix] < frs[1 - uix] - 50:  # Differ by more than 2**50
                        s_c = (31 * self.s_ab[uix] + self.s_ab[1 - uix]) / 32
                    else:
                        # Make a bigger adjustment, about the
                        # size of the requested tolerance.
                        mm = 1 if uix == 0 else -1
                        adj = mm * np.abs(s_c) * 1e-6 + mm * 1e-6
                        s_c = s_u + adj
                    if not self.s_ab[0] < s_c < self.s_ab[1]:
                        s_c = sum(self.s_ab) / 2.0

            # Move the state vector, evaluate the residuals, then calculate the
            # inner product and residual vector norm
            g_c = self._linesearch_objective(s_c, self.s_ab[self._uidx])

            if g_c == 0:
                return True

            s_e, g_e = s_d, g_d
            s_d, g_d, self._uidx = self._update_bracket(s_c, g_c)

            # If the width of the new interval did not decrease enough, bisect
            if self.s_ab[1] - self.s_ab[0] > 0.5 * s_ab_width:
                s_e, g_e = s_d, g_d
                s_z = sum(self.s_ab) / 2.0
                g_z = self._linesearch_objective(s_z, self.s_ab[self._uidx])

                if g_z == 0:
                    return True

                s_d, g_d, self._uidx = self._update_bracket(s_z, g_z)

            # Record d and e for next iteration
            self.s_d, self.g_d = s_d, g_d
            self.s_e, self.g_e = s_e, g_e

            phi = self._iter_get_norm()
            rec.abs = phi
            rec.rel = phi / self._phi0

            self._mpi_print(self._iter_count, phi, s_c)

            status = self._toms748_get_status()
            return status

    def _update_bracket(self, s_c, g_c):
        return _update_bracket(self.s_ab, self.g_ab, s_c, g_c)

    def _toms748_get_status(self):
        """Determine the current status."""
        options = self.options
        maxiter = options["maxiter_root"]
        s_a, s_b = self.s_ab[:2]
        if np.isclose(s_a, s_b, rtol=1e-6, atol=1e-6):
            return True
        if self._iter_count >= maxiter:
            return True
        return False

    def _solve(self):
        """
        Run the iterative solver.
        """
        options = self.options
        atol = options["atol"]
        method = options["root_method"]

        phi = self._iter_initialize()

        self._iter_count = 0

        with Recording("InnerProductLS", self._iter_count, self) as rec:  # NOQA

            # The upper bracket step is the largest step we can take
            # within the bounds.  If the residual norm is less than the
            # requested absolute tolerance, we take the initial step
            # without bracketing or pinpointing.
            if not self._iter_get_norm() < atol:
                # Find the interval that brackets the root.  Analysis errors
                # are caught within the bracketing phase and bounds are enforced
                # at the upper step of the bracket.  If the bracketing phase
                # exits without error, we should be able to find a step length
                # within the bracket which drives the inner product to zero.
                self._bracketing(phi, rec)

                # Find the zero of the inner product between the Newton step
                # and residual vectors within the bracketing interval
                # using the requested root finding algorithm.
                self._iter_count = 0
                if method == "illinois":
                    self._illinois(rec)
                elif method == "brent":
                    self._brentq(rec)
                elif method == "ridder":
                    self._ridder(rec)
                elif method == "toms748":
                    self._toms748(rec)


def _false_position(b, fa, fb, c, fc):
    return c + (c - b) * np.sign(fb - fa) * fc / np.sqrt(fc ** 2 - fb * fa)


def _newton_quadratic(ab, fab, d, fd, k):
    """Apply Newton-Raphson like steps, using divided differences to approximate f'
    ab is a real interval [a, b] containing a root,
    fab holds the real values of f(a), f(b)
    d is a real number outside [ab, b]
    k is the number of steps to apply
    """
    a, b = ab
    fa, fb = fab
    _, B, A = _compute_divided_differences([a, b, d], [fa, fb, fd], forward=True, full=False)

    # _P  is the quadratic polynomial through the 3 points
    def _P(x):
        # Horner evaluation of fa + B * (x - a) + A * (x - a) * (x - b)
        return (A * (x - b) + B) * (x - a) + fa

    if A == 0:
        r = a - fa / B
    else:
        r = a if np.sign(A) * np.sign(fa) > 0 else b
    # Apply k Newton-Raphson steps to _P(x), starting from x=r
    for i in range(k):
        r1 = r - _P(r) / (B + A * (2 * r - a - b))
        if not (ab[0] < r1 < ab[1]):
            if ab[0] < r < ab[1]:
                return r
            r = sum(ab) / 2.0
            break
        r = r1

    return r


def _compute_divided_differences(xvals, fvals, N=None, full=True, forward=True):
    """Return a matrix of divided differences for the xvals, fvals pairs
    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i
    If full is False, just return the main diagonal(or last row):
      f[a], f[a, b] and f[a, b, c].
    If forward is False, return f[c], f[b, c], f[a, b, c]."""
    if full:
        if forward:
            xvals = np.asarray(xvals)
        else:
            xvals = np.array(xvals)[::-1]
        M = len(xvals)
        N = M if N is None else min(N, M)
        DD = np.zeros([M, N])
        DD[:, 0] = fvals[:]
        for i in range(1, N):
            DD[i:, i] = np.diff(DD[i - 1 :, i - 1]) / (xvals[i:] - xvals[: M - i])
        return DD

    xvals = np.asarray(xvals)
    dd = np.array(fvals)
    row = np.array(fvals)
    idx2Use = 0 if forward else -1
    dd[0] = fvals[idx2Use]
    for i in range(1, len(xvals)):
        denom = xvals[i : i + len(row) - 1] - xvals[: len(row) - 1]
        row = np.diff(row)[:] / denom
        dd[i] = row[idx2Use]
    return dd


def _interpolated_poly(xvals, fvals, x):
    """Compute p(x) for the polynomial passing through the specified locations.
    Use Neville's algorithm to compute p(x) where p is the minimal degree
    polynomial passing through the points xvals, fvals"""
    xvals = np.asarray(xvals)
    N = len(xvals)
    Q = np.zeros([N, N])
    D = np.zeros([N, N])
    Q[:, 0] = fvals[:]
    D[:, 0] = fvals[:]
    for k in range(1, N):
        alpha = D[k:, k - 1] - Q[k - 1 : N - 1, k - 1]
        diffik = xvals[0 : N - k] - xvals[k:N]
        Q[k:, k] = (xvals[k:] - x) / diffik * alpha
        D[k:, k] = (xvals[: N - k] - x) / diffik * alpha
    # Expect Q[-1, 1:] to be small relative to Q[-1, 0] as x approaches a root
    return np.sum(Q[-1, 1:]) + Q[-1, 0]


def _inverse_poly_zero(a, b, c, d, fa, fb, fc, fd):
    """Inverse cubic interpolation f-values -> x-values
    Given four points (fa, a), (fb, b), (fc, c), (fd, d) with
    fa, fb, fc, fd all distinct, find poly IP(y) through the 4 points
    and compute x=IP(0).
    """
    return _interpolated_poly([fa, fb, fc, fd], [a, b, c, d], 0)


def _notclose(fs, rtol=1e-6, atol=1e-6):
    # Ensure not None, not 0, all finite, and not very close to each other
    notclosefvals = (
        all(fs)
        and all(np.isfinite(fs))
        and not any(any(np.isclose(_f, fs[i + 1 :], rtol=rtol, atol=atol)) for i, _f in enumerate(fs[:-1]))
    )
    return notclosefvals


def _update_bracket(ab, fab, c, fc):
    """Update a bracket given (c, fc), return the discarded endpoints."""
    fa, fb = fab
    idx = 0 if np.sign(fa) * np.sign(fc) > 0 else 1
    rx, rfx = ab[idx], fab[idx]
    fab[idx] = fc
    ab[idx] = c
    return rx, rfx, idx


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
