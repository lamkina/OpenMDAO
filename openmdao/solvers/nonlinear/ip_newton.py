"""Define the IPNewtonSolver class."""

import os
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import diags

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning, SolverWarning
from openmdao.solvers.linear.linear_blsq import LinearBLSQ
from openmdao.solvers.linesearch.bracketing import BracketingLS
from openmdao.solvers.linesearch.inner_product import InnerProductLS


class IPNewtonSolver(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = "NL: IP Newton"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        self.linesearch = BoundsEnforceLS()

        # Upper and lower bounds
        self._lower_bounds = None
        self._upper_bounds = None

        # Slots for d_alpha vectors
        self._d_alpha_upper = None
        self._d_alpha_lower = None

        # Penalty parameter
        self._mu_lower = None
        self._mu_upper = None

        # Finite bounds masks
        self._lower_finite_mask = None
        self._upper_finite_mask = None

        # Pseudo-Transient time step
        self._tau = None

        # Cache variable to store states for the penalty update algorithm
        self._state_cache = None
        self._du_cache = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare("solve_subsystems", types=bool, desc="Set to True to turn on sub-solvers (Hybrid Newton).")
        self.options.declare("max_sub_solves", types=int, default=10, desc="Maximum number of subsystem solves.")
        self.options.declare(
            "cs_reconverge",
            types=bool,
            default=True,
            desc="When True, when this driver solves under a complex step, nudge "
            "the Solution vector by a small amount so that it reconverges.",
        )
        self.options.declare(
            "reraise_child_analysiserror",
            types=bool,
            default=False,
            desc="When the option is true, a solver will reraise any "
            "AnalysisError that arises during subsolve; when false, it will "
            "continue solving.",
        )
        self.options.declare("beta", types=float, default=10.0, desc="Geometric penalty multiplier.")
        # TODO: Rename 'rho' to something better since it's already used in the line search
        self.options.declare("rho", types=float, default=0.5, desc="Constant penalty scaling term.")
        self.options.declare("mu", types=float, default=1e-10, lower=0.0, upper=1e6, desc="Initial penalty parameter.")
        self.options.declare(
            "tau", types=float, default=0.1, lower=0.0, upper=1e20, desc="Initial pseudo-transient time step."
        )
        self.options.declare(
            "gamma", types=float, default=2.0, lower=0.0, desc="Pseduo-transient time step geometric multiplier."
        )
        self.options.declare(
            "pseudo_transient",
            types=bool,
            default=False,
            desc="When the option is true, pseduo-transient methods are applied.",
        )
        self.options.declare(
            "interior_penalty",
            types=bool,
            default=True,
            desc="When this option is true, interior penalty methods are applied.",
        )
        self.options.declare(
            "unsteady_rhs",
            types=bool,
            default=True,
            desc="When the option is true, the unsteady formulation of the Newton linear system is active.",
        )
        self.options.declare("mu_max", default=1e6, types=float, desc="Maximum penalty parameter value.")

        self.supports["gradients"] = True
        self.supports["implicit_components"] = True

    def _set_bounds(self, system):
        # TODO: write docstring
        # TODO: add comments that explain what this is doing
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

                else:
                    self._lower_bounds = np.full(len(system._outputs), -np.inf)

                if var_upper is not None:
                    if self._upper_bounds is None:
                        self._upper_bounds = np.full(len(system._outputs), np.inf)
                    if not np.isscalar(var_upper):
                        var_upper = var_upper.ravel()
                    self._upper_bounds[start:end] = (var_upper - ref0) / (ref - ref0)

                else:
                    self._upper_bounds = np.full(len(system._outputs), np.inf)

                start = end
        
        # Otherwise, set bounds to infinity
        else:
            self._lower_bounds = np.full(len(system._outputs), -np.inf)
            self._upper_bounds = np.full(len(system._outputs), np.inf)

        self._lower_finite_mask = np.isfinite(self._lower_bounds)
        self._upper_finite_mask = np.isfinite(self._upper_bounds)

        if isinstance(self.linear_solver, LinearBLSQ):
            self.linear_solver.lower_bounds = self._lower_bounds
            self.linear_solver.upper_bounds = self._upper_bounds

        # Bounds and bound masks are in the line search base class,
        # so they can be set universally as long as the linesearch
        # solver is not None
        if self.linesearch is not None:
            self.linesearch._lower_bounds = self._lower_bounds
            self.linesearch._upper_bounds = self._upper_bounds
            self.linesearch._lower_finite_mask = self._lower_finite_mask
            self.linesearch._upper_finite_mask = self._upper_finite_mask

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
        rank = MPI.COMM_WORLD.rank if MPI is not None else 0

        self._disallow_discrete_outputs()

        if not isinstance(self.options._dict["solve_subsystems"]["val"], bool):
            msg = "{}: solve_subsystems must be set by the user."
            raise ValueError(msg.format(self.msginfo))

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)

        # Setup the bounds
        self._set_bounds(system)

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.linear_solver is not None:
            for s in self.linear_solver._assembled_jac_solver_iter():
                yield s

    def _set_solver_print(self, level=2, type_="all"):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        super()._set_solver_print(level=level, type_=type_)

        if self.linear_solver is not None and type_ != "NL":
            self.linear_solver._set_solver_print(level=level, type_=type_)

        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_)

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        self._recording_iter.push(("_run_apply", 0))

        system = self._system()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._apply_nonlinear()
        finally:
            self._recording_iter.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        bool
            Flag for indicating child linerization
        """
        return self.options["solve_subsystems"] and self._iter_count <= self.options["max_sub_solves"]

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.linear_solver is not None:
            self.linear_solver._linearize()

        if self.linesearch is not None:
            self.linesearch._linearize()

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

        # Set the intial penalty
        self._mu_lower = np.full(np.count_nonzero(self._lower_finite_mask), self.options["mu"])
        self._mu_upper = np.full(np.count_nonzero(self._upper_finite_mask), self.options["mu"])

        # Set the penalty in the line search if the unsteady formulation is being used.
        if self.linesearch is not None:
            if self.options["interior_penalty"]:
                self.linesearch._mu_upper = self._mu_upper
                self.linesearch._mu_lower = self._mu_lower

        # Set the pseudo-time step
        self._tau = self.options["tau"]

        if self.options["debug_print"]:
            self._err_cache["inputs"] = system._inputs._copy_views()
            self._err_cache["outputs"] = system._outputs._copy_views()

        # When under a complex step from higher in the hierarchy, sometimes the step is too small
        # to trigger reconvergence, so nudge the outputs slightly so that we always get at least
        # one iteration of Newton.
        if system.under_complex_step and self.options["cs_reconverge"]:
            system._outputs += np.linalg.norm(system._outputs.asarray()) * 1e-10

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        with Recording("Newton_subsolve", 0, self):
            if self.options["solve_subsystems"] and (self._iter_count <= self.options["max_sub_solves"]):

                self._solver_info.append_solver()

                # should call the subsystems solve before computing the first residual
                self._gs_iter()

                self._solver_info.pop()

        self._run_apply()
        norm = self._iter_get_norm()

        norm0 = norm if norm != 0.0 else 1.0

        return norm0, norm

    def _update_penalty(self):
        beta = self.options["beta"]
        rho = self.options["rho"]

        # Get the states and find the length of the state vector
        u = self._state_cache
        du = self._du_cache

        lb = self._lower_bounds
        ub = self._upper_bounds

        lb_mask = self._lower_finite_mask
        ub_mask = self._upper_finite_mask

        # Initialize d_alpha to zeros
        # We only want to store and calculate d_alpha for states that
        # have bounds
        d_alpha_lower = np.zeros(np.count_nonzero(lb_mask))
        d_alpha_upper = np.zeros(np.count_nonzero(ub_mask))

        # Compute d_alpha for all states with finite bounds
        t_lower = lb[lb_mask] - (u[lb_mask] + du[lb_mask])
        t_upper = (u[ub_mask] + du[ub_mask]) - ub[ub_mask]

        # d_alpha > 0 means that the state has violated a bound
        # d_alpha < 0 means that the state has not violated a bound
        d_alpha_lower = t_lower / np.abs(du[lb_mask])
        d_alpha_upper = t_upper / np.abs(du[ub_mask])

        # We want to set all values of d_alpha < 0 to 0 so that the
        # penalty logic and formula won't do anything for those terms
        d_alpha_lower = np.where(d_alpha_lower < 0, 0, d_alpha_lower)
        d_alpha_upper = np.where(d_alpha_upper < 0, 0, d_alpha_upper)

        # Only calculate penalty terms for states with finite bounds.
        # Reducing the array to finite values is built into the method
        # for computing d_alpha
        if d_alpha_lower.size > 0:
            self._mu_lower *= beta * d_alpha_lower + rho

        if d_alpha_upper.size > 0:
            self._mu_upper *= beta * d_alpha_upper + rho

    def _ip_jac_update(self, jac):
        # TODO: Write docstring
        # TODO: Need a way to add the penalty to jacobian free solvers
        system = self._system()

        # Get the states and find the length of the state vector
        u = system._outputs.asarray()
        dp_du_arr = np.zeros(len(u))

        # We want to add the penalty terms to the diagonal of the
        # Jacobian before the linear solve so that we don't need
        # a specialized linear solver.
        mtx = jac._int_mtx._matrix

        # Only add a penalty term to the diagonal if the state has a
        # finite bound. If no bounds are set, the diagonal term will
        # be the same as an unmodified Newton system.
        t_lower = u[self._lower_finite_mask] - self._lower_bounds[self._lower_finite_mask]
        t_upper = self._upper_bounds[self._upper_finite_mask] - u[self._upper_finite_mask]

        if t_lower.size > 0:
            dp_du_arr[self._lower_finite_mask] -= self._mu_lower / (t_lower + 1e-10)

        if t_upper.size > 0:
            dp_du_arr[self._upper_finite_mask] -= self._mu_upper / (t_upper + 1e-10)

        if isinstance(mtx, csc_matrix):  # Need to check for a sparse matrix
            mtx += diags(dp_du_arr, 0, format="csc")
            jac._int_mtx._matrix = mtx

        else:  # Not a sparse matrix
            mtx += np.diag(dp_du_arr)
            jac._int_mtx._matrix = mtx

    def _pt_jac_update(self, jac):
        # TODO: Write the docstring
        system = self._system()

        # Get the states and find the length of the state vector
        size_u = len(system._outputs.asarray())

        mtx = jac._int_mtx._matrix
        pt_arry = 1 / np.full(size_u, self._tau)

        if isinstance(mtx, csc_matrix):  # Need to check for a sparse matrix
            mtx += diags(pt_arry, 0, format="csc")
            jac._int_mtx._matrix = mtx

        else:  # Not a sparse matrix
            mtx += np.diag(pt_arry)
            jac._int_mtx._matrix = mtx

    def _update_pt_step(self):
        # TODO: Wrtie docstring
        # TODO: Add new PT update algorithm
        self._tau *= self.options["gamma"]

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        self._solver_info.append_subsolver()
        do_subsolve = self.options["solve_subsystems"] and (self._iter_count < self.options["max_sub_solves"])
        do_sub_ln = self.linear_solver._linearize_children()

        # Update the penalty term and pseudo-transient step if not the
        # first iteration
        if self._iter_count > 0:
            if self.options["interior_penalty"]:
                self._update_penalty()
                # Only set the penalty in the line search for the unsteady formulation.
                # Otherwise, the penalty isn't used in the line search.
                if self.linesearch is not None:
                    self.linesearch._mu_lower = self._mu_lower
                    self.linesearch._mu_upper = self._mu_upper

            if self.options["pseudo_transient"]:
                self._tau *= self.options["gamma"] * self.linesearch.alpha

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors["residual"]["linear"].set_vec(system._residuals)
        system._vectors["residual"]["linear"] *= -1.0
        my_asm_jac = self.linear_solver._assembled_jac

        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        # Add the penalty terms to the updated Jacobian
        if self.options["interior_penalty"]:
            self._ip_jac_update(my_asm_jac)

        # Add the pt terms to the updated Jacobian
        if self.options["pseudo_transient"]:
            self._pt_jac_update(my_asm_jac)

        self._linearize()

        self.linear_solver.solve("fwd")

        # We need to cache the states and newton step for the adaptive
        # penalty algorithm.
        if isinstance(self.linear_solver, LinearBLSQ):
            self._state_cache = system._outputs.asarray().copy()
            self._du_cache = self.linear_solver.unbounded_step
        else:
            self._state_cache = system._outputs.asarray().copy()
            self._du_cache = system._vectors["output"]["linear"].asarray().copy()

        if self.linesearch:
            self.linesearch._do_subsolve = do_subsolve
            self.linesearch.solve()
        else:
            system._outputs += system._vectors["output"]["linear"]

        self._solver_info.pop()

        # Hybrid newton support.
        if do_subsolve:
            with Recording("Newton_subsolve", 0, self):
                self._solver_info.append_solver()
                self._gs_iter()
                self._solver_info.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

    def _solve(self):
        """
        Run the iterative solver.
        """
        maxiter = self.options["maxiter"]
        atol = self.options["atol"]
        rtol = self.options["rtol"]
        iprint = self.options["iprint"]
        stall_limit = self.options["stall_limit"]
        stall_tol = self.options["stall_tol"]

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)

        stalled = False
        stall_count = 0
        if stall_limit > 0:
            stall_norm = norm0

        while self._iter_count < maxiter and norm > atol and norm / norm0 > rtol and not stalled:
            with Recording(type(self).__name__, self._iter_count, self) as rec:

                if stall_count == 3 and not self.linesearch.options["print_bound_enforce"]:

                    self.linesearch.options["print_bound_enforce"] = True

                    if self._system().pathname:
                        pathname = f"{self._system().pathname}."
                    else:
                        pathname = ""

                    msg = (
                        f"Your model has stalled three times and may be violating the bounds. "
                        f"In the future, turn on print_bound_enforce in your solver options "
                        f"here: \n{pathname}nonlinear_solver.linesearch.options"
                        f"['print_bound_enforce']=True. "
                        f"\nThe bound(s) being violated now are:\n"
                    )
                    issue_warning(msg, category=SolverWarning)

                    self._single_iteration()
                    self.linesearch.options["print_bound_enforce"] = False
                else:
                    self._single_iteration()

                self._iter_count += 1
                self._run_apply()
                norm = self._iter_get_norm()

                # Save the norm values in the context manager so they can also be recorded.
                rec.abs = norm
                if norm0 == 0:
                    norm0 = 1
                rec.rel = norm / norm0
                rec.mu_upper = self._mu_upper
                rec.mu_lower = self._mu_lower
                rec.tau = self._tau

                # Check if convergence is stalled.
                if stall_limit > 0:
                    rel_norm = rec.rel
                    norm_diff = np.abs(stall_norm - rel_norm)
                    if norm_diff <= stall_tol:
                        stall_count += 1
                        if stall_count >= stall_limit:
                            stalled = True
                    else:
                        stall_count = 0
                        stall_norm = rel_norm

            self._mpi_print(self._iter_count, norm, norm / norm0)

        system = self._system()

        # flag for the print statements. we only print on root if USE_PROC_FILES is not set to True
        print_flag = system.comm.rank == 0 or os.environ.get("USE_PROC_FILES")

        prefix = self._solver_info.prefix + self.SOLVER

        # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
        # conditionals.
        if np.isinf(norm) or np.isnan(norm):
            msg = "Solver '{}' on system '{}': residuals contain 'inf' or 'NaN' after {} " + "iterations."
            if iprint > -1 and print_flag:
                print(prefix + msg.format(self.SOLVER, system.pathname, self._iter_count))

            # Raise AnalysisError if requested.
            if self.options["err_on_non_converge"]:
                raise AnalysisError(msg.format(self.SOLVER, system.pathname, self._iter_count))

        # Solver hit maxiter without meeting desired tolerances.
        # Or solver stalled.
        elif (norm > atol and norm / norm0 > rtol) or stalled:

            if stalled:
                msg = "Solver '{}' on system '{}' stalled after {} iterations."
            else:
                msg = "Solver '{}' on system '{}' failed to converge in {} iterations."

            if iprint > -1 and print_flag:
                print(prefix + msg.format(self.SOLVER, system.pathname, self._iter_count))

            # Raise AnalysisError if requested.
            if self.options["err_on_non_converge"]:
                raise AnalysisError(msg.format(self.SOLVER, system.pathname, self._iter_count))

        # Solver converged
        elif iprint == 1 and print_flag:
            print(prefix + " Converged in {} iterations".format(self._iter_count))
        elif iprint == 2 and print_flag:
            print(prefix + " Converged")

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self.linear_solver is not None:
            self.linear_solver._set_complex_step_mode(active)
            if self.linear_solver._assembled_jac is not None:
                self.linear_solver._assembled_jac.set_complex_step_mode(active)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super().cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()
