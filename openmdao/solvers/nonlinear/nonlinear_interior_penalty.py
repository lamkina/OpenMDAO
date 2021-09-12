"""Define the NewtonSolver class."""


import numpy as np
from scipy.sparse import csc_matrix

from openmdao.solvers.linesearch.interior_penalty_linesearch import InteriorPenaltyLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI


class NonlinearIntPen(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = "NL: IntPen"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        self.linesearch = InteriorPenaltyLS()

        # Slots for upper and lower bounds
        self._lower_bounds = None
        self._upper_bounds = None

        # Slots for d_alpha vectors
        self._d_alpha_upper = None
        self._d_alpha_lower = None

        # Penalty parameter
        self._mu_lower = None
        self._mu_upper = None

        # Finite masks
        self._lower_finite_mask = None
        self._upper_finite_mask = None

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
        self.options.declare("mu", lower=0.0, default=1.0, desc="Initial penalty parameter")
        self.options.declare("beta", lower=1.0, default=10.0, desc="Geometric penalty multiplier")
        self.options.declare("rho", lower=0.0, upper=1.0, default=0.5, desc="Constant penalty scaling term")
        self.options.declare("ptol", default=1e-2, desc="Penalty update tolerance")

        self.supports["gradients"] = True
        self.supports["implicit_components"] = True

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
        system = self._system()
        rank = MPI.COMM_WORLD.rank if MPI is not None else 0

        self._disallow_discrete_outputs()

        if not isinstance(self.options._dict["solve_subsystems"]["value"], bool):
            msg = "{}: solve_subsystems must be set by the user."
            raise ValueError(msg.format(self.msginfo))

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)

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
                    self._lower_bounds = np.full(len(system._outputs), np.inf)

                if var_upper is not None:
                    if self._upper_bounds is None:
                        self._upper_bounds = np.full(len(system._outputs), np.inf)
                    if not np.isscalar(var_upper):
                        var_upper = var_upper.ravel()
                    self._upper_bounds[start:end] = (var_upper - ref0) / (ref - ref0)

                else:
                    self._upper_bounds = np.full(len(system._outputs), np.inf)

                start = end

            self._lower_finite_mask = np.isfinite(self._lower_bounds)
            self._upper_finite_mask = np.isfinite(self._upper_bounds)
        else:
            raise RuntimeError("No variable bounds found.  Please use the Newton solver for unbounded models.")

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
        boolean
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

        # Set the initial penalty only for states with finite lower and
        # upper bounds
        self._mu_lower = np.full(np.count_nonzero(self._lower_finite_mask), self.options["mu"])
        self._mu_upper = np.full(np.count_nonzero(self._upper_finite_mask), self.options["mu"])

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

    def _compute_bound_violation(self):
        system = self._system()

        ptol = self.options["ptol"]

        # Get the states and find the length of the state vector
        u = system._outputs.asarray()
        du = system._vectors["output"]["linear"].asarray()

        # Initialize d_alpha to zeros
        # We only want to store and calculate d_alpha for states that
        # have bounds
        d_alpha_lower = np.zeros(np.count_nonzero(self._lower_finite_mask))
        d_alpha_upper = np.zeros(np.count_nonzero(self._upper_finite_mask))

        # Compute d_alpha for all states with finite bounds
        t_lower = self._lower_bounds[self._lower_finite_mask] - u[self._lower_finite_mask]
        t_upper = u[self._upper_finite_mask] - self._upper_bounds[self._upper_finite_mask]

        d_alpha_lower = t_lower / np.abs(du[self._lower_finite_mask])
        d_alpha_upper = t_upper / np.abs(du[self._upper_finite_mask])

        # ==============================================================
        # d_alpha > 0 means that the state has violated a bound
        # d_alpha < 0 means that the state has not violated a bound
        # ==============================================================

        # We want to set all values of d_alpha < 0 to 0 so that
        # the penalty logic and formula won't do anything for those
        # terms.
        self._d_alpha_lower = np.where(d_alpha_lower < 0, 0, d_alpha_lower)
        self._d_alpha_upper = np.where(d_alpha_upper < 0, 0, d_alpha_upper)

        # Now check if any of the d_alpha values are less than the
        # stall tolerance and return an exit code to tell the solver
        # how to proceed
        if np.any(1 - self._d_alpha_lower < ptol) or np.any(1 - self._d_alpha_upper < ptol):
            # We need to place an upper bound on mu to prevent strange
            # penalty function curvature
            if np.any(self._mu_lower > 1e6) or np.any(self._mu_upper > 1e6):
                return False
            else:
                return True
        else:
            return False

    def _update_penalty(self):
        beta = self.options["beta"]
        rho = self.options["rho"]

        # Only calculate penalty terms for states with finite bounds.
        # Reducing the array to finite values is built into the method
        # for computing d_alpha
        if self._d_alpha_lower.size > 0:
            self._mu_lower *= beta * self._d_alpha_lower + rho

        if self._d_alpha_upper.size > 0:
            self._mu_upper *= beta * self._d_alpha_upper + rho

    def _modify_jacobian(self, my_asm_jac):

        system = self._system()

        # Get the states and find the length of the state vector
        u = system._outputs.asarray()
        dp_du_arr = np.zeros(len(u))

        # We want to add the penalty terms to the diagonal of the
        # Jacobian before the linear solve so that we don't need
        # a specialized linear solver.
        mtx = my_asm_jac._int_mtx._matrix

        # Only add a penalty term to the diagonal if the state has a
        # finite bound. If no bounds are set, the diagonal term will
        # be the same as an unmodified Newton system.
        t_lower = u[self._lower_finite_mask] - self._lower_bounds[self._lower_finite_mask]
        t_upper = self._upper_bounds[self._upper_finite_mask] - u[self._upper_finite_mask]

        if t_lower.size > 0:
            dp_du_arr[self._lower_finite_mask] += -self._mu_lower / (t_lower + 1e-10)

        if t_upper.size > 0:
            dp_du_arr[self._upper_finite_mask] += self._mu_upper / (t_upper + 1e-10)

        if isinstance(mtx, csc_matrix):  # Need to check for a sparse matrix
            mtx = mtx.toarray()
            mtx += np.diag(dp_du_arr)
            my_asm_jac._int_mtx._matrix = csc_matrix(mtx)

        else:  # Not a sparse matrix
            mtx += np.diag(dp_du_arr)
            my_asm_jac._int_mtx._matrix = mtx

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        flag = True
        system = self._system()
        self._solver_info.append_subsolver()

        while flag:
            # On the first Newton iteration, we should use the user
            # set penalty value.  After that, we want to use the
            # dynamic formula to update the penalty.
            if self._iter_count > 0:
                self._update_penalty()

            do_subsolve = self.options["solve_subsystems"] and (self._iter_count < self.options["max_sub_solves"])
            do_sub_ln = self.linear_solver._linearize_children()

            # Disable local fd
            approx_status = system._owns_approx_jac
            system._owns_approx_jac = False

            system._vectors["residual"]["linear"].set_vec(system._residuals)
            system._vectors["residual"]["linear"] *= -1.0
            my_asm_jac = self.linear_solver._assembled_jac

            system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
            if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
                my_asm_jac._update(system)

            self._modify_jacobian(my_asm_jac)

            self._linearize()

            self.linear_solver.solve("fwd")

            # Move the states to the Newton update
            system._outputs += system._vectors["output"]["linear"]

            flag = self._compute_bound_violation()
            if self._iter_count == 0:
                flag = False

            # Move the states back for either another iteration of the
            # penalty update or the start of the linesearch
            system._outputs -= system._vectors["output"]["linear"]

        # Set the penalty terms in the linesearch
        self.linesearch._update_penalty(self._mu_lower, self._mu_upper)

        # Run the linesearch to find a Newton step
        self.linesearch._do_subsolve = do_subsolve
        self.linesearch.solve()

        # If we find a state update that produces a successful step,
        # we want to damp the penalty term.  To do that, we need to
        # set d_alpha = 0 for the start of the next iteration
        self._d_alpha_lower = np.zeros(np.count_nonzero(self._lower_finite_mask))
        self._d_alpha_upper = np.zeros(np.count_nonzero(self._upper_finite_mask))

        self._cached_outputs = system._outputs

        self._solver_info.pop()

        # Hybrid newton support.
        if do_subsolve:
            with Recording("Newton_subsolve", 0, self):
                self._solver_info.append_solver()
                self._gs_iter()
                self._solver_info.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

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
