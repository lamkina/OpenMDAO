"""Define the NewtonSolver class."""


import numpy as np

from openmdao.solvers.linesearch.interior_penalty_linesearch import InteriorPenaltyLS
from openmdao.solvers.linear.linear_interior_penalty import LinearInteriorPenalty
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
        # TODO: Generalize linear solver
        self.linear_solver = LinearInteriorPenalty()

        # Slot for linesearch
        self.linesearch = InteriorPenaltyLS()

        self._lower_bounds = None
        self._upper_bounds = None

        # Hack for recording data
        # TODO: Implement this with the OpenMDAO recording functionality
        self._cached_outputs = None

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
        self.options.declare("sigma", lower=0.0, upper=1.0, default=0.5, desc="Penalty paramater decrease factor")
        self.options.declare("mu", lower=0.0, default=100.0, desc="Initial penalty parameter")
        self.options.declare("delta", lower=0.0, default=1.0, desc="Active set tolerance")
        self.options.declare("gamma", lower=0.0, default=0.5, desc="Active set tolerance damping factor")

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

                if var_upper is not None:
                    if self._upper_bounds is None:
                        self._upper_bounds = np.full(len(system._outputs), np.inf)
                    if not np.isscalar(var_upper):
                        var_upper = var_upper.ravel()
                    self._upper_bounds[start:end] = (var_upper - ref0) / (ref - ref0)

                start = end
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

        self._mu = self.options["mu"]
        self._delta = self.options["delta"]
        self._gamma = self.options["gamma"]

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
        system = self._system()

        if self._iter_count > 0:
            sigma = self.options["sigma"]

            self._delta *= self._gamma
            self._mu *= sigma

        # Get the states and find the length of the state vector
        u = system._outputs.asarray()
        n = len(u)

        # Compute the partial derivative of the penalty term with
        # respect to the states and turn the vector into a diagonal matrix
        dp_du_arr = np.zeros(n)

        # We only want to add penalty terms for bounds with finite values.
        lower_finite_mask = np.isfinite(self._lower_bounds)
        upper_finite_mask = np.isfinite(self._upper_bounds)

        lower_active_mask = u - self._lower_bounds <= self._delta
        upper_active_mask = self._upper_bounds - u <= self._delta

        lower_mask = np.logical_and(lower_finite_mask, lower_active_mask)
        upper_mask = np.logical_and(upper_finite_mask, upper_active_mask)

        t_0 = u[lower_mask] - self._lower_bounds[lower_mask]
        t_1 = self._upper_bounds[upper_mask] - u[upper_mask]

        # If no lower or upper bounds are set, there is a chance t_0
        # or t_1 could be empty so we need to check before calculating
        # the penalty terms.
        if t_0.size > 0:
            dp_du_arr[lower_mask] -= self._mu * (np.ones(len(t_0)) / (t_0 + 1e-10))

        if t_1.size > 0:
            dp_du_arr[upper_mask] += self._mu * (np.ones(len(t_1)) / (t_1 + 1e-10))

        # Update the penalty matrix in the linear solver
        self.linear_solver._update_penalty_mtx(dp_du_arr)

        # Set the penalty terms in the linesearch
        self.linesearch._update_penalty(self._mu)
        self.linesearch._update_masks(lower_mask, upper_mask)

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        self._update_penalty()
        system = self._system()
        self._solver_info.append_subsolver()
        do_subsolve = self.options["solve_subsystems"] and (self._iter_count < self.options["max_sub_solves"])
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors["residual"]["linear"].set_vec(system._residuals)
        print(system._vectors["residual"]["linear"])
        system._vectors["residual"]["linear"] *= -1.0
        my_asm_jac = self.linear_solver._assembled_jac

        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        self._linearize()

        self.linear_solver.solve("fwd")

        self.linesearch._do_subsolve = do_subsolve
        self.linesearch.solve()

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
