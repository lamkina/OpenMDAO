"""Define the ActiveSetNewtonSolver class."""


import numpy as np
import os
from scipy.sparse import csc_matrix

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.linesearch.backtracking import ActiveSetLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.mpi import MPI
from openmdao.warnings import issue_warning, SolverWarning


class NonlinearAS(NonlinearSolver):
    """
    Active Set Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = "NL: AS Newton"

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
        self._lower_bounds = None
        self._upper_bounds = None

        # Slot for linesearch
        self.linesearch = ActiveSetLS()

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

        # --- Options specific to active set Newton solver ---
        self.options.declare("gamma", default=1.0)
        self.options.declare("s", default=1.0)
        self.options.declare("delta", default=1e-4)
        self.options.declare("c", default=1.0)

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
                else:
                    self._upper_bounds = np.full(len(system._outputs), np.inf)

                start = end
        else:
            self._lower_bounds = self._upper_bounds = None

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
            self.linear_solver._linearize(self._inactive_set, self._active_set)

        if self.linesearch is not None:
            self.linesearch._linearize()

    def _objective(self):
        phi = self._iter_get_norm()
        return 0.5 * phi ** 2, phi

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

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        options = self.options
        delta = options["delta"]
        c = options["c"]
        gamma = options["gamma"]

        self._solver_info.append_subsolver()
        do_subsolve = self.options["solve_subsystems"] and (self._iter_count < self.options["max_sub_solves"])
        do_sub_ln = self.linear_solver._linearize_children()

        f0, phi0 = self._objective()
        print("=" * 100)
        print(f"f0: {f0}")

        # Set the active set tolerance and store for iteration loop
        self._delta_k = delta_k = min(delta, c * phi0 ** 0.5)

        # Get the states and residuals
        u = system._outputs.asarray()

        # The active set is the indices of states that are sufficiently close to their upper
        # or lower bounds.
        lower_mask = u - self._lower_bounds <= delta_k
        upper_mask = self._upper_bounds - u <= delta_k
        active_mask = np.logical_or(lower_mask, upper_mask)
        self._active_set = np.where(active_mask)[0]
        print(f"ACTIVE SET: {self._active_set}")

        # The inactive set is the indices not in the active set
        inactive_mask = np.logical_and(np.logical_not(lower_mask), np.logical_not(upper_mask))
        self._inactive_set = np.where(inactive_mask)[0]

        # Determine the active set search direction and modify the Newton step
        # vector accordingly.
        d_active = np.zeros(u.size)

        if d_active.size > 0:
            d_active[lower_mask] = self._lower_bounds[lower_mask] - u[lower_mask]
            d_active[upper_mask] = self._upper_bounds[upper_mask] - u[upper_mask]
            d_active = d_active[self._active_set]

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors["residual"]["linear"].set_vec(system._residuals)
        system._vectors["residual"]["linear"] *= -1.0
        my_asm_jac = self.linear_solver._assembled_jac

        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        # Set up the inactive/active set linear system
        try:
            self._linearize()

            # Solve the linear system
            self.linear_solver.solve("fwd", self._inactive_set, self._active_set, d_active)

            # Perform feasibility safegaurd (simple vector bounds enforcement)
            u = system._outputs
            du = system._vectors["output"]["linear"]
            tau1 = (self._lower_bounds[du.asarray() < 0] - u.asarray()[du.asarray() < 0]) / du.asarray()[
                du.asarray() < 0
            ]
            tau2 = (self._upper_bounds[du.asarray() > 0] - u.asarray()[du.asarray() > 0]) / du.asarray()[
                du.asarray() > 0
            ]
            if not tau1.size > 0:
                tau1 = np.inf
            if not tau2.size > 0:
                tau2 = np.inf

            tau = np.min([np.min(tau1), np.min(tau2)])

            # Move states and check sufficient decrease condition
            u.add_scal_vec(tau, du)
            if do_subsolve:
                self._gs_iter()
            self._run_apply()

            fk, _ = self._objective()

            if not fk < gamma * f0:
                # Roll back states to previous point
                u.add_scal_vec(-tau, du)
                if do_subsolve:
                    self._gs_iter()
                self._run_apply()

                # Modify the newton step to be the projected gradient of the
                # objective f(x).
                # The projected gradient is the matrix-vector product of the
                # transpose of the Jacobian and the residuals vector
                residuals = system._vectors["residual"]["linear"]
                residuals *= -1
                mtx = my_asm_jac._int_mtx._matrix

                # Check if Jacobian is sparse so we can turn it into an array
                if isinstance(mtx, csc_matrix):
                    mtx = mtx.toarray()

                proj_grad = mtx.T.dot(residuals.asarray())
                print(f"PG NO ERROR: {-proj_grad}")

                # Set the linear output vector to the negative projected
                # gradient to search along the steepest descent direction
                system._vectors["output"]["linear"].set_val(np.asarray(-proj_grad).reshape(-1))

                # Use a backtracking linesearch along the projected gradient
                # direction
                self.linesearch._do_subsolve = do_subsolve
                self.linesearch.solve()
        except RuntimeError:
            # Modify the newton step to be the projected gradient of the
            # objective f(x).
            # The projected gradient is the matrix-vector product of the
            # transpose of the Jacobian and the residuals vector
            residuals = system._vectors["residual"]["linear"]
            residuals *= -1
            mtx = my_asm_jac._int_mtx._matrix

            # Check if Jacobian is sparse so we can turn it into an array
            if isinstance(mtx, csc_matrix):
                mtx = mtx.toarray()

            proj_grad = mtx.T.dot(residuals.asarray())
            print(f"PG ERROR: {-proj_grad}")

            # Set the linear output vector to the negative projected
            # gradient to search along the steepest descent direction
            system._vectors["output"]["linear"].set_val(np.asarray(-proj_grad).reshape(-1))

            # Use a backtracking linesearch along the projected gradient
            # direction
            self.linesearch._do_subsolve = do_subsolve
            self.linesearch.solve()

        self._solver_info.pop()

        # Hybrid newton support.
        if do_subsolve:
            with Recording("Newton_subsolve", 0, self):
                self._solver_info.append_solver()
                self._gs_iter()
                self._solver_info.pop()

        print(f"STATES: {u}")
        print("=" * 100)
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
