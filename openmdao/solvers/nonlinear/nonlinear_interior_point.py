"""Define the NonlinearIP class."""


import numpy as np
import sys

from openmdao.solvers.linesearch.interior_point_linesearch import InteriorPointLS
from openmdao.solvers.linear.linear_interior_point import LinearIP
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.warnings import issue_warning, SolverWarning
from openmdao.core.analysis_error import AnalysisError
from openmdao.utils.mpi import MPI


class NonlinearIP(NonlinearSolver):
    """
    Interior-Point Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    """

    SOLVER = "NL: Interior-Point"

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
        self.linear_solver = LinearIP(assemble_jac=True)

        self._lower_bounds = None
        self._upper_bounds = None

        # Slot for linesearch
        self.linesearch = InteriorPointLS()

        self._w = None
        self._v = None
        self._v_output_linear = None
        self._v_output_linear = None

        self._n = None
        self._mu = 1.0

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
        self.options.declare("sigma", default=0.5, desc="Penalty parameter scalar")

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
                        self._lower_bounds = np.full(len(system._outputs), -1e10)
                    if not np.isscalar(var_lower):
                        var_lower = var_lower.ravel()
                    self._lower_bounds[start:end] = (var_lower - ref0) / (ref - ref0)
                else:
                    self._upper_bounds = np.full(len(system._outputs), -1e10)

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

        # Create diagonal matrices using states, MCP vars, and bounds
        W = np.diag(self._w)
        V = np.diag(self._v)
        U = np.diag(self._upper_bounds)
        L = np.diag(self._lower_bounds)

        if self.linear_solver is not None:
            self.linear_solver._linearize(W, V, L, U)

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

        # Initialize the w and v vectors for the mixed complementarity problem
        u = system._outputs.asarray()
        self._n = n = u.size  # Size of the state vector

        self._v = np.full(n, 1.0)
        self._w = np.full(n, 1.0)

        self._v_output_linear = np.ones(n)
        self._w_output_linear = np.ones(n)

        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _merit_function(self):
        # Get the system
        system = self._system()

        # Create diagonal matrices using states, MCP vars, and bounds
        X = np.diag(system._outputs.asarray())
        W = np.diag(self._w)
        V = np.diag(self._v)
        U = np.diag(self._upper_bounds)
        L = np.diag(self._lower_bounds)

        # Compute L2 norm squared terms
        L2_norm_1 = np.linalg.norm((X - L).dot(W).dot(np.ones(len(self._n)).reshape((self._n, 1)))) ** 2
        L2_norm_2 = np.linalg.norm((U - X).dot(V).dot(np.ones(len(self._n)).reshape((self._n, 1)))) ** 2
        L2_norm_3 = np.linalg.norm(system._residuals.asarray() - self._w + self._v) ** 2

        # Compose merit function
        phi = L2_norm_1 + L2_norm_2 + L2_norm_3

        return phi

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        sigma = self.options["sigma"]
        self._solver_info.append_subsolver()
        do_subsolve = self.options["solve_subsystems"] and (self._iter_count < self.options["max_sub_solves"])
        do_sub_ln = self.linear_solver._linearize_children()

        # Get the states
        u = system._outputs.asarray()

        # Update the penalty parameter
        self._mu = (
            sigma * ((u - self._lower_bounds).dot(self._w) + (self._upper_bounds - u).dot(self._v)) / (2 * self._n)
        )

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors["residual"]["linear"].set_vec(system._residuals)

        my_asm_jac = self.linear_solver._assembled_jac

        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        # Set up the interior-point linear system
        self._linearize()

        # Solve the linear system with an LU decomp
        W = np.diag(self._w)
        V = np.diag(self._v)
        U = np.diag(self._upper_bounds)
        L = np.diag(self._lower_bounds)

        self.linear_solver.solve(
            "fwd", W, V, L, U, self._v, self._w, self._v_output_linear, self._w_output_linear, self._mu
        )

        self.linesearch._do_subsolve = do_subsolve
        self.linesearch.solve(self._w, self._v, self._w_output_linear, self._v_output_linear, self._mu)

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
