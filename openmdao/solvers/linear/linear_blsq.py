"""Define the LinearUserDefined class."""


import scipy
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import lsq_linear
import numpy as np

from openmdao.utils.array_utils import identity_column_iter
from openmdao.solvers.solver import BoundedLinearSolver
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.matrices.dense_matrix import DenseMatrix


def index_to_varname(system, loc):
    """
    Given a matrix location, return the name of the variable associated with that index.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    loc : int
        Index of row or column.

    Returns
    -------
    str
        String containing variable absolute name (and promoted name if there is one) and index.
    """
    start = end = 0
    varsizes = np.sum(system._owned_sizes, axis=0)
    for i, name in enumerate(system._var_allprocs_abs2meta["output"]):
        end += varsizes[i]
        if loc < end:
            varname = system._var_allprocs_abs2prom["output"][name]
            break
        start = end

    if varname == name:
        name_string = "'{}' index {}.".format(varname, loc - start)
    else:
        name_string = "'{}' ('{}') index {}.".format(varname, name, loc - start)

    return name_string


def loc_to_error_msg(system, loc_txt, loc):
    """
    Given a matrix location, format a coherent error message when matrix is singular.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    loc_txt : str
        Either 'row' or 'col'.
    loc : int
        Index of row or column.

    Returns
    -------
    str
        New error string.
    """
    names = index_to_varname(system, loc)
    msg = "Singular entry found in {} for {} associated with state/residual " + names
    return msg.format(system.msginfo, loc_txt)


def format_singular_error(system, matrix):
    """
    Format a coherent error message for any ill-conditioned mmatrix.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    matrix : ndarray
        Matrix of interest.

    Returns
    -------
    str
        New error string.
    """
    if scipy.sparse.issparse(matrix):
        matrix = matrix.toarray()

    if np.any(np.isnan(matrix)):
        # There is a nan in the matrix.
        return format_nan_error(system, matrix)

    zero_rows = np.where(~matrix.any(axis=1))[0]
    zero_cols = np.where(~matrix.any(axis=0))[0]
    if zero_cols.size <= zero_rows.size:

        if zero_rows.size == 0:
            # In this case, some row is a linear combination of the other rows.

            # SVD gives us some information that may help locate the source of the problem.
            u, _, _ = np.linalg.svd(matrix)

            # Nonzero elements in the left singular vector show the rows that contribute strongly to
            # the singular subspace. Note that sometimes extra rows/cols are included in the set,
            # currently don't have a good way to pare them down.
            tol = 1e-15
            u_sing = np.abs(u[:, -1])
            left_idx = np.where(u_sing > tol)[0]

            msg = (
                "Jacobian in '{}' is not full rank. The following set of states/residuals "
                + "contains one or more equations that is a linear combination of the others: \n"
            )

            for loc in left_idx:
                name = index_to_varname(system, loc)
                msg += " " + name + "\n"

            if len(left_idx) > 2:
                msg += "Note that the problem may be in a single Component."

            return msg.format(system.pathname)

        loc_txt = "row"
        loc = zero_rows[0]
    else:
        loc_txt = "column"
        loc = zero_cols[0]

    return loc_to_error_msg(system, loc_txt, loc)


def format_nan_error(system, matrix):
    """
    Format a coherent error message when the matrix contains NaN.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    matrix : ndarray
        Matrix of interest.

    Returns
    -------
    str
        New error string.
    """
    # Because of how we built the matrix, a NaN in a comp cause the whole row to be NaN, so we
    # need to associate each index with a variable.
    varsizes = np.sum(system._owned_sizes, axis=0)

    nanrows = np.zeros(matrix.shape[0], dtype=bool)
    nanrows[np.where(np.isnan(matrix))[0]] = True

    varnames = []
    start = end = 0
    for i, name in enumerate(system._var_allprocs_abs2meta["output"]):
        end += varsizes[i]
        if np.any(nanrows[start:end]):
            varnames.append("'%s'" % system._var_allprocs_abs2prom["output"][name])
        start = end

    msg = "NaN entries found in {} for rows associated with states/residuals [{}]."
    return msg.format(system.msginfo, ", ".join(varnames))


class LinearBLSQ(DirectSolver, BoundedLinearSolver):
    """
    LinearUserDefined solver.

    This is a solver that wraps a user-written linear solve function.

    Parameters
    ----------
    solve_function : function
        Custom function containing the solve_linear function. The default is None, which means
        the name defaults to "solve_linear".
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    solve_function : function
        Custom function containing the solve_linear function. The default is None, which means
        the name defaults to "solve_linear".
    """

    SOLVER = "LN: BLSQ"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super(LinearBLSQ, self).__init__(**kwargs)
        self.unbounded_step = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            "err_on_singular", types=bool, default=True, desc="Raise an error if Jacobian is singular."
        )

        # Use an assembled jacobian by default.
        self.options["assemble_jac"] = True
    
    def _linearize(self):
        """
        This is needed to prevent the LU factorization (inherited from the DirectSolver) to
        run every iteration. Instead, it should only run within the solve with the least squares fails.
        """
        pass

    def solve(self, mode, rel_systems=None):
        """
        Solve the linear system for the problem in self._system().

        The full solution vector is returned.

        Parameters
        ----------
        mode : str
            Derivative mode, can be 'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        """
        self._rel_systems = rel_systems

        system = self._system()

        u = system._outputs.asarray()

        d_outputs = system._vectors["output"]["linear"]
        d_resids = system._vectors["residual"]["linear"]

        # Assign x and b vectors based on mode
        if mode == 'fwd':
            x_vec = d_outputs.asarray()
            b_vec = d_resids.asarray()
        else:  # rev
            x_vec = d_resids.asarray()
            b_vec = d_outputs.asarray()

        # AssembledJacobians are unscaled
        if self._assembled_jac is not None:
            mtx = self._assembled_jac._int_mtx._matrix

            # run solver
            with system._unscaled_context(outputs=[d_outputs], residuals=[d_resids]):
                opt_result = lsq_linear(
                    mtx,
                    b_vec,
                    bounds=(self.lower_bounds - u, self.upper_bounds - u),
                    method="trf",
                    lsq_solver="lsmr",
                    lsmr_max_iter=3 * min(mtx.shape),  # more iters needed for ill conditioned problems
                    lsmr_tol=1e-14,
                    verbose=2,
                )

        # matrix-vector-product generated jacobians are scaled
        else:
            mtx = self._build_mtx()
            opt_result = lsq_linear(
                mtx, b_vec, bounds=(self.lower_bounds - u, self.upper_bounds - u), method="trf", verbose=0
            )

        # If the bounded least squares solver fails to converge the least squares
        # subproblem, use a direct linear solver and scalar bounds enforcement
        lsmr_exit_flag = opt_result["unbounded_sol"][1]
        if lsmr_exit_flag > 1:
            super(LinearBLSQ, self)._linearize()
            super(LinearBLSQ, self).solve(mode)
            self.unbounded_step = x_vec

            # Perform scalar bounds enforcement by moving any violating states back within the bounds
            change_lower = 0.0 if self._lower_bounds is None else np.maximum(x_vec, self._lower_bounds) - x_vec
            change_upper = 0.0 if self._upper_bounds is None else np.minimum(x_vec, self._upper_bounds) - x_vec
            x_vec += change_lower + change_upper

        else:
            x_vec[:] = opt_result["x"]
            self.unbounded_step = opt_result["unbounded_sol"][0]
