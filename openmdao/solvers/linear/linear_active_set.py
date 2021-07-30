"""LinearSolver that uses linalg.solve or LU factor/solve."""

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse import csc_matrix

from openmdao.solvers.solver import LinearSolver
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

    nanrows = np.zeros(matrix.shape[0], dtype=np.bool)
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


class LinearActiveSet(LinearSolver):
    """
    LinearSolver that uses linalg.solve or LU factor/solve.
    """

    SOLVER = "LN: ACTIVE SET"

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            "err_on_singular", types=bool, default=True, desc="Raise an error if LU decomposition is singular."
        )

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_non_converge")

        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # Use an assembled jacobian by default.
        self.options["assemble_jac"] = True

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)
        self._disallow_distrib_solve()

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        boolean
            Flag for indicating child linearization.
        """
        return False

    def _build_mtx(self):
        """Assemble a Jacobian matrix by matrix-vector-product with columns of identity.

        Returns
        -------
        ndarray
            Jacobian matrix
        """
        system = self._system()
        bvec = system._vectors["residual"]["linear"]
        xvec = system._vectors["output"]["linear"]

        # First make a backup of the vectors
        b_data = bvec.asarray(copy=True)
        x_data = xvec.asarray(copy=True)

        nmtx = x_data.size
        seed = np.zeros(x_data.size)
        mtx = np.empty((nmtx, nmtx), dtype=b_data.dtype)
        scope_out, scope_in = system._get_scope()

        # Assemble the Jacobian by running the identity matrix through apply_linear
        for i in range(nmtx):
            # set value of x vector to provided value
            seed[i - 1] = 0.0
            seed[i] = 1.0
            xvec.set_val(seed)

            # apply linear
            system._apply_linear(self._assembled_jac, self._rel_systems, "fwd", scope_out, scope_in)

            # put new value in out_vec
            mtx[:, i] = bvec.asarray()

        # Restore the backed-up vectors
        bvec.set_val(b_data)
        xvec.set_val(x_data)

        return mtx

    def _linearize(self, inactive_set, active_set):
        """Perform factorization on the square inactive set submatrix of the Jacobian.

        Parameters
        ----------
        inactive_set : ndarray
            Indices of the states that are not close to bounds
        """
        # TODO: Remove Jacobian copies and modify Jacobian directly to save memory
        system = self._system()
        nproc = system.comm.size

        if self._assembled_jac is not None:
            matrix = self._assembled_jac._int_mtx._matrix

            # Create an array copy of the Jacobian so that we don't alter the
            # assembled Jacobian
            mtx_copy = matrix.toarray().copy()

            if matrix is None:
                # this happens if we're not rank 0 when using owned_sizes
                self._lu = self._lup = None

            # Perform dense or sparse lu factorization.
            elif isinstance(matrix, csc_matrix):
                try:
                    mtx_copy[np.ix_(active_set, inactive_set)] = 0.0
                    mtx_copy[np.ix_(active_set, active_set)] = 1.0
                    mtx_sparse = csc_matrix(mtx_copy)
                    self._lu = scipy.sparse.linalg.splu(mtx_sparse)

                except RuntimeError as err:
                    if "exactly singular" in str(err):
                        raise RuntimeError(format_singular_error(system, matrix))
                    else:
                        raise err

            elif isinstance(matrix, np.ndarray):  # dense
                # During LU decomposition, detect singularities and warn user.
                with warnings.catch_warnings():
                    if self.options["err_on_singular"]:
                        warnings.simplefilter("error", RuntimeWarning)
                    try:
                        # Build new matrix for the active set method using the Jacobian, inactive, and active sets
                        mtx_copy[np.ix_(active_set, inactive_set)] = 0.0
                        mtx_copy[np.ix_(active_set, active_set)] = 1.0
                        self._lup = scipy.linalg.lu_factor(mtx_copy)

                    except RuntimeWarning:
                        raise RuntimeError(format_singular_error(system, matrix))

                    # NaN in matrix.
                    except ValueError:
                        raise RuntimeError(format_nan_error(system, matrix))

            # Note: calling scipy.sparse.linalg.splu on a COO actually transposes
            # the matrix during conversion to csc prior to LU decomp, so we can't use COO.
            else:
                raise RuntimeError(
                    "Direct solver not implemented for matrix type %s"
                    " in %s." % (type(self._assembled_jac._int_mtx), system.msginfo)
                )
        else:
            if nproc > 1:
                raise RuntimeError(
                    "DirectSolvers without an assembled jacobian are not supported "
                    "when running under MPI if comm.size > 1."
                )

            mtx = self._build_mtx()
            mtx_copy = mtx.copy()

            # During LU decomposition, detect singularities and warn user.
            with warnings.catch_warnings():

                if self.options["err_on_singular"]:
                    warnings.simplefilter("error", RuntimeWarning)

                try:
                    mtx_copy[np.ix_(active_set, inactive_set)] = 0.0
                    mtx_copy[np.ix_(active_set, active_set)] = 1.0
                    self._lup = scipy.linalg.lu_factor(mtx_copy)

                except RuntimeWarning:
                    raise RuntimeError(format_singular_error(system, mtx))

                # NaN in matrix.
                except ValueError:
                    raise RuntimeError(format_nan_error(system, mtx))

    def solve(self, mode, inactive_set, active_set, d_active, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.
        """
        system = self._system()
        iproc = system.comm.rank
        nproc = system.comm.size

        d_residuals = system._vectors["residual"]["linear"]
        d_outputs = system._vectors["output"]["linear"]

        # assign x and b vectors based on mode
        if mode == "fwd":
            x_vec = d_outputs.asarray()
            b_vec = d_residuals.asarray()
            trans_lu = 0
            trans_splu = "N"
        else:  # rev
            x_vec = d_residuals.asarray()
            b_vec = d_outputs.asarray()
            trans_lu = 1
            trans_splu = "T"

        # Assembled Jacobians are unscaled.
        if self._assembled_jac is not None:
            # Build the right hand side of the linear system
            rhs_vec = np.zeros(len(b_vec))
            rhs_vec[active_set] = d_active
            rhs_vec[inactive_set] = b_vec[inactive_set]

            with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                if isinstance(self._assembled_jac._int_mtx, DenseMatrix):
                    arr = scipy.linalg.lu_solve(self._lup, rhs_vec, trans=trans_lu)
                else:
                    arr = self._lu.solve(rhs_vec, trans_splu)

                x_vec[:] = arr

        # matrix-vector-product generated jacobians are scaled.
        else:
            rhs_vec = np.zeros(len(b_vec))
            rhs_vec[active_set] = d_active
            rhs_vec[inactive_set] = b_vec[inactive_set]
            x_vec[:] = scipy.linalg.lu_solve(self._lup, rhs_vec, trans=trans_lu)
