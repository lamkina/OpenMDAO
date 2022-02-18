""" Test for the Bracketing Line Search"""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from numpy.testing import assert_array_equal


def create_comp(res_func, shape, lower=None, upper=None, deriv_method="cs"):
    """Create an OpenMDAO component from a residual function.

    Parameters
    ----------
    res_func : function handle
        Takes in ndarray u and returns residual vector r
    shape : tuple
        Shape of input ndarray u and residual vector of res_func
    lower : optional
        Lower bound, see OpenMDAO add_output docs for acceptable formats,
        by default None
    upper : optional
        Upper bound, see OpenMDAO add_output docs for acceptable formats,
        by default None
    deriv_method : str, optional
        Method with which to compute partial derivatives,
        by default complex step (for other options, see possible
        values for the `method` option in declare_partials)

    Returns
    -------
    OpenMDAO ImplicitComponent
        Component with output state u that computes the residual
        using res_func.
    """

    class Comp(om.ImplicitComponent):
        def setup(self):
            self.add_output("u", shape=shape, lower=lower, upper=upper)

        def setup_partials(self):
            self.declare_partials(of=["*"], wrt=["*"], method=deriv_method)

        def apply_nonlinear(self, inputs, outputs, residuals):
            residuals["u"] = res_func(outputs["u"])

    return Comp


def create_problem(res_func, shape, lower=None, upper=None, deriv_method="cs"):
    """Create OpenMDAO Problem to test the line search by using a top
       level Newton solver with maxiter of 1 and adding the BracketingLS.

    Parameters
    ----------
    res_func : function handle
        Takes in ndarray u and returns residual vector r
    shape : tuple
        Shape of input ndarray u and residual vector of res_func
    lower : optional
        Lower bound, see OpenMDAO add_output docs for acceptable formats,
        by default None
    upper : optional
        Upper bound, see OpenMDAO add_output docs for acceptable formats,
        by default None
    deriv_method : str, optional
        Method with which to compute partial derivatives,
        by default complex step (for other options, see possible
        values for the `method` option in declare_partials)
    iprint : int, optional
        iprint for the Newton solver and line search, by default 2
    ls_opts : additional options
        Additional options for BracketingLS

    Returns
    -------
    OpenMDAO Problem
        Problem (already setup) with comp as the model and a
        Newton solver with maxiter=1 and the BracketingLS
    """
    p = om.Problem()
    p.model.add_subsystem(
        "comp", create_comp(res_func, shape, lower=lower, upper=upper, deriv_method=deriv_method)(), promotes=["*"]
    )
    p.setup()
    return p


class TestIPNewtonUnboundedScalar(unittest.TestCase):
    def setUp(self):
        pass

    def test_solve_subsystems_error(self):
        """Test to ensure the correct error is thrown when the user
        forgets to set solve_subsystems
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        lin_solver = p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        ls = nl_solver.linesearch = None

        with self.assertRaises(ValueError) as context:
            p.run_model()

        msg = "IPNewtonSolver in <model> <class Group>: solve_subsystems must be set by the user."
        self.assertEqual(str(context.exception), msg)

    def test_unbounded_direct_no_linesearch(self):
        """Test to make sure the solver reaches the solution to a
        linear problem in a single iteration.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["pseudo_transient"] = False

        p.run_model()

        assert_near_equal(p.get_val("u"), 0.0)
        self.assertEqual(nl_solver._iter_count, 1)

    def test_direct_no_linesearch_with_pt(self):
        """Test that pseudo transient continuation will work without
        a line search and will take more than a single iteration.
        - Bounds: None
        - LinearSolver: Direct
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 15
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["tau"] = 0.5

        p.set_val("u", 1.5)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        self.assertNotEqual(p.get_val("u"), 1)
        assert_near_equal(p.get_val("u"), 0.0, tolerance=1e-10)

    def test_direct_be_linesearch(self):
        """Test that the solver will find the solution in a single major
        iteration when using the bounds enforce line search without bounds.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BoundsEnforceLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        self.assertEqual(p.get_val("u"), 0)

    def test_direct_ag_linesearch(self):
        """Test that the solver will find the solution in a single major
        iteration when using the bounds enforce line search without bounds.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.ArmijoGoldsteinLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        self.assertEqual(p.get_val("u"), 0)

    def test_direct_ip_linesearch(self):
        """Test that the solver will find the solution in a single major
        iteration when using the bounds enforce line search without bounds.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.InnerProductLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        self.assertEqual(p.get_val("u"), 0)

    def test_direct_brk_linesearch(self):
        """Test that the solver will find the solution in a single major
        iteration when using the bounds enforce line search without bounds.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BracketingLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        self.assertEqual(p.get_val("u"), 0)


class TestIPNewtonUnboundedVec(unittest.TestCase):
    def setUp(self):
        pass


class TestIPNewtonBoundedScalar(unittest.TestCase):
    def setUp(self):
        pass

    def test_lower_bound_blsq_no_linesearch(self):
        """Test that the solver wil find the lower bound in one iteration
        when using the BLSQ linear solver without a line search.
        - Bounds: lower=1,upper=None
        - LinearSolver: LinearBLSQ
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,), lower=1)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.LinearBLSQ(assemble_jac=True)
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 10
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 1.5)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        assert_near_equal(p.get_val("u"), 1)

    def test_upper_bound_blsq_no_linesearch(self):
        """Test that the solver will find the upper bound in one iteration
        when using the BLSQ linear solver without a line search.
        - Bounds: lower=1,upper=None
        - LinearSolver: LinearBLSQ
        - Linesearch: None
        """
        p = create_problem(lambda x: -x + 3, (1,), upper=2)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.LinearBLSQ(assemble_jac=True)
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 10
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 1.5)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        assert_near_equal(p.get_val("u"), 2)

    def test_direct_no_linesearch_no_penalty(self):
        """Test that a bounded problem with a direct linear solver and
        no line search will find the unbounded solution.  We have to
        turn off the default starting penalty so that it will converge
        in a single iteration.
        - Bounds: lower=1, upper=2
        - LinearSolver: DirectSolver
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,), lower=1, upper=2)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["pseudo_transient"] = False

        p.set_val("u", 1.5)

        p.run_model()

        self.assertEqual(nl_solver._iter_count, 1)
        self.assertNotEqual(p.get_val("u"), 1)
        assert_near_equal(p.get_val("u"), 0.0)

    def test_direct_no_linesearch_with_penalty(self):
        """Test that a bounded problem with a direct linear solver and
        no line search will find the unbounded solution.  In this test,
        we use the default penalty value which should make the solver
        take more than a single iteration to find the unbounded solution.
        - Bounds: lower=1, upper=2
        - LinearSolver: DirectSolver
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,), lower=1, upper=2)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = True

        p.set_val("u", 1.5)
        p.run_model()

        self.assertGreaterEqual(nl_solver._iter_count, 1)
        self.assertNotEqual(p.get_val("u"), 1)
        assert_near_equal(p.get_val("u"), 0.0)

    def test_set_bounds_be_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,), lower=1.0, upper=5.0)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BoundsEnforceLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        self.assertEqual(p.get_val("u"), 1.0)
        self.assertEqual(nl_solver.linesearch._lower_bounds, 1.0)
        self.assertEqual(nl_solver.linesearch._upper_bounds, 5.0)

    def test_set_bounds_ag_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: ArmijoGoldsteinLS
        """
        p = create_problem(lambda x: x, (1,), lower=1.0, upper=5.0)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.ArmijoGoldsteinLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        self.assertEqual(p.get_val("u"), 1.0)
        self.assertEqual(nl_solver.linesearch._lower_bounds, 1.0)
        self.assertEqual(nl_solver.linesearch._upper_bounds, 5.0)

    def test_set_bounds_ip_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: InnerProductLS
        """
        p = create_problem(lambda x: x, (1,), lower=1.0, upper=5.0)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.InnerProductLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        assert_near_equal(p.get_val("u"), 1.0, tolerance=1e-13)
        self.assertEqual(nl_solver.linesearch._lower_bounds, 1.0)
        self.assertEqual(nl_solver.linesearch._upper_bounds, 5.0)

    def test_set_bounds_brk_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BracketingLS
        """
        p = create_problem(lambda x: x, (1,), lower=1.0, upper=5.0)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BracketingLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", 4.0)
        p.run_model()

        assert_near_equal(p.get_val("u"), 1.0, tolerance=1e-13)
        self.assertEqual(nl_solver.linesearch._lower_bounds, 1.0)
        self.assertEqual(nl_solver.linesearch._upper_bounds, 5.0)


class TestIPNewtonBoundedVec(unittest.TestCase):
    def setUp(self):
        pass

    def test_set_bounds_be_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BoundsEnforceLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_bounds = np.array([1.0, 1.0, -np.inf, -np.inf, 1.0])
        upper_bounds = np.array([5.0, np.inf, np.inf, np.inf, 5.0])

        assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)
        assert_array_equal(nl_solver.linesearch._lower_bounds, nl_solver._lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, nl_solver._upper_bounds)

    def test_set_bounds_ag_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: ArmijoGoldsteinLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.ArmijoGoldsteinLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_bounds = np.array([1.0, 1.0, -np.inf, -np.inf, 1.0])
        upper_bounds = np.array([5.0, np.inf, np.inf, np.inf, 5.0])

        assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)
        assert_array_equal(nl_solver.linesearch._lower_bounds, nl_solver._lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, nl_solver._upper_bounds)

    def test_set_bounds_ip_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: InnerProductLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.InnerProductLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_bounds = np.array([1.0, 1.0, -np.inf, -np.inf, 1.0])
        upper_bounds = np.array([5.0, np.inf, np.inf, np.inf, 5.0])

        assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)
        assert_array_equal(nl_solver.linesearch._lower_bounds, nl_solver._lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, nl_solver._upper_bounds)

    def test_set_bounds_brk_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BracketingLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BracketingLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_bounds = np.array([1.0, 1.0, -np.inf, -np.inf, 1.0])
        upper_bounds = np.array([5.0, np.inf, np.inf, np.inf, 5.0])

        assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)
        assert_array_equal(nl_solver.linesearch._lower_bounds, nl_solver._lower_bounds)
        assert_array_equal(nl_solver.linesearch._upper_bounds, nl_solver._upper_bounds)

    def test_set_finite_mask_be_linesearch(self):
        """Test that the finite masks on the bounds are set correctly in
        the line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BoundsEnforceLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_finite_mask = np.array([True, True, False, False, True])
        upper_finite_mask = np.array([True, False, False, False, True])

        assert_array_equal(nl_solver.linesearch._upper_finite_mask, nl_solver._upper_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, nl_solver._lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._upper_finite_mask, upper_finite_mask)

    def test_set_finite_mask_ag_linesearch(self):
        """Test that the finite masks on the bounds are set correctly in
        the line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: ArmijoGoldsteinLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.ArmijoGoldsteinLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_finite_mask = np.array([True, True, False, False, True])
        upper_finite_mask = np.array([True, False, False, False, True])

        assert_array_equal(nl_solver.linesearch._upper_finite_mask, nl_solver._upper_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, nl_solver._lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._upper_finite_mask, upper_finite_mask)

    def test_set_finite_mask_ip_linesearch(self):
        """Test that the finite masks on the bounds are set correctly in
        the line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: InnerProductLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.InnerProductLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_finite_mask = np.array([True, True, False, False, True])
        upper_finite_mask = np.array([True, False, False, False, True])

        assert_array_equal(nl_solver.linesearch._upper_finite_mask, nl_solver._upper_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, nl_solver._lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._upper_finite_mask, upper_finite_mask)

    def test_set_finite_mask_brk_linesearch(self):
        """Test that the finite masks on the bounds are set correctly in
        the line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BracketingLS
        """
        p = create_problem(
            lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
        )
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = om.BracketingLS()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        p.set_val("u", np.full(5, 4.0))
        p.run_model()

        lower_finite_mask = np.array([True, True, False, False, True])
        upper_finite_mask = np.array([True, False, False, False, True])

        assert_array_equal(nl_solver.linesearch._upper_finite_mask, nl_solver._upper_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, nl_solver._lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._lower_finite_mask, lower_finite_mask)
        assert_array_equal(nl_solver.linesearch._upper_finite_mask, upper_finite_mask)


if __name__ == "__main__":
    unittest.main()
