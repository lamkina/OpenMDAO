""" Test for the Bracketing Line Search"""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from numpy.testing import assert_array_equal

ls_list = [om.BoundsEnforceLS, om.ArmijoGoldsteinLS, om.InnerProductLS, om.BracketingLS]
lin_sol_list = [om.DirectSolver, om.LinearBLSQ]


class MultOutComp1(om.ImplicitComponent):
    def setup(self):
        self.add_output("u1")
        self.add_output("u2", lower=-1.0)
        self.add_output("u3", upper=-3.0)
        self.add_output("u4", lower=-2.0, upper=3.0)

    def setup_partials(self):
        self.declare_partials(of=["*"], wrt=["*"], method="cs")

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["u1"] = outputs["u1"] * outputs["u2"]
        residuals["u2"] = np.cos(outputs["u3"]) * np.exp(outputs["u4"]) - outputs["u1"]
        residuals["u3"] = -outputs["u3"]
        residuals["u4"] = 1 / outputs["u3"]


class MultOutComp2(om.ImplicitComponent):
    def setup(self):
        self.add_output("u1", lower=0.0)
        self.add_output("u2", lower=-1.0)
        self.add_output("u3", lower=-3.0)
        self.add_output("u4", lower=-2.0)

    def setup_partials(self):
        self.declare_partials(of=["*"], wrt=["*"], method="cs")

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["u1"] = outputs["u1"] * outputs["u2"]
        residuals["u2"] = np.cos(outputs["u3"]) * np.exp(outputs["u4"]) - outputs["u1"]
        residuals["u3"] = -outputs["u3"]
        residuals["u4"] = 1 / outputs["u3"]


class MultOutComp3(om.ImplicitComponent):
    def setup(self):
        self.add_output("u1", upper=10.0)
        self.add_output("u2", upper=5.0)
        self.add_output("u3", upper=-3.0)
        self.add_output("u4", upper=3.0)

    def setup_partials(self):
        self.declare_partials(of=["*"], wrt=["*"], method="cs")

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["u1"] = outputs["u1"] * outputs["u2"]
        residuals["u2"] = np.cos(outputs["u3"]) * np.exp(outputs["u4"]) - outputs["u1"]
        residuals["u3"] = -outputs["u3"]
        residuals["u4"] = 1 / outputs["u3"]


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
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        nl_solver.linesearch = None

        with self.assertRaises(ValueError) as context:
            p.setup()
            p.run_model()

        msg = "IPNewtonSolver in <model> <class Group>: solve_subsystems must be set by the user."
        self.assertEqual(str(context.exception), msg)

    def test_unbounded_no_linesearch(self):
        """Test to make sure the solver reaches the solution to a
        linear problem in a single iteration.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["pseudo_transient"] = False

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)

            p.setup()

            p.run_model()

            assert_near_equal(p.get_val("u"), 0.0)
            self.assertEqual(nl_solver._iter_count, 1)

    def test_no_linesearch_with_pt(self):
        """Test that pseudo transient continuation will work without
        a line search and will take more than a single iteration.
        - Bounds: None
        - LinearSolver: Direct
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 15
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["tau"] = 0.5

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)

            p.setup()

            p.set_val("u", 1.5)
            p.run_model()

            self.assertGreaterEqual(nl_solver._iter_count, 1)
            self.assertNotEqual(p.get_val("u"), 1)
            assert_near_equal(p.get_val("u"), 0.0, tolerance=1e-10)

    def test_with_pt(self):
        """Test that pseudo transient continuation will work with
        a line search and will take more than a single iteration.
        - Bounds: None
        - LinearSolver: Direct and BLSQ
        - Linesearch: BoundsEnforceLS (Default)
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 15
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["tau"] = 0.5

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)

            p.setup()

            p.set_val("u", 1.5)
            p.run_model()

            self.assertGreaterEqual(nl_solver._iter_count, 1)
            self.assertNotEqual(p.get_val("u"), 1)
            assert_near_equal(p.get_val("u"), 0.0, tolerance=1e-10)

    def test_direct_with_linesearch(self):
        """Test that the solver will find the solution in a single major
        iteration when using the bounds enforce line search without bounds.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for ls in ls_list:
            nl_solver.linesearch = ls()
            p.setup()
            p.set_val("u", 4.0)
            p.run_model()

            self.assertGreaterEqual(nl_solver._iter_count, 1)
            self.assertEqual(p.get_val("u"), 0)

    def test_blsq_with_linesearch(self):
        """Test that the solver will find the solution in a single major
        iteration when using the bounds enforce line search without bounds.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.LinearBLSQ(assemble_jac=True)

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for ls in ls_list:
            nl_solver.linesearch = ls()
            p.setup()
            p.set_val("u", 4.0)
            p.run_model()

            self.assertGreaterEqual(nl_solver._iter_count, 1)
            self.assertEqual(p.get_val("u"), 0)


class TestIPNewtonUnboundedVec(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_linesearch(self):
        """Test to check if the solver can find the solution to a linear
        problem with vectorized states.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: None
        """
        p = create_problem(lambda x: x, (5,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)
            p.setup()

            p.set_val("u", 4.0)
            p.run_model()

            self.assertEqual(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), np.zeros(5), tolerance=1e-14)

    def test_direct_with_linesearch(self):
        """Test to see if the solver can find the solution to the linear
        problem with vectorized states.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS (Default)
        """
        p = create_problem(lambda x: x, (5,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for ls in ls_list:
            nl_solver.linesearch = ls()

            p.setup()
            p.set_val("u", 4.0)
            p.run_model()

            self.assertEqual(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), np.zeros(5), tolerance=1e-14)

    def test_blsq_with_linesearch(self):
        """Test to see if the solver can find the solution to the linear
        problem with vectorized states.
        - Bounds: None
        - LinearSolver: DirectSolver
        - Linesearch: All
        """
        p = create_problem(lambda x: x, (5,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.LinearBLSQ(assemble_jac=True)

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for ls in ls_list:
            nl_solver.linesearch = ls()

            p.setup()
            p.set_val("u", 4.0)
            p.run_model()

            self.assertEqual(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), np.zeros(5), tolerance=1e-14)

    def test_pt_no_linesearch(self):
        p = create_problem(lambda x: x, (5,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        nl_solver.linesearch = None

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["tau"] = 5.0

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)
            p.setup()
            p.set_val("u", np.full(5, 4))
            p.run_model()

            self.assertGreater(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), np.zeros(5), tolerance=1e-10)

    def test_pt_with_linesearch(self):
        p = create_problem(lambda x: x, (5,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = True
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["tau"] = 10.0

        for lin_sol in lin_sol_list:
            for ls in ls_list:
                p.model.linear_solver = lin_sol(assemble_jac=True)

                nl_solver.linesearch = ls()
                if isinstance(nl_solver.linesearch, om.BracketingLS):
                    nl_solver.linesearch.options["spi_tol"] = 1e-15
                    nl_solver.linesearch.options["maxiter"] = 50
                p.setup()
                p.set_val("u", np.full(5, 4))
                p.run_model()

                # If we are using the inner product line search, it should overcome the pt
                # limit and find the solution to the linear problem in one major iteration.
                print(f"{type(nl_solver.linesearch)}, {nl_solver._iter_count}")
                if isinstance(nl_solver.linesearch, (om.InnerProductLS, om.BracketingLS)):
                    self.assertEqual(nl_solver._iter_count, 1)
                else:
                    self.assertGreater(nl_solver._iter_count, 1)

                assert_near_equal(p.get_val("u"), np.zeros(5), tolerance=1e-9)

    def test_penalty_no_linesearch(self):
        p = create_problem(lambda x: x, (5,))
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = True

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)
            p.setup()
            p.set_val("u", np.full(5, 4))
            p.run_model()

            self.assertEqual(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), np.zeros(5), tolerance=1e-10)


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

        p.setup()
        p.set_val("u", 1.5)
        p.run_model()

        self.assertEqual(nl_solver._iter_count, 10)
        assert_near_equal(p.get_val("u"), 1, tolerance=1e-10)

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

        p.setup()
        p.set_val("u", 1.5)
        p.run_model()

        self.assertEqual(nl_solver._iter_count, 10)
        assert_near_equal(p.get_val("u"), 2, tolerance=1e-10)

    def test_lower_bound_no_penalty(self):
        """Test that a bounded problem with a direct linear solver and
        the default line search will find the bounded solution in a single
        iteration when the penalty and pt are turned off.
        - Bounds: lower=1, upper=2
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS (Default)
        """
        p = create_problem(lambda x: x, (1,), lower=1, upper=2)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 1
        nl_solver.options["iprint"] = 1
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["pseudo_transient"] = False

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)
            p.setup()
            p.set_val("u", 1.5)

            p.run_model()

            self.assertEqual(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), 1.0, tolerance=1e-10)

    def test_upper_bound_no_penalty(self):
        """Test that a bounded problem with a direct linear solver and
        the default line search will find the bounded solution in a single
        iteration when the penalty and pt are turned off.
        - Bounds: lower=1, upper=2
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS (Default)
        """
        p = create_problem(lambda x: -x + 3, (1,), lower=1, upper=2)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["maxiter"] = 1
        nl_solver.options["iprint"] = 1
        nl_solver.options["interior_penalty"] = False
        nl_solver.options["pseudo_transient"] = False

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)
            p.setup()
            p.set_val("u", 1.5)

            p.run_model()

            self.assertEqual(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), 2.0, tolerance=1e-10)

    def test_direct_with_penalty(self):
        """Test that a bounded problem with a direct linear solver and
        the default line search will find the bounded solution.  In this test,
        we use the default penalty value which should make the solver
        take more than a single iteration to find the unbounded solution.
        - Bounds: lower=1, upper=2
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS (Default)
        """
        p = create_problem(lambda x: x, (1,), lower=1, upper=2)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = True

        for lin_sol in lin_sol_list:
            p.model.linear_solver = lin_sol(assemble_jac=True)
            p.setup()
            p.set_val("u", 1.5)
            p.run_model()

            self.assertGreater(nl_solver._iter_count, 1)
            assert_near_equal(p.get_val("u"), 1.0, 1e-2)

    def test_set_lower_bound_with_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=None
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: x, (1,), lower=1.0)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for lin_sol in lin_sol_list:
            for ls in ls_list:
                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()
                p.setup()
                p.set_val("u", 4.0)
                p.run_model()

                assert_near_equal(p.get_val("u"), 1.0, tolerance=1e-12)
                self.assertEqual(nl_solver.linesearch._lower_bounds, 1.0)
                self.assertEqual(nl_solver.linesearch._upper_bounds, np.inf)

    def test_set_upper_bound_with_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=None, upper=2.0
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """
        p = create_problem(lambda x: -x + 3, (1,), upper=2.0)
        nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

        nl_solver.options["solve_subsystems"] = True
        nl_solver.options["iprint"] = 1
        nl_solver.options["pseudo_transient"] = False
        nl_solver.options["interior_penalty"] = False

        for lin_sol in lin_sol_list:
            for ls in ls_list:
                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()

                p.setup()
                p.set_val("u", 1.0)
                p.run_model()

                assert_near_equal(p.get_val("u"), 2.0, tolerance=1e-12)
                self.assertEqual(nl_solver.linesearch._lower_bounds, -np.inf)
                self.assertEqual(nl_solver.linesearch._upper_bounds, 2.0)


class TestIPNewtonBoundedVec(unittest.TestCase):
    def setUp(self):
        pass

    def test_set_bounds_with_linesearch(self):
        """Test that the bounds are set correctly in each line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """

        lower_bounds = np.array([1.0, 1.0, -np.inf, -np.inf, 1.0])
        upper_bounds = np.array([5.0, np.inf, np.inf, np.inf, 5.0])

        for lin_sol in lin_sol_list:
            for ls in ls_list:
                p = create_problem(
                    lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
                )
                nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

                nl_solver.options["solve_subsystems"] = True
                nl_solver.options["iprint"] = 1
                nl_solver.options["pseudo_transient"] = False
                nl_solver.options["interior_penalty"] = False

                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()

                p.setup()
                p.set_val("u", np.full(5, 4.0))
                p.run_model()

                assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)
                assert_array_equal(nl_solver.linesearch._lower_bounds, nl_solver._lower_bounds)
                assert_array_equal(nl_solver.linesearch._upper_bounds, nl_solver._upper_bounds)

    def test_set_finite_mask_with_linesearch(self):
        """Test that the finite masks on the bounds are set correctly in
        the line search
        - Bounds: lower=1, upper=5
        - LinearSolver: DirectSolver
        - Linesearch: BoundsEnforceLS
        """

        lower_finite_mask = np.array([True, True, False, False, True])
        upper_finite_mask = np.array([True, False, False, False, True])

        for lin_sol in lin_sol_list:
            for ls in ls_list:
                p = create_problem(
                    lambda x: x, (5,), lower=np.array([1, 1, None, None, 1]), upper=np.array([5, None, None, None, 5])
                )
                nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()

                nl_solver.options["solve_subsystems"] = True
                nl_solver.options["iprint"] = 1
                nl_solver.options["pseudo_transient"] = False
                nl_solver.options["interior_penalty"] = False

                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()

                p.setup()
                p.set_val("u", np.full(5, 4.0))
                p.run_model()

                assert_array_equal(nl_solver.linesearch._upper_finite_mask, nl_solver._upper_finite_mask)
                assert_array_equal(nl_solver.linesearch._lower_finite_mask, nl_solver._lower_finite_mask)
                assert_array_equal(nl_solver.linesearch._lower_finite_mask, lower_finite_mask)
                assert_array_equal(nl_solver.linesearch._upper_finite_mask, upper_finite_mask)

    def test_set_bounds_multiple_outputs_case_1(self):
        lower_bounds = np.array([-np.inf, -1.0, -np.inf, -2.0])
        upper_bounds = np.array([np.inf, np.inf, -3.0, 3.0])

        for ls in ls_list:
            for lin_sol in lin_sol_list:
                p = om.Problem()
                p.model.add_subsystem("mult", MultOutComp1(), promotes=["*"])
                nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=0, solve_subsystems=True)
                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()
                p.setup()

                p.set_val("u1", val=-1.0)
                p.set_val("u2", val=-0.3)
                p.set_val("u3", val=-10.0)
                p.set_val("u4", val=1.0)

                p.run_model()

                assert_array_equal(nl_solver._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver._upper_bounds, upper_bounds)
                assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)

                if isinstance(lin_sol, om.LinearBLSQ):
                    assert_array_equal(nl_solver.linear_solver.lower_bounds, lower_bounds)
                    assert_array_equal(nl_solver.linear_solver.upper_bounds, upper_bounds)

    def test_set_bounds_multiple_outputs_case_2(self):
        lower_bounds = np.array([0.0, -1.0, -3.0, -2.0])
        upper_bounds = np.full(4, np.inf)

        for ls in ls_list:
            for lin_sol in lin_sol_list:
                p = om.Problem()
                p.model.add_subsystem("mult", MultOutComp2(), promotes=["*"])
                nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=0, solve_subsystems=True)
                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()
                p.setup()

                p.set_val("u1", val=-1.0)
                p.set_val("u2", val=-0.3)
                p.set_val("u3", val=-10.0)
                p.set_val("u4", val=1.0)

                p.run_model()

                assert_array_equal(nl_solver._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver._upper_bounds, upper_bounds)
                assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)

                if isinstance(lin_sol, om.LinearBLSQ):
                    assert_array_equal(nl_solver.linear_solver.lower_bounds, lower_bounds)
                    assert_array_equal(nl_solver.linear_solver.upper_bounds, upper_bounds)

    def test_set_bounds_multiple_outputs_case_3(self):
        lower_bounds = np.full(4, -np.inf)
        upper_bounds = np.array([10.0, 5.0, -3.0, 3.0])

        for ls in ls_list:
            for lin_sol in lin_sol_list:
                p = om.Problem()
                p.model.add_subsystem("mult", MultOutComp3(), promotes=["*"])
                nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=0, solve_subsystems=True)
                p.model.linear_solver = lin_sol(assemble_jac=True)
                nl_solver.linesearch = ls()
                p.setup()

                p.set_val("u1", val=-1.0)
                p.set_val("u2", val=-0.3)
                p.set_val("u3", val=-10.0)
                p.set_val("u4", val=1.0)

                p.run_model()

                assert_array_equal(nl_solver._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver._upper_bounds, upper_bounds)
                assert_array_equal(nl_solver.linesearch._lower_bounds, lower_bounds)
                assert_array_equal(nl_solver.linesearch._upper_bounds, upper_bounds)

                if isinstance(lin_sol, om.LinearBLSQ):
                    assert_array_equal(nl_solver.linear_solver.lower_bounds, lower_bounds)
                    assert_array_equal(nl_solver.linear_solver.upper_bounds, upper_bounds)


if __name__ == "__main__":
    unittest.main()
