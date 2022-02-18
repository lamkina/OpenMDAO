""" Test for the Bracketing Line Search"""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


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


class TestIPNewtonLinear(unittest.TestCase):
    def setUp(self):
        p = self.prob = create_problem(lambda x: x, (1,))
        self.nl_solver = p.model.nonlinear_solver = om.IPNewtonSolver()
        self.lin_solver = p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        self.ls = self.nl_solver.linesearch = None

    def test_solve_subsystems_error(self):
        """Test to ensure the correct error is thrown when the user
        forgets to set solve_subsystems
        """
        p = self.prob

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
        p = self.prob
        self.nl_solver.options["solve_subsystems"] = True
        self.nl_solver.options["interior_penalty"] = False
        self.nl_solver.options["pseudo_transient"] = False

        p.run_model()

        assert_near_equal(p.get_val("u"), 0.0)
        self.assertEqual(self.nl_solver._iter_count, 1)

    def test_bounded_direct_no_linesearch_no_penalty(self):
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

    def test_bounded_direct_no_linesearch_penalty(self):
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

    def test_unbounded_direct_no_linesearch_pt(self):
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

    def test_lower_bound_blsq_no_linesearch(self):
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


if __name__ == "__main__":
    unittest.main()
