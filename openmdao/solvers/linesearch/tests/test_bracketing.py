""" Test for the Bracketing Line Search"""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

def create_comp(res_func, shape, lower=None, upper=None, deriv_method='cs'):
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
            self.add_output('u', shape=shape, lower=lower, upper=upper)
        def setup_partials(self):
            self.declare_partials(of=['*'], wrt=['*'], method=deriv_method)
        def apply_nonlinear(self, inputs, outputs, residuals):
            residuals['u'] = res_func(outputs['u'])

    return Comp

def create_problem(res_func, shape, lower=None, upper=None,
                   deriv_method='cs', iprint=2, **ls_opts):
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
    p.model.add_subsystem('comp',
                          create_comp(res_func, shape, lower=lower, upper=upper,
                                      deriv_method=deriv_method)(),
                          promotes=['*'])
    p.model.nonlinear_solver = om.NewtonSolver(maxiter=1, iprint=iprint, solve_subsystems=True)
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=iprint, **ls_opts)
    p.setup()
    return p


class TestBracketingCubic(unittest.TestCase):
    """
    r = u^3 + u^2
    """
    def setUp(self):
        self.p = create_problem(lambda x: x**3 + x**2, (1,),
                                maxiter=100, spi_tol=1e-6)
    
    def test_backtracking_finds_min(self):
        """
        In this case, the Newton solver overshoots the minimum
        by enough that the objective at the Newton step is greater
        than where it starts off. This should send the line search
        backward. When alpha is cut in half, the objective is now
        less than at alpha of 0 or 1, so it brackets and calls SPI
        to converge to the solution (hopfeully).
        """
        p = self.p
        p.set_val('u', val=-0.58)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-6)

    def test_backtracking_finds_min_harder(self):
        """
        In this case, the Newton solver overshoots the minimum
        by enough that the objective at the Newton step is greater
        than where it starts off. This should send the line search
        backward. When alpha is cut in half, the objective is still
        greater than the initial one, so it needs to backtrack until
        it finds a bracket. Then it can finally call SPI to converge
        to the solution (hopfeully).
        """
        p = self.p
        p.set_val('u', val=-0.64)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-6)
    
    def test_backtracking_finds_min_cusp(self):
        """
        In this case, the Newton solver overshoots the minimum
        by enough that the objective at the Newton step is greater
        than where it starts off. This should send the line search
        backward. Then it will use SPI to find a minimum (which
        happens to have a C1 discontinuous L2 norm).
        """
        p = self.p
        p.set_val('u', val=-0.82)
        p.run_model()
        assert_near_equal(p.get_val('u'), -1., tolerance=1e-6)
    
    def test_backtracking_finds_min_cusp_harder(self):
        """
        In this case, the Newton solver overshoots the minimum
        by enough that the objective at the Newton step is greater
        than where it starts off. This should send the line search
        backward. When alpha is cut in half, the objective is still
        greater than the initial one, so it needs to backtrack until
        it finds a bracket. Then it can finally call SPI to converge
        to the solution (which happens to have a C1 discontinuous L2 norm).
        """
        p = self.p
        p.set_val('u', val=-0.72)
        p.run_model()
        assert_near_equal(p.get_val('u'), -1., tolerance=1e-6)
    
    def test_forwardtracking_easy(self):
        """
        Point it toward the C1 continuous minimum, but from a
        point that's concave up. This means the Newton step
        undershoots to solution and the line search will need
        to forward track.
        """
        p = self.p
        p.set_val('u', val=-0.34)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-6)
    
    def test_forwardtracking_cusp(self):
        """
        Point it toward the C1 discontinuous minimum. The
        objective at the Newton step will be less than the one
        at the initial point, but it's actually past the minimum.
        Still the line search should forward track and eventually
        converge to the minimum.
        """
        p = self.p
        p.set_val('u', val=-0.84)
        p.run_model()
        assert_near_equal(p.get_val('u'), -1., tolerance=1e-6)
    
    def test_forwardtracking_further(self):
        """
        This case requires far more forward tracking iterations
        than the other ones so far. Eventually, it will bracket
        both minimums and converge to the one at 0.
        """
        p = self.p
        p.set_val('u', val=3.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-5)

class TestBracketingMonatonic(unittest.TestCase):
    """
    r = e^(-u)     and     r = e^u

    This test case handles simple bounds testing and
    hitting alpha max.
    """
    def test_forward(self):
        """
        The function decreases forever, so it should step
        only to alpha max (100) and return that value.
        Conveniently, the Newton step starting at u = 0
        is to u = 1, so alpha = u for this case.
        """
        alpha_max = 10
        p = create_problem(lambda x: np.exp(-x), (1,),
                           maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()
        assert_near_equal(p.get_val('u'), alpha_max, tolerance=1e-10)
    
    def test_forward_simple_upper_bound(self):
        """
        The function is bounded at an alpha > 1
        but < alpha_max, so it should hit and stop.
        """
        alpha_max = 10
        upper = alpha_max - 1
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return np.exp(-x)
        p = create_problem(func, (1,),
                           upper=upper, maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()

        # Should never reach the bound, just be very close to it
        assert_near_equal(p.get_val('u'), upper, tolerance=1e-10)
        self.assertLess(p.get_val('u')[0], upper)
    
    def test_backward_simple_upper_bound(self):
        """
        The function is bounded at an alpha < 1,
        so it should back up to the bound and stop.
        """
        alpha_max = 10
        upper = 0.5
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return np.exp(-x)
        p = create_problem(func, (1,),
                           upper=upper, maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()

        # Should never reach the bound, just be very close to it
        assert_near_equal(p.get_val('u'), upper, tolerance=1e-10)
        self.assertLess(p.get_val('u')[0], upper)
    
    def test_forward_simple_lower_bound(self):
        """
        The function is bounded at an alpha > 1
        but < alpha_max, so it should hit and stop.
        """
        alpha_max = 10
        lower = -alpha_max + 1
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return np.exp(x)
        p = create_problem(func, (1,),
                           lower=lower, maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()

        # Should never reach the bound, just be very close to it
        assert_near_equal(p.get_val('u'), lower, tolerance=1e-10)
        self.assertGreater(p.get_val('u')[0], lower)
    
    def test_backward_simple_lower_bound(self):
        """
        The function is bounded at an alpha < 1,
        so it should back up to the bound and stop.
        """
        alpha_max = 10
        lower = -0.5
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return np.exp(x)
        p = create_problem(func, (1,),
                           lower=lower, maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()

        # Should never reach the bound, just be very close to it
        assert_near_equal(p.get_val('u'), lower, tolerance=1e-10)
        self.assertGreater(p.get_val('u')[0], lower)
    
    def test_forward_alpha_max_is_upper(self):
        """
        Alpha max brings the line search exactly to the bound.
        This checks that it still never hits the bound.
        """
        alpha_max = 10
        upper = alpha_max
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return np.exp(-x)
        p = create_problem(func, (1,),
                           upper=upper, maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()

        # Should never reach the bound, just be very close to it
        assert_near_equal(p.get_val('u'), upper, tolerance=1e-10)
        self.assertLess(p.get_val('u')[0], upper)
    
    def test_forward_alpha_max_is_lower(self):
        """
        Alpha max brings the line search exactly to the bound.
        This checks that it still never hits the bound.
        """
        alpha_max = 10
        lower = -alpha_max
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return np.exp(x)
        p = create_problem(func, (1,),
                           lower=lower, maxiter=100, spi_tol=1e-6,
                           alpha_max=alpha_max, beta=2)
        p.set_val('u', val=0.)
        p.run_model()

        # Should never reach the bound, just be very close to it
        assert_near_equal(p.get_val('u'), lower, tolerance=1e-10)
        self.assertGreater(p.get_val('u')[0], lower)

class TestBracketingParabola(unittest.TestCase):
    """
    r = u^2
    """
    def test_forward(self):
        """
        Should find the solution on the first SPI iteration.
        """
        p = create_problem(lambda x: x**2, (1,),
                           spi_tol=1e-6, beta=4)
        p.set_val('u', val=-1.)
        p.run_model()

        # Should only take three iterations:
        #   First evaluation at alpha = 1
        #   Forward track to alpha = 1 * beta = 3
        #   Build parabola and solve for minimum (SPI)
        assert_near_equal(p.get_val('u'), 0.)
        self.assertEqual(p.model.nonlinear_solver.linesearch._iter_count, 3)

class TestBracketingTrickyBounds(unittest.TestCase):
    """
    These tests verifies the accuracy of SPI and
    also more complex cases where there is a bound
    but still a minimum in the feasible region.
    """   
    def test_forward_bracketing_upper_bound(self):
        """
        In this case, it forward tracks to the upper bound and the
        objective at the bound is greater than at the initial point.
        Thus, it brackets a minimum and should converge to it.
        """
        upper = 1.25
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return x**2
        p = create_problem(func, (1,), upper=upper,
                           maxiter=10, spi_tol=1e-6,
                           alpha_max=10, beta=5)
        p.set_val('u', val=-1.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0.)
    
    def test_forward_bracketing_lower_bound(self):
        """
        In this case, it forward tracks to the lower bound and the
        objective at the bound is greater than at the initial point.
        Thus, it brackets a minimum and should converge to it.
        """
        lower = -1.25
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return x**2
        p = create_problem(func, (1,), lower=lower,
                           maxiter=10, spi_tol=1e-6,
                           alpha_max=10, beta=5)
        p.set_val('u', val=1.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0.)

    def test_forward_bracketingish_upper_bound(self):
        """
        In this case, it forward tracks to the upper bound and the
        objective at the bound is less than at the initial point. BUT
        it is greater than the point at alpha = 1, so it should
        still bracket and find the minimum.
        """
        upper = 0.6
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return x**2
        p = create_problem(func, (1,), upper=upper,
                           maxiter=10, spi_tol=1e-6,
                           alpha_max=10, beta=5)
        p.set_val('u', val=-1.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0.)
    
    def test_forward_bracketingish_lower_bound(self):
        """
        In this case, it forward tracks to the lower bound and the
        objective at the bound is less than at the initial point. BUT
        it is greater than the point at alpha = 1, so it should
        still bracket and find the minimum.
        """
        lower = -0.6
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return x**2
        p = create_problem(func, (1,), lower=lower,
                           maxiter=10, spi_tol=1e-6,
                           alpha_max=10, beta=5)
        p.set_val('u', val=1.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0.)

    def test_backward_bracketing_upper_bound(self):
        """
        In this case, it hits the upper bound at alpha < 1. The objective
        at the bound is greater than the initial point, so it brackets.
        Thus, it should converge to the minimum.
        """
        upper = 0.5
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return x**3 + x**2
        p = create_problem(func, (1,), upper=upper,
                           maxiter=20, spi_tol=1e-6,
                           alpha_max=10, beta=2)
        p.set_val('u', val=-0.6)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-5)
    
    def test_backward_bracketing_lower_bound(self):
        """
        In this case, it hits the lower bound at alpha < 1. The objective
        at the bound is greater than the initial point, so it brackets.
        Thus, it should converge to the minimum.
        """
        lower = -0.5
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return (-x)**3 + x**2
        p = create_problem(func, (1,), lower=lower,
                           maxiter=20, spi_tol=1e-6,
                           alpha_max=10, beta=2)
        p.set_val('u', val=0.6)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-5)

class TestBracketingBack(unittest.TestCase):
    """
    Test that the backtracking part of bracketing works properly.
    """
    def test_backward_bracketing_case_one_return(self):
        """
        The objective at the Newton step is greater than at the initial point.
        The line search cuts the step by beta, and the objective there is even
        greater than at the Newton step. The line search is cut off at this point
        by setting maxiter appropriately. This test checks that returned value is
        at the Newton step (since it's lower) rather than the most recently
        evaluated point (alpha = 1/1.55).
        """
        p = create_problem(lambda x: (x - 1)**2 * (2*x + 1)**2 * x**2, (1,),
                           maxiter=2, beta=1.55)
        p.set_val('u', val=-0.263)
        p.run_model()
        assert_near_equal(p.get_val('u'), 1.0716, tolerance=1e-4)  # state at the Newton step
    
    def test_backward_bracketing_case_one(self):
        """
        The objective at the Newton step is greater than at the initial point.
        The line search cuts the step in half, and the objective there is even
        greater than at the Newton step. The backtracking should continue
        until it brackets the minimum at u = 0 and then use SPI to converge.
        """
        p = create_problem(lambda x: (x - 1)**2 * (2*x + 1)**2 * x**2, (1,),
                           maxiter=30, spi_tol=1e-6, beta=1.55)
        p.set_val('u', val=-0.263)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-5)
    
    def test_backward_bracketing_case_two_return(self):
        """
        The objective at the Newton step is greater than at the initial point.
        The line search cuts the step by beta, and the objective there is less
        than at the Newton step but greater than the initial point. The line
        search is cut off at this point by setting maxiter appropriately. This
        test checks that returned value is at the alpha = 1/beta case rather than
        at the Newton step.
        """
        beta = 2.4
        p = create_problem(lambda x: (x - 1)**2 * (2*x + 1)**2 * x**2, (1,),
                           maxiter=2, beta=beta)
        p.set_val('u', val=-0.265)
        p.run_model()
        second_u = (1.3543 + 0.265) / beta - 0.265
        assert_near_equal(p.get_val('u'), second_u, tolerance=1e-3)

    def test_backward_bracketing_case_two(self):
        """
        The objective at the Newton step is greater than at the initial point.
        The line search cuts the step by beta, and the objective there is less
        than at the Newton step but greater than the initial point. The line
        search should then converge to the minimum at u = 0.
        """
        beta = 2.4
        p = create_problem(lambda x: (x - 1)**2 * (2*x + 1)**2 * x**2, (1,),
                           maxiter=20, spi_tol=1e-6, beta=beta)
        p.set_val('u', val=-0.265)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-5)
    
    def test_backward_bracketing_case_three_return(self):
        """
        The objective at the Newton step is greater than at the initial point.
        The line search cuts the step by beta, and the objective there is less
        than at the Newton step and at the initial point. The line search is
        cut off at this point by setting maxiter appropriately. The returned
        value should be at alpha = 0.5.
        """
        p = create_problem(lambda x: (x - 1)**2 * (2*x + 1)**2 * x**2, (1,),
                           maxiter=2, beta=2)
        p.set_val('u', val=-0.24)
        p.run_model()
        assert_near_equal(p.get_val('u'), -0.01815, tolerance=1e-3)
    
    def test_backward_bracketing_case_three(self):
        """
        The objective at the Newton step is greater than at the initial point.
        The line search cuts the step by beta, and the objective there is less
        than at the Newton step and at the initial point. The line search
        should then converge to the minimum at u = 0.
        """
        p = create_problem(lambda x: (x - 1)**2 * (2*x + 1)**2 * x**2, (1,),
                           maxiter=20, spi_tol=1e-6, beta=2)
        p.set_val('u', val=-0.24)
        p.run_model()
        assert_near_equal(p.get_val('u'), 0., tolerance=1e-6)

class TestBracketingHighCurvature(unittest.TestCase):
    def test_high_curvature(self):
        """
        r = -u - 0.1 log(-2 - x)

        This function has high curvature (basically a log barrier
        with a low mu) to test the convergence on a sharp problem.
        """
        p = create_problem(lambda x: -x - 0.01 * np.log(-2 - x), (1,),
                           upper=-2, maxiter=10000, spi_tol=1e-8, beta=2)
        p.set_val('u', val=-2.2)
        p.run_model()
        assert_near_equal(p.get_val('u'), -2.01, tolerance=1e-4)

class TestBracketingMultDim(unittest.TestCase):
    """
    Test the line search on multidimensional problems.
    """
    def test_unbounded_paraboloid(self):
        """
        Test that it converges to the solution of an
        unbounded parabola in 3D in three iterations.
        """
        p = create_problem(lambda x: np.array([ 4*x[0]**2, 3*x[1]**2, x[2]**2 ]),
                           (3,), maxiter=10, spi_tol=1e-6, beta=3.5)
        p.set_val('u', val=np.array([-1, -0.5, 2.]))
        p.run_model()
        assert_near_equal(p.get_val('u'), np.zeros(3), tolerance=1e-14)
        self.assertEqual(p.model.nonlinear_solver.linesearch._iter_count, 3)

    def test_upper_bounded_paraboloid(self):
        """
        Test that it converges to the upper bound
        with a bounded parabola in 2D.
        """
        p = create_problem(lambda x: np.array([ 4*x[0]**2, 3*x[1]**2 ]),
                           (2,), upper=np.array([-0.7, -0.5]), maxiter=10,
                           spi_tol=1e-6, beta=3.5)
        p.set_val('u', val=np.array([-1, -1]))
        p.run_model()
        assert_near_equal(p.get_val('u'), np.array([-0.7, -0.7]), tolerance=1e-10)

    def test_lower_bounded_paraboloid(self):
        """
        Test that it converges to the lower bound
        with a bounded parabola in 2D.
        """
        p = create_problem(lambda x: np.array([ 4*x[0]**2, 3*x[1]**2 ]),
                           (2,), lower=np.array([0.5, 0.7]), maxiter=10,
                           spi_tol=1e-6, beta=3.5)
        p.set_val('u', val=np.array([1, 1]))
        p.run_model()
        assert_near_equal(p.get_val('u'), np.array([0.7, 0.7]), tolerance=1e-10)

class TestBracketingPenalty(unittest.TestCase):
    """
    Test the penalty function implementation.
    """
    def test_simple_upper(self):
        """
        Upper bound on a r = -u at u = -1. Should
        converge to the penalized minimum
        (not the bound).
        """
        upper = -1.
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return -x

        p = om.Problem()
        p.model.add_subsystem('comp',
                              create_comp(func, (1,), upper=upper)(),
                              promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=1., solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u', val=-10.)
        p.run_model()
        assert_near_equal(p.get_val('u'), -2., tolerance=1e-3)
    
    def test_simple_lower(self):
        """
        Lower bound on a r = u at u = 1. Should
        converge to the penalized minimum
        (not the bound).
        """
        lower = 1.
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return x

        p = om.Problem()
        p.model.add_subsystem('comp',
                              create_comp(func, (1,), lower=lower)(),
                              promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=1., solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u', val=10.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 2., tolerance=1e-3)
    
    def test_simple_upper_negative(self):
        """
        Upper bound on a r = u at u = -1, so now the residuals
        will be negative (objective should be unchanged
        from test_simple_upper test case). Should
        converge to the penalized minimum
        (not the bound).
        """
        upper = -1.
        def func(x):
            if x >= upper:
                raise ValueError(f"Upper bound of {upper} violated by input of {x[0]}")
            return x

        p = om.Problem()
        p.model.add_subsystem('comp',
                              create_comp(func, (1,), upper=upper)(),
                              promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=1., solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u', val=-10.)
        p.run_model()
        assert_near_equal(p.get_val('u'), -2., tolerance=1e-3)
    
    def test_simple_lower_negative(self):
        """
        Lower bound on a r = -u at u = 1, so now the residuals
        will be negative (objective should be unchanged
        from test_simple_lower test case). Should
        converge to the penalized minimum
        (not the bound).
        """
        lower = 1.
        def func(x):
            if x <= lower:
                raise ValueError(f"Lower bound of {lower} violated by input of {x[0]}")
            return -x

        p = om.Problem()
        p.model.add_subsystem('comp',
                              create_comp(func, (1,), lower=lower)(),
                              promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=1., solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u', val=10.)
        p.run_model()
        assert_near_equal(p.get_val('u'), 2., tolerance=1e-3)

class TestBracketingRegression(unittest.TestCase):
    """
    A few more complicated regression tests to catch any multi-dimensional
    nuances that may not have been caught in previous simpler tests.
    """
    def test_five_D_bounded_with_penalty(self):
        lower = np.array([-1., -5., 1., -3., 2.])
        upper = np.array([0., -3., 10., 5., 2.1])
        def func(x):
            if np.any(x <= lower):
                raise ValueError(f"Lower bound of {lower} violated by input of {x}")
            if np.any(x >= upper):
                raise ValueError(f"Upper bound of {upper} violated by input of {x}")
            return np.array([
                np.sin(x[1]) + x[0]**2 * x[2] + 1/x[3] - 1/x[4],
                x[0] + x[3]**2,
                x[4] * np.exp(x[1]/x[0]),
                x[0] + 3*x[1] - x[2] + x[3]*x[4],
                x[4] / x[3] - x[0] - 5
            ])

        p = om.Problem()
        p.model.add_subsystem('comp',
                              create_comp(func, (5,), lower=lower, upper=upper)(),
                              promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=1., solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u', val=np.array([-0.9, -3.1, 5., 0.1, 2.0001]))
        p.run_model()
        assert_near_equal(p.get_val('u'), np.array([-0.97199227, -3.12976849, 1.27969818, 0.11538653, 2.00018035]), tolerance=1e-7)

    def test_four_D_semibounded_with_penalty(self):
        lower = np.array([-np.inf, -5., -np.inf, -3.])
        upper = np.array([np.inf, -3., 10., np.inf])
        def func(x):
            if np.any(x <= lower):
                raise ValueError(f"Lower bound of {lower} violated by input of {x}")
            if np.any(x >= upper):
                raise ValueError(f"Upper bound of {upper} violated by input of {x}")
            return np.array([
                np.sin(x[1]) + x[0]**2 * x[2] + 1/x[3] - 1/x[1],
                x[0] - x[3]**2,
                x[2] * np.exp(x[1]/x[0]),
                x[0] + 3*x[1] - x[2] + x[3],
            ])

        p = om.Problem()
        p.model.add_subsystem('comp',
                              create_comp(func, (4,), lower=lower, upper=upper)(),
                              promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=1., solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u', val=np.array([-0.9, -3.1, 5., 0.1]))
        p.run_model()
        assert_near_equal(p.get_val('u'), np.array([-0.73079907, -3.11435048, 0.02976901, 0.092921]), tolerance=1e-6)
    
    def test_multi_state_comp_semibounded(self):
        class Comp(om.ImplicitComponent):
            def setup(self):
                self.add_output('u1')
                self.add_output('u2', lower=-1.)
                self.add_output('u3', upper=-3.)
                self.add_output('u4', lower=-2., upper=3.)
            def setup_partials(self):
                self.declare_partials(of=['*'], wrt=['*'], method='cs')
            def apply_nonlinear(self, inputs, outputs, residuals):
                if outputs['u2'] <= -1.:
                    raise ValueError(f"Lower bound of -1 on state u2 violated by value of {outputs['u2']}")
                if outputs['u3'] >= -3.:
                    raise ValueError(f"Upper bound of -3 on state u3 violated by value of {outputs['u3']}")
                if outputs['u4'] <= -2.:
                    raise ValueError(f"Lower bound of -2 on state u4 violated by value of {outputs['u4']}")
                if outputs['u4'] >= 3.:
                    raise ValueError(f"Upper bound of 3 on state u4 violated by value of {outputs['u4']}")
                residuals['u1'] = outputs['u1'] * outputs['u2']
                residuals['u2'] = np.cos(outputs['u3']) * np.exp(outputs['u4']) - outputs['u1']
                residuals['u3'] = -outputs['u3']
                residuals['u4'] = 1 / outputs['u3']

        p = om.Problem()
        p.model.add_subsystem('comp', Comp(), promotes=['*'])
        p.model.nonlinear_solver = om.IPNewtonSolver(maxiter=1, iprint=2, interior_penalty=True,
                                                     mu=0.1, solve_subsystems=True)
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=2, maxiter=500, spi_tol=1e-6)
        p.setup()
        
        p.set_val('u1', val=-1.)
        p.set_val('u2', val=-0.3)
        p.set_val('u3', val=-10.)
        p.set_val('u4', val=1.)
        p.run_model()
        assert_near_equal(p.get_val('u1'), 1.56992041, tolerance=1e-6)
        assert_near_equal(p.get_val('u2'), -1., tolerance=1e-6)
        assert_near_equal(p.get_val('u3'), -7.59984122, tolerance=1e-6)
        assert_near_equal(p.get_val('u4'), -1.85961775, tolerance=1e-6)

# TODO: check error checking? For example, what happens if the line search starts in the infeasible region? How to get it to search uphill just in case?

if __name__ == "__main__":
    unittest.main()
