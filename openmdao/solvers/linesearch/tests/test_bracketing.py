""" Test for the Bracketing Line Search"""

import sys
import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.general_utils import printoptions


def create_comp(res_func, shape, deriv_method='cs'):
    """Create an OpenMDAO component from a residual function.

    Parameters
    ----------
    res_func : function handle
        Takes in ndarray u and returns residual vector r
    shape : tuple
        Shape of input ndarray u and residual vector of res_func
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
            self.add_output('u', shape=shape)
        def setup_partials(self):
            self.declare_partials(of=['*'], wrt=['*'], method=deriv_method)
        def apply_nonlinear(self, inputs, outputs, residuals):
            residuals['u'] = res_func(outputs['u'])

    return Comp

def create_problem(res_func, shape, deriv_method='cs', iprint=2, **ls_opts):
    """Create OpenMDAO Problem to test the line search by using a top
       level Newton solver with maxiter of 1 and adding the BracketingLS.

    Parameters
    ----------
    res_func : function handle
        Takes in ndarray u and returns residual vector r
    shape : tuple
        Shape of input ndarray u and residual vector of res_func
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
                          create_comp(res_func, shape, deriv_method=deriv_method)(),
                          promotes=['*'])
    p.model.nonlinear_solver = om.NewtonSolver(maxiter=1, iprint=iprint, solve_subsystems=True)
    p.model.linear_solver = om.DirectSolver()
    p.model.nonlinear_solver.linesearch = om.BracketingLS(iprint=iprint, **ls_opts)
    p.setup()
    return p


class TestBracketingCubic(unittest.TestCase):
    def setUp(self):
        self.p = create_problem(lambda x: x**3 + x**2, (1,),
                                maxiter=20, spi_tol=1e-6)
    
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


if __name__ == "__main__":
    unittest.main()
