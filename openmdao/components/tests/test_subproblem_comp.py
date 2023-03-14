from numpy import pi

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import unittest


class TestSubproblemComp(unittest.TestCase):
    def test_subproblem_comp(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.SubproblemComp(model=submodel1, inputs=['r', 'theta'],
                                  outputs=['x'])
        subprob2 = om.SubproblemComp(model=submodel2, inputs=['r', 'theta'],
                                  outputs=['y'])

        p.model.add_subsystem('sub1', subprob1, promotes_inputs=['r','theta'],
                                    promotes_outputs=['x'])
        p.model.add_subsystem('sub2', subprob2, promotes_inputs=['r','theta'],
                                    promotes_outputs=['y'])
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                                    promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        p.set_val('r', 1)
        p.set_val('theta', pi)

        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_near_equal(p.get_val('z'), 1.0)

    def test_no_io(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.SubproblemComp(model=submodel1)
        subprob2 = om.SubproblemComp(model=submodel2)

        p.model.add_subsystem('sub1', subprob1)
        p.model.add_subsystem('sub2', subprob2)
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                                    promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        with self.assertRaises(Exception) as ctx:
            p.set_val('r', 1)
            p.set_val('theta', pi)

            p.run_model()

    def test_variable_alias(self):
        p = om.Problem()
        model = om.Group()

        model.add_subsystem('subsys', om.ExecComp('z = x**2 + y**2'))
        subprob = om.SubproblemComp(model=model, inputs=[('subsys.x', 'a'), ('subsys.y', 'b')],
                                    outputs=[('subsys.z', 'c')])

        p.model.add_subsystem('subprob', subprob, promotes_inputs=['a', 'b'], promotes_outputs=['c'])
        p.setup()

        p.set_val('a', 1)
        p.set_val('b', 2)

        p.run_model()

        inputs = p.model.subprob.list_inputs()
        outputs = p.model.subprob.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert(inputs['a'] == 1)
        assert(inputs['b'] == 2)
        assert(outputs['c'] == 5)

    def test_unconnected_same_var(self):
        p = om.Problem()

        model = om.Group()

        model.add_subsystem('x1Comp', om.ExecComp('x1 = x*3'))
        model.add_subsystem('x2Comp', om.ExecComp('x2 = x**3'))
        model.connect('x1Comp.x1', 'model.x1')
        model.connect('x2Comp.x2', 'model.x2')
        model.add_subsystem('model', om.ExecComp('z = x1**2 + x2**2'))

        subprob = om.SubproblemComp(model=model, inputs=[('x1Comp.x', 'x'), ('x2Comp.x', 'y')],
                                    outputs=[('model.z', 'z')])

        p.model.add_subsystem('subprob', subprob)
        p.setup()

        p.set_val('subprob.x', 1)
        p.set_val('subprob.y', 2)

        p.run_model()

        inputs = p.model.subprob.list_inputs()
        outputs = p.model.subprob.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert(inputs['x'] == 1)
        assert(inputs['y'] == 2)
        assert(outputs['z'] == 73)

    def test_wildcard(self):
        p = om.Problem()
        model = om.Group()

        model.add_subsystem('subsys', om.ExecComp('z = x1**2 + x2**2 + x3**2'), promotes=['*'])
        subprob = om.SubproblemComp(model=model, inputs=['x*'], outputs=['*'])

        p.model.add_subsystem('prob', subprob, promotes_inputs=['*'], promotes_outputs=['*'])
        p.setup()

        p.set_val('x1', 1)
        p.set_val('x2', 2)
        p.set_val('x3', 3)

        p.run_model()

        inputs = p.model.prob.list_inputs()
        outputs = p.model.prob.list_outputs()

        inputs = {inputs[i][0]: inputs[i][1]['val'] for i in range(len(inputs))}
        outputs = {outputs[i][0]: outputs[i][1]['val'] for i in range(len(outputs))}

        assert(inputs['x1'] == 1)
        assert(inputs['x2'] == 2)
        assert(inputs['x3'] == 3)
        assert(outputs['z'] == 14)
    
    def test_add_io_before_setup(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.SubproblemComp(model=submodel1)
        subprob2 = om.SubproblemComp(model=submodel2)
        
        subprob1.add_input('r')
        subprob1.add_input('theta')
        subprob2.add_input('r')
        subprob2.add_input('theta')
        
        subprob1.add_output('x')
        subprob2.add_output('y')

        p.model.add_subsystem('sub1', subprob1, promotes_inputs=['r','theta'],
                                    promotes_outputs=['x'])
        p.model.add_subsystem('sub2', subprob2, promotes_inputs=['r','theta'],
                                    promotes_outputs=['y'])
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                                    promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        p.set_val('r', 1)
        p.set_val('theta', pi)

        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)
        
        assert_near_equal(p.get_val('z'), 1.0)

    def test_add_io_after_setup(self):
        class Subsys1(om.Group):
            def setup(self):
                model = om.Group()
                comp = om.ExecComp('x = r*cos(theta)')
                model.add_subsystem('comp', comp, promotes_inputs=['r', 'theta'],
                                    promotes_outputs=['x'])
                self.add_subsystem('subprob1', om.SubproblemComp(model=model))

            def configure(self):
                self._get_subsystem('subprob1').add_input('r')
                self._get_subsystem('subprob1').add_input('theta')
                self._get_subsystem('subprob1').add_output('x')

                self.promotes('subprob1', ['r'])
                self.promotes('subprob1', ['theta'])
                self.promotes('subprob1', ['x'])

        class Subsys2(om.Group):
            def setup(self):
                # adds subsystems -> subprobs
                model = om.Group()
                comp = om.ExecComp('y = r*sin(theta)')
                model.add_subsystem('comp', comp, promotes_inputs=['r', 'theta'],
                                    promotes_outputs=['y'])
                self.add_subsystem('subprob2', om.SubproblemComp(model=model))

            def configure(self):
                self._get_subsystem('subprob2').add_input('r')
                self._get_subsystem('subprob2').add_input('theta')
                self._get_subsystem('subprob2').add_output('y')

                self.promotes('subprob2', ['r'])
                self.promotes('subprob2', ['theta'])
                self.promotes('subprob2', ['y'])

        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        p.model.add_subsystem('sub1', Subsys1(), promotes_inputs=['r', 'theta'],
                              promotes_outputs=['x'])
        p.model.add_subsystem('sub2', Subsys2(), promotes_inputs=['r', 'theta'],
                              promotes_outputs=['y'])
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                              promotes_outputs=['z'])

        p.setup(force_alloc_complex=True)

        p.set_val('r', 1)
        p.set_val('theta', pi)

        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_near_equal(p.get_val('z'), 1.0)

    def test_invalid_io(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('z = x**2 + y'),
                            promotes_inputs=['x', 'y'],
                            promotes_outputs=['z'])

        submodel1 = om.Group()
        submodel1.add_subsystem('sub1_ivc_r', om.IndepVarComp('r', 1.),
                                promotes_outputs=['r'])
        submodel1.add_subsystem('sub1_ivc_theta', om.IndepVarComp('theta', pi),
                                promotes_outputs=['theta'])
        submodel1.add_subsystem('subComp1', om.ExecComp('x = r*cos(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['x'])

        submodel2 = om.Group()
        submodel2.add_subsystem('sub2_ivc_r', om.IndepVarComp('r', 2),
                                promotes_outputs=['r'])
        submodel2.add_subsystem('sub2_ivc_theta', om.IndepVarComp('theta', pi/2),
                                promotes_outputs=['theta'])
        submodel2.add_subsystem('subComp2', om.ExecComp('y = r*sin(theta)'),
                                promotes_inputs=['r', 'theta'],
                                promotes_outputs=['y'])

        subprob1 = om.SubproblemComp(model=submodel1)
        subprob2 = om.SubproblemComp(model=submodel2)

        subprob1.add_input('psi')

        p.model.add_subsystem('sub1', subprob1)
        p.model.add_subsystem('sub2', subprob2)
        p.model.add_subsystem('supModel', model, promotes_inputs=['x','y'],
                              promotes_outputs=['z'])

        with self.assertRaises(Exception) as ctx:
            p.setup(force_alloc_complex=True)

    def test_multiple_setups(self):
        p = om.Problem()

        model = om.Group()
        model.add_subsystem('supComp', om.ExecComp('y = 3*x + 4'),
                            promotes_inputs=['x'], promotes_outputs=['y'])

        submodel = om.Group()
        submodel.add_subsystem('subComp', om.ExecComp('x = 6*z + 3'),
                               promotes_inputs=['z'], promotes_outputs=['x'])

        subprob = om.SubproblemComp(model=submodel)
        subprob.add_input('z')
        subprob.add_output('x')

        p.model.add_subsystem('subprob', subprob, promotes_inputs=['z'],
                              promotes_outputs=['x'])
        p.model.add_subsystem('comp', model, promotes_inputs=['x'],
                              promotes_outputs=['y'])

        p.setup(force_alloc_complex=False)
        p.setup(force_alloc_complex=False)
        p.setup(force_alloc_complex=True)

        p.set_val('z', 1)

        p.run_model()

        assert_near_equal(p.get_val('z'), 1)
        assert_near_equal(p.get_val('x'), 9)
        assert_near_equal(p.get_val('y'), 31)
