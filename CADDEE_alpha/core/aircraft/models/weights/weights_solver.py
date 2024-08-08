import csdl_alpha as csdl


class WeightsSolverModel:
    def evaluate(self, gross_weight_guess : csdl.ImplicitVariable, *component_weights):
        csdl.check_parameter(gross_weight_guess, "gross_weight_guess", types=(csdl.ImplicitVariable, csdl.Variable))
        gross_weight = csdl.Variable(shape=(1, ), value=0)

        for weight in component_weights:
            gross_weight =  gross_weight +  weight


        weight_residual = gross_weight_guess - gross_weight

        solver = csdl.nonlinear_solvers.GaussSeidel()
        solver.add_state(gross_weight_guess, weight_residual)
        solver.run()
