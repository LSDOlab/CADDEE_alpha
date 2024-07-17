import csdl_alpha as csdl


def load_var(var : csdl.Variable, var_dict : dict={}, dv_flag: bool=True, lower=None, upper=None, scaler=None):
    if var.name in var_dict:
        var.value = var_dict[var.name]

    if dv_flag:
        var.set_as_design_variable(upper=upper, lower=lower, scaler=scaler)
