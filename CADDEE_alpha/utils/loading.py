import csdl_alpha as csdl


def load_var(var : csdl.Variable, var_dict : dict={}, dv_flag: bool=True, lower=None, upper=None, scaler=None, scale_by_value=False):
    if var.name in var_dict:
        var.value = var_dict[var.name]
        if scale_by_value:
            scaler_input = 1 / var_dict[var.name]
        else:
            scaler_input = scaler
    if scale_by_value is False:
        scaler_input=scaler

    if dv_flag:
        var.set_as_design_variable(upper=upper, lower=lower, scaler=scaler_input)
