# Should contain all chosen values for c1, c2, w when using PSO to optimize the weights of the Adam optimizer
def get_vals_pso_opt_adam_params():
    random_c1_values = [
        1.88411,
        1.95259,
        1.61497,
        1.56401,
        1.70871,
        1.70053,
        1.90821,
        1.58269,
        1.8663,
        1.7258,
    ]

    random_c2_values = [
        1.67804,
        1.80616,
        1.68749,
        1.94016,
        1.54662,
        1.89665,
        1.69339,
        1.9903,
        1.86793,
        1.79575,
    ]

    random_w_values = [
        0.838,
        0.887,
        0.56,
        0.743,
        0.515,
        0.978,
        0.814,
        0.771,
        0.612,
        0.579,
    ]

    return random_c1_values, random_c2_values, random_w_values
