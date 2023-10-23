def generate_array(weight_bound, bias_bound):
    array = []

    for _ in range(3):
        for _ in range(36):
            array.append((-weight_bound, weight_bound))
        for _ in range(6):
            array.append((-bias_bound, bias_bound))
    for _ in range(12):
        array.append((-weight_bound, weight_bound))
    for _ in range(2):
        array.append((-bias_bound, bias_bound))

    return array


position_bounds_opt_ann = generate_array(4.0, 2.0)
velocity_bounds_opt_ann = generate_array(0.2, 0.1)
