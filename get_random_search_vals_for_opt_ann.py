import random

import numpy as np


def generate_uniform_numbers(lower_limit, upper_limit, count=10):
    if lower_limit >= upper_limit:
        raise ValueError("Lower limit must be less than upper limit")

    uniform_numbers = [random.uniform(
        lower_limit, upper_limit) for _ in range(count)]
    return uniform_numbers


count = 100


# Cp_min = [0.5255807110471932, 0.3429381734058268, 0.5473953596260367, 0.5892357456424455, 0.6079222888523057,
#           0.6344813277971112, 0.19014413636290764, 0.2117808640159182, 0.8455214965782043, 0.33172269877085303]
# Cp_max = [1.6561572229917194, 1.6942349478931538, 1.8117721144510008, 3.111438073361451, 2.488047379454802,
#           3.1738084595383205, 3.249682100682639, 2.2498771078645934, 3.026638675276517, 2.791080482283482]
# Cg_min = [0.24866719100172174, 0.1063992296063363, 0.6399466405615443, 0.7789907590032242, 0.7714898739594362,
#           0.11531231543022723, 0.35601705530224337, 0.2657472414099825, 0.5060031362324593, 0.17612180788459356]
# Cg_max = [3.401729764116632, 2.7656770286923624, 1.7226611789385775, 1.9783997678963696, 1.563962743441281,
#           2.8485602766725373, 2.6324175260968463, 1.884195354440522, 3.0420218358982307, 1.845946114117474]
# w_min = [0.3277489076256416, 0.10323507627270086, 0.6903954593094627, 0.2512192466357657, 0.31828822621491576,
#          0.23769350357392033, 0.3592722860920172, 0.13808108256010332, 0.5815967759592018, 0.5758491828008468]
# w_max = [1.090799879141887, 1.2011811660684617, 1.0519962824603226, 1.4027237055762312, 1.3946578910060003,
#          1.90350611469437, 1.245797074484059, 0.5383041243044694, 0.6896131233011519, 1.8905367686610268]
# gwn_std_dev = [0.06427594732038355, 0.09711193898395054, 0.18030623089238926, 0.18926183363254914, 0.17275962871453357,
#                0.026820486300555134, 0.08580276947147242, 0.19679651652191496, 0.07150807341695263, 0.1315206414296331]

Cp_min = [0.1, 0.5]
Cp_max = [2.5, 3.5]
Cg_min = [0.5, 0.9]
Cg_max = [2.5, 3.5]
w_min = [0.3, 0.4]
w_max = [0.9, 1.7]
gwn_std_dev = [0.07, 0.15]

random_Cp_min = generate_uniform_numbers(Cp_min[0], Cp_min[1], count)
random_Cp_max = generate_uniform_numbers(Cp_max[0], Cp_max[1], count)
random_Cg_min = generate_uniform_numbers(Cg_min[0], Cg_min[1], count)
random_Cg_max = generate_uniform_numbers(Cg_max[0], Cg_max[1], count)
random_w_min = generate_uniform_numbers(w_min[0], w_min[1], count)
random_w_max = generate_uniform_numbers(w_max[0], w_max[1], count)
random_gwn_std_dev = generate_uniform_numbers(
    gwn_std_dev[0], gwn_std_dev[1], count)


# print(random_Cp_min)
# print(random_Cp_max)
# print(random_Cg_min)
# print(random_Cg_max)
# print(random_w_min)
# print(random_w_max)
# print(random_gwn_std_dev)

# c1 = [1.49445, 2.0]
# c2 = [1.49445, 2.0]
# inertia = [0.729, 0.8]

# random_c1 = generate_uniform_numbers(c1[0], c1[1], count)
# random_c2 = generate_uniform_numbers(c2[0], c2[1], count)
# random_inertia = generate_uniform_numbers(inertia[0], inertia[1], count)

print(random_Cp_min)
print("--------------------------------")
print(random_Cp_max)
print("--------------------------------")
print(random_Cg_min)
print("--------------------------------")
print(random_Cg_max)
print("--------------------------------")
print(random_w_min)
print("--------------------------------")
print(random_w_max)
print("--------------------------------")
print(random_gwn_std_dev)
