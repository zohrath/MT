import random


num_values = 100

# Generate random values between 0.05 and 0.4
uniform_values = [random.uniform(0.04, 0.8) for _ in range(num_values)]

print(uniform_values)
