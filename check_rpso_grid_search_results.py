import pandas as pd

df = pd.read_csv("best_parameter_results.csv")
df_sorted = df.sort_values(by="Best Fitness")

smallest_10_values = df_sorted.head(10)

print(smallest_10_values)
