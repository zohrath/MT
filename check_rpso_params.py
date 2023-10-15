import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("best_parameter_results.csv")

# Sort the DataFrame by the "Best Fitness" column in ascending order
df_sorted = df.sort_values(by="Best Fitness")

# Get the smallest 10 values from the "Best Fitness" column
smallest_10_values = df_sorted.head(10)

# Display the result
print(smallest_10_values)
