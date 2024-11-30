import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# 1. Load Database
database_path = './prev data/Database_IDs.xlsx'
database = pd.read_excel(database_path)
print("Database Loaded:")
print(database.head())

# 2. Extract Data for Previous Years
previous_years_data = []
file_paths = [
    './prev data/201122_C1_blue.mat',
    './prev data/201122_C1_white.mat',
    './prev data/231122_C2_blue.mat',
    './prev data/231122_C2_green.mat',
    './prev data/111222_C2_blue.mat',
    './prev data/111222_C2_green.mat',
    './prev data/161122_C2_blue.mat',
    './prev data/161122_C2_red.mat',
    './prev data/161122_C2_white.mat'
]

for file_path in file_paths:
    data = loadmat(file_path)
    if 'crossing_times' in data:
        times_crossing = np.array(data['crossing_times']).flatten()
        total_distance = len(times_crossing)  # מספר חציות כמרחק כולל
        previous_years_data.append(total_distance)
    else:
        print(f"File {file_path} does not contain 'crossing_times'.")

print(f"Previous Years Data: {previous_years_data}")

# 3. Extract Data for Current Year
current_year_data = []
current_file_paths = [
    './C1B_241124_new.mat',
    './C1W_241124_new.mat',
    './C1R_241124_new.mat',
    './C1G_241124_new.mat',
    './C2B_241124_new.mat',
    './C2W_241124_new.mat',
    './C2R_241124_new.mat',
    './C2G_241124_new.mat'
]

for file_path in current_file_paths:
    data = loadmat(file_path)
    if 'crossing_times' in data:
        times_crossing = np.array(data['crossing_times']).flatten()
        total_distance = len(times_crossing)
        current_year_data.append(total_distance)
    else:
        print(f"File {file_path} does not contain 'crossing_times'.")

print(f"Current Year Data: {current_year_data}")

# 4. Perform Statistical Test
t_stat, p_value = ttest_ind(current_year_data, previous_years_data)
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

if p_value < 0.05:
    print("The distances this year are significantly greater than previous years.")
else:
    print("No significant difference in distances between this year and previous years.")

# 5. Visualization - Boxplot
data_to_plot = [previous_years_data, current_year_data]
labels = ['Previous Years', 'Current Year']

plt.figure(figsize=(8, 6))
plt.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True)
plt.title("Comparison of Distances Between Current Year and Previous Years")
plt.ylabel("Total Distance (Number of Crossings)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 6. Visualization - Bar Chart
# Calculate means and standard deviations
mean_previous = np.mean(previous_years_data)
std_previous = np.std(previous_years_data, ddof=1)

mean_current = np.mean(current_year_data)
std_current = np.std(current_year_data, ddof=1)

# Data for bar chart
means = [mean_previous, mean_current]
stds = [std_previous, std_current]

plt.figure(figsize=(8, 6))
plt.bar(labels, means, yerr=stds, capsize=10, color=['skyblue', 'orange'], alpha=0.8)
plt.title("Average Crossings: Current Year vs. Previous Years")
plt.ylabel("Average Number of Crossings")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()