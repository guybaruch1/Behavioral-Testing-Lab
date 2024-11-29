import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr

# Load the data from the text file into a DataFrame
# Replace 'yourfile.txt' with your actual file path
data = pd.read_csv('241124.txt', sep='\t')

# List of all 8 mice
mice = ['C1W', 'C1B', 'C1R', 'C1G', 'C2W', 'C2B', 'C2R', 'C2G']

# Initialize an empty matrix (8 mice x 5 trials)
duration_matrix = np.zeros((8, 5))

data.columns = data.columns.str.strip()

# Loop over each mouse
for i, mouse in enumerate(mice):
    # Filter the data for this specific mouse
    mouse_data = data[data['Subject ID'].str.strip() == mouse]

    # Extract the 'Duration(sec)' values for each of the 5 trials
    trial_durations = mouse_data['Duration(sec)'].values[:5]

    # Store the durations in the matrix (row corresponds to the mouse)
    duration_matrix[i, :] = trial_durations

# Plotting the data
plt.figure(figsize=(10, 6))

# Loop over each mouse and plot their trial durations
for i, mouse in enumerate(mice):
    plt.plot(range(1, 6), duration_matrix[i, :], label=mice[i])  # Plot each mouse's data

# Adding labels and title
plt.xlabel('Trial Number')  # x-axis label
plt.ylabel('Duration (sec)')  # y-axis label
plt.title('Latency to fall for Each Mouse')  # plot title

# Adding a legend to differentiate between mice
plt.legend()

plt.xticks(np.arange(1, 6, 1))

# Show the plot
plt.grid(axis='y')
plt.show()

# Calculate the mean duration for each trial (1 to 5)
mean_durations = np.mean(duration_matrix, axis=0)

# Plotting the individual mice data
plt.figure(figsize=(10, 6))

# Plot the mean of each trial
plt.plot(range(1, 6), mean_durations, label='Mean Duration', color='black', linestyle='-', marker='o', markersize=6)

# Adding labels and title
plt.xlabel('Trial Number')  # x-axis label
plt.ylabel('Duration (sec)')  # y-axis label
plt.title('Mean Latency to Fall (All Mice)')  # plot title

# Adding a legend
plt.legend()

# Ensure x-axis ticks are integers (1, 2, 3, 4, 5)
plt.xticks(np.arange(1, 6, 1))

# Show horizontal grid lines only
plt.grid(axis='y')

# Show the plot
plt.show()

file_paths = [
    './C1W_241124_new.mat',
    './C1B_241124_new.mat',
    './C1R_241124_new.mat',
    './C1G_241124_new.mat',
    './C2W_241124_new.mat',
    './C2B_241124_new.mat',
    './C2R_241124_new.mat',
    './C2G_241124_new.mat',
]

all_crossings = []

for i in range(len(file_paths)):
    try:
        data = loadmat(file_paths[i])
        if 'crossing_times' in data:
            times_crossing = np.array(data['crossing_times']).flatten()
        else:
            print(f"'crossing_times' not found in file {file_paths[i]}")
            times_crossing = np.array([])
        all_crossings.append(len(times_crossing))
    except Exception as e:
        print(f"Error processing file {file_paths[i]}: {e}")

# המרה למטריצה
all_crossings = np.array(all_crossings)

mean_durations_per_mouse = np.mean(duration_matrix, axis=1)

corr_stat, corr_p_value = pearsonr(all_crossings, mean_durations_per_mouse)

# Fit a line to the data
slope, intercept = np.polyfit(all_crossings, mean_durations_per_mouse, 1)
line = np.poly1d([slope, intercept])

# Plotting the data
plt.figure(figsize=(8, 6))
plt.scatter(all_crossings, mean_durations_per_mouse, color='blue', label='Mouse Data')

# Plot the regression line
plt.plot(all_crossings, line(all_crossings), color='red', label='Fit Line')

# Adding labels and title
plt.xlabel('Number of Crossing (All Crossing)')  # x-axis label
plt.ylabel('Mean Duration (sec)')  # y-axis label
plt.title('Pearson Correlation Between All Crossings and Mean Durations')  # plot title

# Displaying the correlation coefficient and p-value below the plot
plt.figtext(0.5, -0.1, f"Pearson r = {corr_stat:.2f}, p-value = {corr_p_value:.3f}",
            ha="center", fontsize=12, color='red')

# Show grid and plot
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to fit everything nicely
plt.show()
