import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import sem

# List of all uploaded files for mice
file_paths = [
    './C1B_241124_new.mat',
    './C1W_241124_new.mat',
    './C1R_241124_new.mat',
    './C1G_241124_new.mat',
    './C2B_241124_new.mat',
    './C2W_241124_new.mat',
    './C2G_241124_new.mat',
]

# Placeholder for Grooming and Thigmotaxis data
grooming_data = []
bin_size = 150

# Load all MAT files
for file_path in file_paths:
    data = loadmat(file_path)
    # Extract Grooming and Thigmotaxis (assuming keys 'Grooming' and 'Thigmotaxis')
    grooming_data.append(data.get('grooming_start_stop'))

for file_path in file_paths:
    data = loadmat(file_path)
    print(f"Keys in file {file_path}: {list(data.keys())}")

# Ensure data is valid
grooming_data = [g if g is not None else np.zeros((600,)) for g in grooming_data]

# Convert to numpy arrays
# Convert grooming intervals to relative time in each second
max_length = 600  # סך הכל 600 שניות
grooming_data_fixed = []

for entry in grooming_data:
    # בדוק אם `entry` תקין
    if entry is None or len(entry) == 0:
        grooming_data_fixed.append(np.zeros(max_length))  # נתונים ריקים -> וקטור של אפסים
    else:
        # צרו מערך של 600 שניות והגדירו את האינטרוולים כזמן יחסי
        fixed_entry = np.zeros(max_length)
        for interval in entry.T:  # מניחים ש-`entry` הוא בצורת (2, N)
            start, stop = interval[0], interval[1]

            # חשב את הגבולות השבריים
            start_sec = int(start)
            stop_sec = int(stop)

            if start_sec == stop_sec:
                # אם הכל באותה שנייה, הוסף את החלק היחסי
                fixed_entry[start_sec] += stop - start
            else:
                # הוסף את החלק היחסי של השנייה הראשונה
                fixed_entry[start_sec] += 1 - (start - start_sec)

                # הוסף את החלק היחסי של השנייה האחרונה
                fixed_entry[stop_sec] += stop - stop_sec

                # הוסף ערך שלם לכל השניות המלאות שבין הגבולות
                if stop_sec > start_sec + 1:
                    fixed_entry[start_sec + 1:stop_sec] += 1

        grooming_data_fixed.append(fixed_entry)

# המרה למערך numpy
grooming_data = np.array(grooming_data_fixed)

# Define bin size and compute means per bin
bin_size = 150  # כל אינטרוול הוא 150 שניות (2.5 דקות)
num_bins = grooming_data.shape[1] // bin_size

# סכום ערכים בכל אינטרוול לכל עכבר
grooming_sums = grooming_data.reshape(grooming_data.shape[0], num_bins, bin_size).sum(axis=2)

# חישוב ממוצע ערכים בכל אינטרוול עבור כל עכבר
grooming_binned_means = grooming_sums / len(file_paths)

# ממוצע סופי על פני כל העכברים עבור כל אינטרוול
avg_grooming = np.sum(grooming_binned_means, axis=0)

# שגיאה תקנית (SEM) על פני כל העכברים עבור כל אינטרוול
sem_grooming = sem(grooming_binned_means, axis=0)

# הדפסת תוצאות לבדיקה
# print("Grooming Binned Means (Per Mouse):", grooming_binned_means)
# print("Average Grooming (Across Mice):", avg_grooming)
# print("SEM Grooming (Across Mice):", sem_grooming)

# גרף עמודות: Grooming ממוצע
plt.figure(figsize=(8, 5))
time_labels = [f"Bin {i + 1}" for i in range(num_bins)]

# גרף עמודות עם שגיאה תקנית
plt.bar(time_labels, avg_grooming, yerr=sem_grooming, capsize=5, color='skyblue', label='Mean ± SEM')

plt.title("Average Grooming Across All Mice (4 Time Bins)")
plt.xlabel("Time Bins (2.5 min each)")
plt.ylabel("Grooming Time (sec)")
plt.grid(axis='y')  # קווי עזר על ציר ה-Y בלבד
plt.legend()
plt.tight_layout()
plt.show()

# גרף 2: Grooming של כל עכבר באינטרוולים
mice_labels = ['C1B', 'C1W', 'C1R', 'C1G', 'C2B', 'C2W', 'C2G']

# Plotting each mouse's grooming data with custom labels
plt.figure(figsize=(10, 6))
for i in range(len(grooming_binned_means)):
    plt.plot(time_labels, grooming_binned_means[i], label=mice_labels[i])

plt.title("Individual Grooming Over Time Bins")
plt.xlabel("Time Bins (2.5 min each)")
plt.ylabel("Grooming Time (sec)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def calculate_thigmotaxis(times_crossing, times_periphery, bin_size, total_duration):
    """
    מחשבת את Thigmotaxis עבור עכבר אחד.
    times_crossing: רשימת זמנים של כלל החציות
    times_periphery: רשימת זמנים של החציות בפריפריה
    bin_size: גודל בין (ביחידות שניות)
    total_duration: זמן כולל של הניסוי (בשניות)
    """
    num_bins = total_duration // bin_size
    thigmotaxis_ratios = []

    for i in range(num_bins):
        # חשב את גבולות הבין
        start = i * bin_size
        end = start + bin_size

        # ספירת חציות בפריפריה ובמרכז
        crossings_in_bin = np.sum((times_crossing >= start) & (times_crossing < end))
        periphery_in_bin = np.sum((times_periphery >= start) & (times_periphery < end))

        # חישוב יחס (Thigmotaxis)
        if crossings_in_bin > 0:  # הימנע מחלוקה באפס
            ratio = periphery_in_bin / crossings_in_bin
        else:
            ratio = 0  # אם אין חציות בבין הזה, היחס הוא 0

        thigmotaxis_ratios.append(ratio)

    return thigmotaxis_ratios


# דוגמה לשימוש בפונקציה עבור כל עכבר
bin_size = 150  # 2.5 דקות
total_duration = 600  # זמן כולל של הניסוי בשניות
all_thigmotaxis = []

for i in range(len(file_paths)):
    try:
        data = loadmat(file_paths[i])
        if 'crossing_times' in data:
            times_crossing = np.array(data['crossing_times']).flatten()
        else:
            print(f"'crossing_times' not found in file {file_paths[i]}")
            times_crossing = np.array([])

        if 'periphery_times' in data:
            times_periphery = np.array(data['periphery_times']).flatten()
        else:
            print(f"'periphery_times' not found in file {file_paths[i]}")
            times_periphery = np.array([])

        thigmotaxis_ratios = calculate_thigmotaxis(times_crossing, times_periphery, bin_size, total_duration)
        all_thigmotaxis.append(thigmotaxis_ratios)
    except Exception as e:
        print(f"Error processing file {file_paths[i]}: {e}")

# המרה למטריצה
all_thigmotaxis = np.array(all_thigmotaxis)

# הדפסת תוצאות
print("Thigmotaxis Ratios (Per Mouse):", all_thigmotaxis)

# חישוב ממוצעים ושגיאות תקניות לכל בין
thigmotaxis_binned_means = np.mean(all_thigmotaxis, axis=0)
thigmotaxis_binned_sems = sem(all_thigmotaxis, axis=0)
thigmotaxis_binned_means *= 100
thigmotaxis_binned_sems *= 100

# תוויות הבינים
time_labels = [f"Bin {i + 1}" for i in range(len(thigmotaxis_binned_means))]

# גרף עמודות להצגת Thigmotaxis ממוצע
plt.figure(figsize=(8, 5))

# גרף עמודות עם שגיאה תקנית
plt.bar(time_labels, thigmotaxis_binned_means, yerr=thigmotaxis_binned_sems, capsize=5, color='lightgreen',
        label='Mean ± SEM')

# עיצוב הגרף
plt.title("Average Thigmotaxis Across All Mice (Time Bins)")
plt.xlabel("Time Bins (2.5 min each)")
plt.ylabel("Thigmotaxis Ratio (%)")
plt.grid(axis='y')  # קווי עזר על ציר ה-Y בלבד
plt.legend()
plt.tight_layout()

# הצגת הגרף
plt.show()

# גרף 2: Thigmotaxis לכל עכבר בנפרד
plt.figure(figsize=(10, 6))
all_thigmotaxis *= 100  # to fix percentage
for i, thigmotaxis_ratios in enumerate(all_thigmotaxis):
    plt.plot(time_labels, thigmotaxis_ratios, label=mice_labels[i])

# עיצוב הגרף
plt.title("Individual Thigmotaxis Over Time Bins")
plt.xlabel("Time Bins (2.5 min each)")
plt.ylabel("Thigmotaxis Ratio (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# הצגת הגרף
plt.show()
