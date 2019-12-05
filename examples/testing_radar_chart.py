import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['A', 'B', 'C', 'D'],
    'var1': [38, 1.5, 30, 4],
    'var2': [29, 10, 9, 34],
    'var3': [8, 39, 23, 24],
    'var4': [7, 31, 33, 14],
    'var5': [28, 15, 32, 14]
})

title = 'Titulo'
radial_labels = ["var 1", "var 2", "var 3", "var 4", "var 5"]
case_data = [[38, 29,  8, 7, 28],
             [1.5, 10, 39, 31, 15]]
case_labels = ["A", "B"]  #, "C", "D"]


# number of variable
categories = radial_labels
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)

# ------- PART 2: Add plots

# Plot each individual = each line of the data
for i, cl in enumerate(case_labels):
    values = case_data[i]  # df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    print(angles)
    print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="group " + cl)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
