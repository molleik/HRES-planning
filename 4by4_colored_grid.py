import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlabel('VOLL', fontsize=19)
ax.set_ylabel('Electricity Price P', fontsize=19)

y_labels = ['', 'PV LCOE', 'Wind LCOE', 'Diesel LCOE']
x_labels = ['', 'PV LCOE', 'Wind LCOE', 'Diesel LCOE']

ax.set_xticks(np.arange(0, 4, 1))
ax.set_yticks(np.arange(0, 4, 1))
ax.set_xticklabels(x_labels, fontsize=13)
ax.set_yticklabels(y_labels, fontsize=13, rotation=90, va='center')

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

for x in range(5):
    ax.axvline(x, color='black', linestyle='-', linewidth=1)
for y in range(5):
    ax.axhline(y, color='black', linestyle='-', linewidth=1)

grid_text = [
    [
        ['PV ✘',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✔',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✔',
            'Diesel ✔',
         ]
    ],
    [
        ['PV ✘',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✔',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✔',
            'Diesel ?',
         ]
    ],
    [
        ['PV ✘',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ?',
            'Diesel ✘',
         ],
        ['PV ✔',
            'Wind ?',
            'Diesel ?',
         ]
    ],
    [
        ['PV ✘',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ?',
            'Wind ✘',
            'Diesel ✘',
         ],
        ['PV ?',
            'Wind ?',
            'Diesel ✘',
         ],
        ['PV ?',
            'Wind ?',
            'Diesel ?',
         ]
    ]
]

for row in range(4):
    for col in range(4):
        if (row + col) <= 3:
            color = 'green'
        elif (row + col) <= 4:
            color = 'yellow'
        elif (row + col) <= 5:
            color = 'orange'
        elif row == 3 and col == 3:
            color = 'red'
        else:
            color = 'white'

        rect = patches.Rectangle((col, 3-row), 1, 1, linewidth=1,
                                 edgecolor='black', facecolor=color, alpha=0.5)
        ax.add_patch(rect)

        for line_num in range(3):
            ax.text(col + 0.5, 3-row + 0.7 - line_num*0.2,
                    grid_text[row][col][line_num],
                    ha='center', va='center', fontsize=15)

plt.tight_layout()
plt.show()
