# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:53:37 2025

@author: User
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

main_dir = os.getcwd()

volls = [0.02, 0.06, 0.17]

fig, ax = plt.subplots(1, 1)

colors = ['lightblue', 'darkblue']
for i in range(len(volls)):
    colors += colors

to_plot = []

for voll in volls:

    to_plot.append(pd.read_excel(
        os.path.join(
            main_dir,
            f'VOLL = {voll}',
            f'VOLL = {voll} Results',
            'Summary',
            'NPV_Summary.xlsx'
        ),
        sheet_name='Value of Certainty',
        index_col=[0, 1, 2]
    ).values.flatten()*100)

    to_plot.append(pd.read_excel(
        os.path.join(
            main_dir,
            f'VOLL = {voll}',
            f'VOLL = {voll} Decoupled Model',
            'Decoupled Model.xlsx'
        ),
        sheet_name='Sheet1',
        index_col=[0, 1, 2]
    ).values.flatten())

    # nd_voc = pd.read_excel(
    #     os.path.join(
    #         main_dir,
    #         f'VOLL = {voll}',
    #         f'Naive Deterministic',
    #         'Naive Deterministic Results.xlsx'
    #     ),
    #     sheet_name='Value of Certainty',
    #     index_col=[0]
    # ).loc[47].values.flatten()*100

to_plot.append(pd.read_excel(
    os.path.join(
        main_dir,
        f'Base Case',
        f'Base Case Results',
        'Summary',
        'NPV_Summary.xlsx'
    ),
    sheet_name='Value of Certainty',
    index_col=[0, 1, 2]
).values.flatten()*100)

to_plot.append(pd.read_excel(
    os.path.join(
        main_dir,
        f'Base Case',
        f'Base Case Decoupled Model',
        'Decoupled Model.xlsx'
    ),
    sheet_name='Sheet1',
    index_col=[0, 1, 2]
).values.flatten())

bplot = ax.boxplot(
    to_plot,
    patch_artist=True
)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

tech = ['PV', 'Wind', 'Diesel']
for x in range(len(volls)):
    ax.axvline(2.5+2*x, linestyle='--', color='black')
    ax.text(2.4+2*x, 0.175,
            f'{tech[x]} LCOE',
            bbox={'x': x, 'color': 'yellow'},
            rotation=90
            )

custom_lines = [Line2D([0], [0], color='lightblue', lw=4),
                Line2D([0], [0], color='darkblue', lw=4),
                ]

ax.legend(
    custom_lines,
    [
        'Receding Horizon',
        'Decoupled'
    ]
)

labels = [f'VOLL \n = {voll}' for voll in volls] + ['VOLL \n = 0.7']

ticks = (
    [2*x+1.5 for x in range(len(volls)+1)]
)

# ticks.sort()

ax.set_xticks(
    ticks,
    labels,
    rotation=90,
    fontsize=10
)


plt.show()
