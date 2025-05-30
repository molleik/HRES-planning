import gc
import os
import time

import pandas as pd
import numpy as np
import gurobipy as gp

from dispatch_model import DispatchModel

with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 0)


def generate_scenarios(T, FiT, P):

    scenarios = []

    for tgrid in T:
        for fit in FiT:
            for tar in P:
                if fit < tar:
                    scenarios.append((tgrid, fit, tar))

    return scenarios


main_dir = 'Base Case'
results_dir = os.path.join(
    main_dir,
    'Base Case Results'
)

data = pd.read_excel(
    os.path.join(
        main_dir,
        'Base Case Inputs.xlsx',
    ),
    sheet_name=None
)

parameters = data['General Parameters']
Y = int(data['Time Frame']['Years'].iloc[0])
grid_start = int(parameters['Tgrid Start'].iloc[0])
grid_end = int(parameters['Tgrid End'].iloc[0])
T = [x for x in range(grid_start, grid_end + 1)]
FiT = parameters['FiTs'].dropna()
P = parameters['EDLTars']

scenarios = generate_scenarios(T, FiT, P)
scenarios.append((Y, 0, 0))
num_scenarios = len(scenarios)

dispmodel = DispatchModel(
    data,
    input_as_dict=True
)

VoC_data = np.zeros(num_scenarios, num_scenarios)

for p in range(num_scenarios):

    if p == 80:
        predicted_path = os.path.join(
            results_dir,
            'T = Never',
            'D_No Grid.xlsx'
        )
    else:
        predicted_path = os.path.join(
            results_dir,
            f'T = {scenarios[p][0]}',
            f'P = {scenarios[p][2]}',
            f'FiT = {scenarios[p][1]}',
            f'D_T = {scenarios[p][0]}, P = {scenarios[p][2]},'
            'FiT = {scenarios[p][1]}.xlsx'
        )

    predicted = pd.read_excel(
        predicted_path,
        sheet_name=None,
        index_col=[0]
    )

    dispmodel.input_installations(
        predicted
    )

    for a in range(num_scenarios):
        dispmodel.input_scenario(
            ys=0,
            scenario=scenarios[a]
        )

        dispmodel.disp_solve()

        actual = {}
        dispmodel.disp_store_decisions(
            actual
        )

        if a == 80:
            actual_path = os.path.join(
                results_dir,
                'T = Never',
                'D_No Grid.xlsx'
            )
        else:
            actual_path = os.path.join(
                results_dir,
                f'T = {scenarios[a][0]}',
                f'P = {scenarios[a][2]}',
                f'FiT = {scenarios[a][1]}',
                f'D_T = {scenarios[a][0]}, P = {scenarios[a][2]},'
                'FiT = {scenarios[a][1]}.xlsx'
            )

        det_npv = pd.read_excel(
            actual_path,
            sheet_name='Total NPV',
            index_col=[0]
        )

        voc = (
            (
                actual['Total NPV']['Total Value'][0]
                - det_npv['Total Value'][0]
            )
            / abs(actual['Total NPV']['Total Value'][0])
        )

        VoC_data[p, a] = voc

    gc.collect()


# %%
VoC_df = pd.DataFrame(
    VoC_data
)
out_dir = r'Base Case\Naive Deterministic'

output_path = os.path.join(
    out_dir,
    'Naive Deterministic Results.xlsx',
)

writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
VoC_df.to_excel(writer, sheet_name='Value of Certainty', merge_cells=False)
writer.close()
