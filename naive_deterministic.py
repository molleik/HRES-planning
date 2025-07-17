import gc
import os

import pandas as pd
import numpy as np
import gurobipy as gp

from dispatch_model import DispatchModel
from helper_functions import generate_scenarios

with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 0)


def naive_deterministic(input_data, input_files, output_location,
                        dnf=False, fit_factor=1.2, day_threshold=0.05):
    '''
    Runs a naive deterministic model for the given input parameters.

    Parameters
    ----------
    input_data : str, dict
        Pass str if reading data directly from .xlsx file. Otherwise, data
        can be passed as dict containing DataFrames.
    input_files : str
        Path to folder where deterministic model results, to be used as inputs,
        are stored.
    output_location : str
        Path to folder where output files are to be stored.
    dnf : bool, optional
        If True, enables variable FiT scheme. The default is False.
    fit_factor : float, optional
        Factor by which FiT is multiplied during the night. The default
        is 1.2.
    day_threshold : float, optional
        Hours having a PV capacity factor higher than day_threshold are
        defined as day hours. The default is 0.05.

    Returns
    -------
    None.

    '''

    if isinstance(input_data, dict):
        data = input_data
    else:
        data = pd.read_excel(input_data, sheet_name=None)

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
    )

    VoC_data = np.zeros((num_scenarios, num_scenarios))

    for p in range(num_scenarios):

        if p == 80:
            predicted_path = os.path.join(
                input_files,
                'T = Never',
                'D_No Grid.xlsx'
            )
        else:
            predicted_path = os.path.join(
                input_files,
                f'T = {scenarios[p][0]}',
                f'P = {scenarios[p][2]}',
                f'FiT = {scenarios[p][1]}',
                f'D_T = {scenarios[p][0]}, P = {scenarios[p][2]}, '
                f'FiT = {scenarios[p][1]}.xlsx'
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
                    input_files,
                    'T = Never',
                    'D_No Grid.xlsx'
                )
            else:
                actual_path = os.path.join(
                    input_files,
                    f'T = {scenarios[a][0]}',
                    f'P = {scenarios[a][2]}',
                    f'FiT = {scenarios[a][1]}',
                    f'D_T = {scenarios[a][0]}, P = {scenarios[a][2]}, '
                    f'FiT = {scenarios[a][1]}.xlsx'
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

    VoC_df = pd.DataFrame(
        VoC_data
    )

    os.makedirs(
        os.path.join(
            output_location,
            'Naive Deterministic'
        )
    )

    output_file = os.path.join(
        output_location,
        'Naive Deterministic',
        'Naive Determinstic Results.xlsx'
    )

    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    VoC_df.to_excel(writer, sheet_name='Value of Certainty', merge_cells=False)
    writer.close()


main_dir = os.getcwd()

input_data = os.path.join(
    main_dir,
    'Results',
    'Base Case',
    'Base Case Inputs.xlsx'
)

input_files = os.path.join(
    main_dir,
    'Results',
    'Base Case',
    'Base Case Results'
)

output_location = os.path.join(
    main_dir,
    'Results',
    'Base Case',
)

naive_deterministic(input_data, input_files, output_location)
