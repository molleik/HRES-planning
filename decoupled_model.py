import os
import gc

import pandas as pd
import numpy as np
import helper_functions as fnc

from deterministic_model import DeterministicModel
from stochastic_model import StochasticModel


def decoupled_model(input_data, output_location,
                    dnf=False, fit_factor=1.2, day_threshold=0.05):
    '''
    Runs a decoupled (heuristic) model for the given input parameters.

    Parameters
    ----------
    input_data : str, dict
        Pass str if reading data directly from .xlsx file. Otherwise, data
        can be passed as dict containing DataFrames.
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

    save_dir = os.path.join(
        output_location,
        (output_location.split('\\')[-1] + ' Decoupled Model')
    )

    os.makedirs(save_dir, exist_ok=True)

    fnc.make_folders(save_dir, [t for t in range(1, 11)], [0.03, 0.08, 0.15],
                     [0.1, 0.185, 0.27]
                     )

    if isinstance(input_data, dict):
        data = input_data
    else:
        data = pd.read_excel(input_data, sheet_name=None)

    # Parameters
    parameters = data['General Parameters']
    Y = int(data['Time Frame']['Years'].iloc[0])
    grid_start = int(parameters['Tgrid Start'].iloc[0])
    grid_end = int(parameters['Tgrid End'].iloc[0])
    T = [x for x in range(grid_start, grid_end + 1)]
    FiT = parameters['FiTs'].dropna()
    P = parameters['EDLTars']

    scenarios = fnc.generate_scenarios(T, FiT, P)
    num_scenarios = len(scenarios)

    scenario_multindex = pd.MultiIndex.from_tuples(
        scenarios
    )

    npvs = np.zeros((num_scenarios))

    # Scenarios Tree
    deterministic_model = DeterministicModel(
        input_data=data,
    )

    no_grid_model = StochasticModel(
        input_data=data,
    )

    no_grid_model.input_scenarios(
        ys=0,
        scenarios=[(Y, 0, 0)],
        probabilities=[1],
        day_night_fit=dnf,
        fit_factor=fit_factor,
        day_threshold=day_threshold
    )

    no_grid_model.stoch_solve()

    for scenario in scenarios:

        T = scenario[0]

        no_grid_output = {}
        no_grid_model.stoch_store_decisions(
            no_grid_output,
            yl=T
        )

        deterministic_model.input_scenario(
            ys=T,
            scenario=scenario,
            day_night_fit=dnf,
            fit_factor=fit_factor,
            day_threshold=day_threshold
        )

        deterministic_model.input_installations(no_grid_output)

        deterministic_model.det_solve()

        det_output = {}
        deterministic_model.det_store_decisions(det_output)

        scenario_index = scenarios.index(scenario)

        npvs[scenario_index] = (
            no_grid_output['Total NPV']['Total Value'].loc[0]
            +
            det_output['Total NPV']['Total Value'].loc[0]
        )

        out_path = os.path.join(
            save_dir,
            f'T = {T}',
            f'P = {scenario[2]}',
            f'FiT = {scenario[1]}'
        )

        fnc.sd_merge(no_grid_output, det_output, out_path,
                     decoupled=True)

        gc.collect()

    case = output_location.split('\\')[-1]

    det_npvs = pd.read_excel(
        os.path.join(
            output_location,
            'Receding Horizon',
            'Summary',
            'NPV_Summary.xlsx'
        ),
        sheet_name='Deterministic',
        index_col=[0, 1, 2]
    ).values.flatten()[:80]

    new_voc = ((npvs - det_npvs)/det_npvs)*100

    writer = pd.ExcelWriter(
        os.path.join(
            output_location,
            save_dir,
            'Decoupled Model.xlsx'
        ),
        engine='xlsxwriter'
    )
    to_write = pd.DataFrame(
        new_voc,
        index=scenario_multindex
    )

    to_write.to_excel(writer, merge_cells=False)
    writer.close()


main_dir = os.getcwd()

input_data = pd.read_excel(
    os.path.join(
        main_dir,
        'Results',
        'Base Case',
        'Base Case Inputs.xlsx'
    ),
    sheet_name=None
)

output_location = os.path.join(
    main_dir,
    'Results',
    'Base Case'
)

decoupled_model(
    input_data,
    output_location
)
