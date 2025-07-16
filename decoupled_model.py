import os
import gc

import pandas as pd
import numpy as np

from deterministic_model import DeterministicModel
from stochastic_model import StochasticModel
from createFolders import make_folders
from merge_functions import certainMerge


def generate_scenarios(T, FiT, P):

    scenarios = []

    for tgrid in T:
        for fit in FiT:
            for tar in P:
                if fit < tar:
                    scenarios.append((tgrid, fit, tar))

    return scenarios


def print_excel(results, file_path, model_type):
    import pandas as pd

    model_name = model_type + '_' + results['Model Name'] + '.xlsx'

    file_name = os.path.join(
        file_path,
        model_name
    )

    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for key in results:
        if type(results[key]) is pd.core.frame.DataFrame:
            results[key].to_excel(
                writer,
                sheet_name=key,
                merge_cells=False
            )

    writer.close()


def decoupled_model(input_data, out_dir,
                    data_as_dict=False,
                    dnf=False, fit_factor=1.2, day_threshold=0.05):

    save_dir = os.path.join(
        out_dir,
        (out_dir.split('\\')[-1] + ' Decoupled Model')
    )

    make_folders(save_dir, [t for t in range(1, 11)], [0.03, 0.08, 0.15],
                 [0.035, 0.185, 0.27]
                 )

    if data_as_dict:
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

    scenarios = generate_scenarios(T, FiT, P)
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

        to_print = certainMerge(no_grid_output, det_output, out_path,
                                decoupled=True)

        del to_print

        # print_excel(det_output, out_path, 'D')

        gc.collect()

    case = out_dir.split('\\')[-1]

    det_npvs = pd.read_excel(
        os.path.join(
            out_dir,
            f'{case} Results',
            'Summary',
            'NPV_Summary.xlsx'
        ),
        sheet_name='Deterministic',
        index_col=[0, 1, 2]
    ).values.flatten()[:70]

    new_voc = ((npvs - det_npvs)/det_npvs)*100

    writer = pd.ExcelWriter(
        os.path.join(
            out_dir,
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


output_location = (
    r'C:\Users\User\OneDrive - American University of Beirut'
    r'\Decentralized Planning Grid Uncertainty\Codes - Kareem\GitHub\Low Price'
)

input_data = pd.read_excel(
    os.path.join(
        output_location,
        'Low Price Inputs.xlsx'
    ),
    sheet_name=None
)

decoupled_model(input_data, output_location, data_as_dict=True)

# det_npvs = pd.read_excel(
#     os.path.join(
#         output_location,
#         'VOLL 0% Discount',
#         'VOLL = 0.0007',
#         'Summary',
#         'NPV_Summary.xlsx'
#     ),
#     sheet_name='Deterministic',
#     index_col=[0, 1, 2]
# ).values.flatten()[:80]

# rh_voc = pd.read_excel(
#     os.path.join(
#         output_location,
#         'VOLL 0% Discount',
#         'VOLL = 0.0007',
#         'Summary',
#         'NPV_Summary.xlsx'
#     ),
#     sheet_name='Value of Certainty',
#     index_col=[0, 1, 2]
# ).values.flatten()[:80]*100

# new_voc = ((npvs - det_npvs)/det_npvs)*100

# writer = pd.ExcelWriter('PseudoRH 0% Discount.xlsx', engine='xlsxwriter')
# to_write = pd.DataFrame(new_voc)
# to_write.to_excel(writer, merge_cells=False)
# writer.close()
