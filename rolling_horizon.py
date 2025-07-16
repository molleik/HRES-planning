import gc
import os

import pandas as pd
import numpy as np
import merge_functions as mg

from copy import deepcopy
from deterministic_model import DeterministicModel
from stochastic_model import StochasticModel


def generate_scenarios(T, FiT, P):

    scenarios = []

    for tgrid in T:
        for fit in FiT:
            for tar in P:
                if fit < tar:
                    scenarios.append((tgrid, fit, tar))

    return scenarios


def make_folders(directory, Tgrids, FiTs, EDLTars):

    if not os.path.exists(directory):
        os.makedirs(directory)

    for tgrid in Tgrids:
        for edltar in EDLTars:
            for fit in FiTs:
                if fit < edltar:

                    fileFolder = os.path.join(
                        directory,
                        f'T = {tgrid}',
                        f'P = {edltar}',
                        f'FiT = {fit}'
                    )

                    if not os.path.exists(fileFolder):
                        os.makedirs(fileFolder)

    summaryFolder = os.path.join(directory, 'Summary')
    if not os.path.exists(summaryFolder):
        os.makedirs(summaryFolder)

    noneTgridFolder = os.path.join(directory, 'T = Never')
    if not os.path.exists(noneTgridFolder):
        os.makedirs(noneTgridFolder)

    stochasticFolder = os.path.join(directory, 'Stochastic')
    if not os.path.exists(stochasticFolder):
        os.makedirs(stochasticFolder)


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


def rolling_horizon(input_data, output_location,
                    data_as_dict=False,
                    no_grid_scenario=False, dnf=False,
                    fit_factor=1.2, day_threshold=0.05):

    rolling_horizon_output = {}
    stochastic_output = {}
    deterministic_output = {}
    decisions_so_far = {}

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

    make_folders(output_location, T, FiT, P)

    scenario_index = 0
    scenarios = generate_scenarios(T, FiT, P)
    if no_grid_scenario:
        scenarios.append((Y, 0, 0))
    scenarios_copy = deepcopy(scenarios)
    num_scenarios = len(scenarios)

    # probabilities = data['Probabilities']['Probabilities']
    probabilities = [1/num_scenarios for n in range(num_scenarios)]

    deterministic_npvs = np.zeros((num_scenarios))
    rolling_horizon_npvs = np.zeros((num_scenarios))

    # Initial Solve
    stochastic_model = StochasticModel(
        input_data=data,
    )

    stochastic_model.input_scenarios(
        scenarios,
        probabilities,
        day_night_fit=dnf,
        fit_factor=fit_factor,
        day_threshold=day_threshold
    )

    stochastic_model.stoch_solve()

    stochastic_model.stoch_store_decisions(
        stochastic_output
    )

    stoc_path = os.path.join(
        output_location,
        'Stochastic'
    )

    print_excel(
        stochastic_output,
        stoc_path,
        'S'
    )

    stochastic_model.stoch_store_decisions(
        decisions_so_far,
        grid_start
    )

    stochastic_npvs = stochastic_model.NPVC

    gc.collect()

    # Scenarios Tree

    deterministic_model = DeterministicModel(
        input_data=data,
    )

    for ys in range(grid_start, grid_end+1):

        # Deterministic scenarios
        scenarios_this_year = 0

        grid_year = scenarios[scenarios_this_year][0]

        while scenarios[scenarios_this_year][0] == grid_year:

            # Rolling Horizon
            deterministic_model.input_scenario(
                ys=ys,
                scenario=scenarios[scenarios_this_year],
                day_night_fit=dnf,
                fit_factor=fit_factor,
                day_threshold=day_threshold
            )

            det_path = os.path.join(
                output_location,
                ('T = ' + str(scenarios[scenarios_this_year][0])),
                ('P = ' + str(scenarios[scenarios_this_year][2])),
                ('FiT = ' + str(scenarios[scenarios_this_year][1])),
            )

            deterministic_model.input_installations(
                decisions_so_far
            )

            deterministic_model.det_solve()

            deterministic_model.det_store_decisions(
                rolling_horizon_output
            )

            path_result = mg.certainMerge(
                decisions_so_far,
                rolling_horizon_output,
                det_path
            )

            rolling_horizon_npvs[scenario_index] = (
                path_result['Total NPV']['Total Value']
            )

            # Deterministic
            deterministic_model.input_scenario(
                ys=0,
                scenario=scenarios[scenarios_this_year],
                day_night_fit=dnf,
                fit_factor=fit_factor,
                day_threshold=day_threshold
            )

            deterministic_model.det_solve()

            deterministic_model.det_store_decisions(
                deterministic_output
            )
            print_excel(
                deterministic_output,
                det_path,
                'D'
            )

            deterministic_npvs[scenario_index] = (
                deterministic_output['Total NPV']['Total Value']
            )

            scenario_index += 1

            gc.collect()

            if scenarios_this_year < len(scenarios) - 1:
                scenarios_this_year += 1
            else:
                break

        # Probabilities Update
        transfer_prob = (
            sum(probabilities[:scenarios_this_year])
            / (
                len(scenarios) - (scenarios_this_year)
            )
        )

        scenarios = scenarios[scenarios_this_year:]
        probabilities = probabilities[scenarios_this_year:]

        for p in range(len(probabilities)):
            probabilities[p] += transfer_prob

        # Stochastic Re-solve
        if ys < grid_end:
            stochastic_model.input_scenarios(
                ys=ys,
                scenarios=scenarios,
                probabilities=probabilities,
                day_night_fit=dnf,
                fit_factor=fit_factor,
                day_threshold=day_threshold
            )

            stochastic_model.input_installations(
                decisions_so_far
            )

            stochastic_model.stoch_solve()

            stochastic_model.stoch_store_decisions(
                stochastic_output,
                yl=ys + 1
            )

            mg.uncertainMerge(
                decisions_so_far,
                stochastic_output
            )

        gc.collect()

    if no_grid_scenario:
        # No Grid scenarios
        no_grid_path = os.path.join(
            output_location,
            'T = Never'
        )

        deterministic_model.input_scenario(
            ys=grid_end,
            scenario=scenarios[0],
            day_night_fit=dnf,
            fit_factor=fit_factor,
            day_threshold=day_threshold
        )

        deterministic_model.input_installations(
            decisions_so_far
        )

        deterministic_model.det_solve()

        deterministic_model.det_store_decisions(
            deterministic_output
        )

        path_result = mg.certainMerge(
            decisions_so_far,
            deterministic_output,
            no_grid_path
        )

        rolling_horizon_npvs[scenario_index] = (
            path_result['Total NPV']['Total Value']
        )

        deterministic_model.input_scenario(
            ys=0,
            scenario=scenarios[0]
        )

        deterministic_model.det_solve()

        deterministic_model.det_store_decisions(
            deterministic_output
        )

        print_excel(
            deterministic_output,
            no_grid_path,
            'D'
        )

        deterministic_npvs[scenario_index] = (
            deterministic_output['Total NPV']['Total Value']
        )

        gc.collect()

    # Summary
    output_location += r'\Summary'
    Summary = {}
    grid_index = pd.MultiIndex.from_tuples(scenarios_copy,
                                           names=['T', 'FiT', 'EDLTar'])

    # Rolling Horizon NPVs
    cols = ['Total NPV']

    rh_df = pd.DataFrame(rolling_horizon_npvs,
                         index=grid_index, columns=cols
                         )

    Summary['Rolling Horizon'] = rh_df

    # Deterministic NPVs
    det_df = pd.DataFrame(deterministic_npvs,
                          index=grid_index, columns=cols)

    Summary['Deterministic'] = det_df

    # Stochastic NPVs
    stoc_df = det_df.copy(deep=True)

    stoc_df['Total NPV'] = stochastic_npvs

    Summary['Stochastic'] = stoc_df

    valueOfCertainty_data = np.zeros(num_scenarios)
    for v in range(num_scenarios):
        valueOfCertainty_data[v] = (
            (
                rolling_horizon_npvs[v] - deterministic_npvs[v]
            )
            / abs(rolling_horizon_npvs[v])
        )

    VoC_df = pd.DataFrame(valueOfCertainty_data,
                          index=grid_index, columns=['Value'])
    Summary['Value of Certainty'] = VoC_df

    # Value of Waiting
    valueOfWaiting_data = np.zeros(num_scenarios)
    for w in range(num_scenarios):
        valueOfWaiting_data[w] = (
            (
                stochastic_npvs[w] - rolling_horizon_npvs[w]
            )
            / abs(stochastic_npvs[w])
        )

    VoW_df = pd.DataFrame(valueOfWaiting_data,
                          index=grid_index, columns=['Value'])
    Summary['Value of Waiting'] = VoW_df

    summary_path = output_location + r'\NPV_Summary.xlsx'
    writer = pd.ExcelWriter(summary_path, engine='xlsxwriter')
    for key in Summary:
        Summary[key].to_excel(writer, sheet_name=key, merge_cells=False)
    writer.close()

# %% Run


# for voll in [
#         0.00002,
#         0.00006,
#         0.00017
# ]:

folder = rf'C:\Users\User\OneDrive - American University of Beirut\Decentralized Planning Grid Uncertainty\Codes - Kareem\GitHub\Low Price'

input_data = pd.read_excel(
    r"C:\Users\User\OneDrive - American University of Beirut\Decentralized Planning Grid Uncertainty\Codes - Kareem\GitHub\Low Price\Low Price Inputs.xlsx",
    sheet_name=None
)

# input_data['General Parameters'].loc[0, 'VOLL'] = voll

output_location = os.path.join(
    folder,
    f'Low Price Results'
)

rolling_horizon(
    input_data,
    output_location,
    no_grid_scenario=True,
    dnf=False,
    data_as_dict=True
)
