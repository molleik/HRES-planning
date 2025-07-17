import os

import pandas as pd

from copy import deepcopy


def print_excel(results, file_path, model_type):
    '''
    Prints output of model to .xlsx file.

    Parameters
    ----------
    results : dict
        Dict containing model output.
    file_path : str
        Location where file is to be printed.
    model_type : str
        Type of model output is sourced from ('RH', 'D', 'S','H' etc.).

    Returns
    -------
    None.

    '''

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


def generate_scenarios(T, FiT, P):
    '''
    Generates scenario tuples from given grid policy conditions.

    Parameters
    ----------
    T : list
        Possible years where the grid will be available.
    FiT : list
        List of possible FiT values.
    P : list
        List of possible electricity prices.

    Returns
    -------
    scenarios : list
        List of possible scenarios.

    '''

    scenarios = []

    for t in T:
        for fit in FiT:
            for p in P:
                if fit < p:
                    scenarios.append((t, fit, p))

    return scenarios


def make_folders(directory, T, FiT, P):
    '''
    Creates folders to store output of models.

    Parameters
    ----------
    directory : str
        Main folder where subfolders will be created.
    T : list
        Possible years where the grid will be available.
    FiT : list
        List of possible FiT values.
    P : list
        List of possible electricity prices.

    Returns
    -------
    None.

    '''

    if not os.path.exists(directory):
        os.makedirs(directory)

    for t in T:
        for p in P:
            for fit in FiT:
                if fit < p:

                    fileFolder = os.path.join(
                        directory,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}'
                    )

                    if not os.path.exists(fileFolder):
                        os.makedirs(fileFolder)

    summary_folder = os.path.join(directory, 'Summary')
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)

    nogrid_folder = os.path.join(directory, 'T = Never')
    if not os.path.exists(nogrid_folder):
        os.makedirs(nogrid_folder)

    stoc_folder = os.path.join(directory, 'Stochastic')
    if not os.path.exists(stoc_folder):
        os.makedirs(stoc_folder)


def ss_merge(so_far, current):
    '''
    Merge outputs of 2 stochastic models with sequential, connected planning
    horizons.

    Parameters
    ----------
    so_far : dict
        Output of first stochastic model.
    current : dict
        Output of second stochastic model.

    Returns
    -------
    None.

    '''

    yc = current['ys']
    C = len(current['Dispatched Electricity'].index.levels[0])
    D = len(current['Dispatched Electricity'].index.levels[3])
    H = len(current['Dispatched Electricity'].columns)
    GT = current['Installed Capacity'].index
    ST = current['Installed Storage Power'].index
    scenarios_per_year = (
        len(so_far['Dispatched Electricity'].index.levels[0])
        - len(current['Dispatched Electricity'].index.levels[0])
    )

    # Installation
    indexed_by_g = ['Installed ', 'Added ']
    indexed_by_gy = ['Retired ', 'Remaining ']
    techs = ['Capacity', 'Storage Power', 'Storage Energy']

    gy_index = pd.MultiIndex.from_product([GT,
                                           range(yc+1)
                                           ],
                                          names=[
        'Generation Technology',
        'Year'
    ]
    )

    sy_index = pd.MultiIndex.from_product([ST,
                                           range(yc+1)
                                           ],
                                          names=[
        'Storage Technology',
        'Year'
    ]
    )

    for type_ in indexed_by_g:
        for tech in techs:
            data = type_ + tech
            so_far[data] = pd.concat(
                [
                    so_far[data],
                    current[data]
                ],
                axis=1,
                copy=False
            )

    for type_ in indexed_by_gy:
        for tech in techs:
            if tech == 'Capacity':
                level = GT
                index = gy_index
            else:
                level = ST
                index = sy_index

            parts = []
            data = type_ + tech

            for i in range(len(level)):
                parts.append(
                    so_far[data].loc[level[i]]
                )

                parts.append(
                    current[data].loc[level[i]]
                )

            so_far[data] = pd.concat(
                parts,
                axis=0,
                copy=False
            )

            so_far[data].index = index

            del parts

    # Operations
    cyd_index = pd.MultiIndex.from_product([range(C),
                                            range(yc+1),
                                            range(D)],
                                           names=['Scenario',
                                                  'Year',
                                                  'Day']
                                           )

    cgyd_index = pd.MultiIndex.from_product([range(C),
                                             GT,
                                             range(yc+1),
                                             range(D)],
                                            names=['Scenario',
                                                   'Generation Technology',
                                                   'Year',
                                                   'Day']
                                            )

    csyd_index = pd.MultiIndex.from_product([range(C),
                                             ST,
                                             range(yc+1),
                                             range(D)],
                                            names=['Scenario',
                                                   'Storage Technology',
                                                   'Year',
                                                   'Day']
                                            )

    index_map = {
        'cyd': cyd_index,
        'cgyd': cgyd_index,
        'csyd': csyd_index
    }

    h_columns = ['Hour ' + str(h) for h in range(H)]

    indexed_by = {
        'cyd': [
            'Unserved Demand',
            'Excess Electricity',
            'Purchased Electricity'
        ],
        'cgyd': [
            'Dispatched Electricity',
            'Sold (Generated)'
        ],
        'csyd': [
            'Charging',
            'Discharging',
            'State of Charge',
            'Sold (Stored)'
        ]
    }

    for index_set in indexed_by:
        outputs = indexed_by[index_set]

        for output in outputs:
            parts = []

            so_far_df = so_far[output]
            current_df = current[output]

            if (so_far_df.index.names[1] == 'Generation Technology'
                    or so_far_df.index.names[1] == 'Storage Technology'):
                level_one = so_far_df.index.levels[1]

                for c in range(C):
                    for one in level_one:
                        parts.append(so_far_df.loc[
                            (c + scenarios_per_year, one)
                        ]
                        )

                        parts.append(current_df.loc[
                            (c, one)
                        ]
                        )
            else:
                for c in range(C):
                    parts.append(so_far_df.loc[
                        (c + scenarios_per_year)
                    ]
                    )

                    parts.append(current_df.loc[
                        (c)
                    ]
                    )

            merged_df = pd.concat(parts,
                                  axis=0,
                                  copy=False
                                  )

            merged_df.index = index_map[index_set]
            merged_df.columns = h_columns

            so_far[output] = merged_df

            del parts
            del merged_df

    # Unsatisfied Reserve

    soFar_USRes_df = so_far['Unsatisfied Reserve'][(
        scenarios_per_year):].reset_index(drop=True)
    current_USRes_df = current['Unsatisfied Reserve'].reset_index(drop=True)

    merged_USRes_df = pd.concat([soFar_USRes_df, current_USRes_df],
                                axis=1,
                                copy=False
                                )

    merged_USRes_df.index = range(C)

    so_far['Unsatisfied Reserve'] = merged_USRes_df

    # NPV
    soFar_NPV_df = so_far['Total NPV']
    current_NPV_df = current['Total NPV']

    certainValues = ['Capex', 'Fixed Opex',
                     'Retirement Revenues']

    uncertainValues = ['Variable Opex', 'Fuel Opex',
                       'Emissions Tax', 'Lost Load Cost',
                       'Excess Load Cost', 'Purchased Electricity Cost',
                       'Electricity Revenues', 'Unsatisfied Reserve Cost']

    for key in certainValues:
        so_far[key] = pd.concat([so_far[key], current[key]],
                                axis=1)

    for key in uncertainValues:
        soFar_NPVC_df = so_far[key][scenarios_per_year:]
        soFar_NPVC_df.index = range(C)
        so_far[key] = pd.concat([soFar_NPVC_df, current[key]],
                                axis=1)

    for column in soFar_NPV_df:
        soFar_NPV_df[column] = soFar_NPV_df[column] + current_NPV_df[column]

    # Yearly Values
    yearlyNames = ['Unserved Demand', 'Charging',
                   'Sold (Generated)', 'Sold (Stored)', 'Dispatched Diesel',
                   'Dispatched PV', 'Dispatched Wind', 'Discharging',
                   'Excess Electricity', 'Purchased Electricity']

    yL = len(yearlyNames)

    so_far['Yearly Demand'] = pd.concat([so_far['Yearly Demand'],
                                         current['Yearly Demand']])
    yearlySoFar = so_far['Yearly Values'][scenarios_per_year*yL:]
    yearlySoFar.index = pd.MultiIndex.from_product([range(C), yearlyNames],
                                                   names=['Scenario', 'Value'])
    so_far['Yearly Values'] = pd.concat([yearlySoFar,
                                         current['Yearly Values']],
                                        axis=1)

    so_far['Scenarios'] = current['Scenarios']


def sd_merge(so_far, rem_output, directory, decoupled=False):
    '''
    Merges outputs of a stochastic model and a deterministic model.

    Parameters
    ----------
    so_far : dict
        Output of stochastic model.
    rem_output : dict
        Output of deterministic model.
    directory : str
        Folder where merged output will be saved an a .xlsx file.
    decoupled : bool, optional
        If function is being used in decoupled model. In this case,
        output file will not be printed and scenario_index will default to 0.
        The default is False.

    Returns
    -------
    final_output : dict
        Merged output of both models.

    '''

    final_output = deepcopy(so_far)

    T = rem_output['Scenario'].loc[0, 'T']
    FiT = rem_output['Scenario'].loc[0, 'FiT']
    P = rem_output['Scenario'].loc[0, 'P']

    num_scenarios = len(so_far['Scenarios'])
    detScenario = (
        T,
        FiT,
        P
    )

    Y = len(rem_output['Retired Capacity'].columns)
    D = len(rem_output['Dispatched Electricity'].index.levels[2])
    H = len(rem_output['Dispatched Electricity'].columns)
    GT = rem_output['Installed Capacity'].index
    ST = rem_output['Installed Storage Power'].index

    scenarioIndex = None
    for i in range(num_scenarios):
        if final_output['Scenarios'][i] == detScenario:
            scenarioIndex = i

    if scenarioIndex is None:
        scenarioIndex = 0

    # Installation
    indexed_by_g = ['Installed ', 'Added ']
    indexed_by_gy = ['Retired ', 'Remaining ']
    techs = ['Capacity', 'Storage Power', 'Storage Energy']

    gy_index = pd.MultiIndex.from_product([GT,
                                           range(Y)
                                           ],
                                          names=[
        'Generation Technology',
        'Year'
    ]
    )

    sy_index = pd.MultiIndex.from_product([ST,
                                           range(Y)
                                           ],
                                          names=[
        'Storage Technology',
        'Year'
    ]
    )

    for type_ in indexed_by_g:
        for tech in techs:
            data = type_ + tech
            final_output[data] = pd.concat(
                [
                    so_far[data],
                    rem_output[data]
                ],
                axis=1,
                copy=False
            )

    for type_ in indexed_by_gy:
        for tech in techs:
            if tech == 'Capacity':
                level = GT
                index = gy_index
            else:
                level = ST
                index = sy_index

            parts = []
            data = type_ + tech
            # print(data)

            for i in range(len(level)):
                parts.append(
                    so_far[data].loc[level[i]]
                )

                parts.append(
                    rem_output[data].loc[level[i]]
                )

            final_output[data] = pd.concat(
                parts,
                axis=0,
                copy=False
            )

            final_output[data].index = index

            del parts

    # Operations
    yd_index = pd.MultiIndex.from_product(
        [
            range(Y),
            range(D)
        ],
        names=[
            'Year',
            'Day'
        ]
    )

    gyd_index = pd.MultiIndex.from_product([
        GT,
        range(Y),
        range(D)
    ],
        names=[
            'Generation Technology',
            'Year',
            'Day'
    ]
    )

    syd_index = pd.MultiIndex.from_product(
        [
            ST,
            range(Y),
            range(D)
        ],
        names=[
            'Storage Technology',
            'Year',
            'Day'
        ]
    )

    index_map = {
        'yd': yd_index,
        'gyd': gyd_index,
        'syd': syd_index
    }

    h_columns = ['Hour ' + str(h) for h in range(H)]

    indexed_by = {
        'yd': [
            'Unserved Demand',
            'Excess Electricity',
            'Purchased Electricity'
        ],
        'gyd': [
            'Dispatched Electricity',
            'Sold (Generated)'
        ],
        'syd': [
            'Charging',
            'Discharging',
            'State of Charge',
            'Sold (Stored)'
        ]
    }

    for index_set in indexed_by:
        outputs = indexed_by[index_set]

        for output in outputs:
            # print(output)
            parts = []

            so_far_df = so_far[output].loc[scenarioIndex]
            remaining_df = rem_output[output]

            if (so_far_df.index.names[0] == 'Generation Technology'
                    or so_far_df.index.names[0] == 'Storage Technology'):
                level_one = so_far_df.index.levels[0]

                for one in level_one:
                    parts.append(
                        so_far_df.loc[
                            one
                        ]
                    )

                    parts.append(
                        remaining_df.loc[
                            one
                        ]
                    )
            else:
                parts.append(
                    so_far_df
                )

                parts.append(
                    remaining_df
                )

            merged_df = pd.concat(parts,
                                  axis=0,
                                  copy=False
                                  )

            merged_df.index = index_map[index_set]
            merged_df.columns = h_columns

            final_output[output] = merged_df

            del parts
            del merged_df

    # Unsatisfied Reserve
    stoch_USRes_df = final_output['Unsatisfied Reserve'].loc[[scenarioIndex]]
    stoch_USRes_df.index = ['Value']
    det_USRes_df = rem_output['Unsatisfied Reserve']

    merged_USRes_df = pd.concat([stoch_USRes_df,
                                 det_USRes_df],
                                axis=1, copy=False)
    final_output['Unsatisfied Reserve'] = merged_USRes_df

    # NPV
    final_NPV_df = final_output['Total NPV']
    det_NPV_df = rem_output['Total NPV']

    certainValues = ['Capex', 'Fixed Opex',
                     'Retirement Revenues']

    uncertainValues = ['Variable Opex', 'Fuel Opex',
                       'Emissions Tax', 'Lost Load Cost',
                       'Excess Load Cost', 'Purchased Electricity Cost',
                       'Electricity Revenues', 'Unsatisfied Reserve Cost']

    for key in certainValues:
        detThisNPV = rem_output['Yearly NPV'][[key]].transpose()
        detThisNPV.index = [0]
        final_output[key] = pd.concat([final_output[key],
                                      detThisNPV],
                                      axis=1, copy=False)

        final_output[key].columns = ['Year ' + str(y) for y in range(Y)]

    for key in uncertainValues:
        finalThisNPV = final_output[key][scenarioIndex:scenarioIndex+1]
        finalThisNPV.index = [0]
        detThisNPV = rem_output['Yearly NPV'][[key]].transpose()
        detThisNPV.index = [0]
        final_output[key] = pd.concat([finalThisNPV,
                                       detThisNPV],
                                      axis=1, copy=False)

        final_output[key].columns = ['Year ' + str(y) for y in range(Y)]

    for column in final_NPV_df:
        final_NPV_df[column] = final_NPV_df[column] + det_NPV_df[column]

    # Yearly Values
    yearlyNames = ['Unserved Demand', 'Charging',
                   'Sold (Generated)', 'Sold (Stored)', 'Dispatched Diesel',
                   'Dispatched PV', 'Dispatched Wind', 'Discharging',
                   'Excess Electricity', 'Purchased Electricity']

    finalDemand = pd.concat([final_output['Yearly Demand'],
                             rem_output['Yearly Values']['Demand']])
    finalDemand.index = range(Y)
    finalYearly = final_output['Yearly Values'].loc[scenarioIndex].transpose()
    finalYearly = pd.concat([finalYearly,
                             rem_output['Yearly Values'][yearlyNames]])
    finalYearly.index = range(Y)
    final_output['Yearly Values'] = pd.concat(
        [finalDemand, finalYearly], axis=1)
    final_output['Yearly Values'].index = range(Y)

    # Excel Output
    if not decoupled:
        if T >= Y:
            modelName = ('No Grid')
            model_type = 'RH'
        else:

            modelName = ('T = ' + str(T)
                         + ', P = ' + str(P)
                         + ', FiT = ' + str(FiT)
                         )
            model_type = 'RH'
    else:
        if T >= Y:
            modelName = ('No Grid')
            model_type = 'H'
        else:

            modelName = ('T = ' + str(T)
                         + ', P = ' + str(P)
                         + ', FiT = ' + str(FiT)
                         )
            model_type = 'H'

    # final_output['Scenario'] = rem_output['Scenario']
    # writer = pd.ExcelWriter((fileName + '.xlsx'), engine='xlsxwriter')

    # printable = ['Installed Capacity', 'Added Capacity',
    #              'Retired Capacity', 'Remaining Capacity',
    #              'Installed Storage Power', 'Added Storage Power',
    #              'Retired Storage Power', 'Remaining Storage Power',
    #              'Installed Storage Energy', 'Added Storage Energy',
    #              'Retired Storage Energy', 'Remaining Storage Energy',
    #              'Dispatched Electricity', 'Charging',
    #              'Discharging', 'State of Charge',
    #              'Unserved Demand', 'Excess Demand',
    #              'Sold (Generated)', 'Sold (Stored)', 'Purchased Electricity',
    #              'Unsatisfied Reserve', 'Unsatisfied Reserve Cost',
    #              'Total NPV', 'Capex',
    #              'Fixed Opex', 'Variable Opex',
    #              'Fuel Opex', 'Emissions Tax',
    #              'Lost Load Cost', 'Excess Load Cost',
    #              'Purchased Electricity Cost', 'Electricity Revenues',
    #              'Value of Early Retirement', 'Yearly Values'
    #              ]

    # for key in final_output:
    #     if key in printable:
    #         final_output[key].to_excel(
    #             writer, sheet_name=key, merge_cells=False)
    # writer.close()

    final_output['Model Name'] = modelName

    print_excel(final_output, directory, model_type)

    if not decoupled:
        return final_output
