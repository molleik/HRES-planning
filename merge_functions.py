import pandas as pd
from copy import deepcopy


def uncertainMerge(so_far, current):

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


def certainMerge(so_far, rem_output, directory, decoupled=False):

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
            fileName = directory + r'\RH_' + modelName
        else:

            modelName = ('T = ' + str(T)
                         + ', P = ' + str(P)
                         + ', FiT = ' + str(FiT)
                         )
            fileName = directory + r'\RH_' + modelName
    else:
        if T >= Y:
            modelName = ('No Grid')
            fileName = directory + r'\H_' + modelName
        else:

            modelName = ('T = ' + str(T)
                         + ', P = ' + str(P)
                         + ', FiT = ' + str(FiT)
                         )
            fileName = directory + r'\H_' + modelName

    final_output['Scenario'] = rem_output['Scenario']
    writer = pd.ExcelWriter((fileName + '.xlsx'), engine='xlsxwriter')

    printable = ['Installed Capacity', 'Added Capacity',
                 'Retired Capacity', 'Remaining Capacity',
                 'Installed Storage Power', 'Added Storage Power',
                 'Retired Storage Power', 'Remaining Storage Power',
                 'Installed Storage Energy', 'Added Storage Energy',
                 'Retired Storage Energy', 'Remaining Storage Energy',
                 'Dispatched Electricity', 'Charging',
                 'Discharging', 'State of Charge',
                 'Unserved Demand', 'Excess Demand',
                 'Sold (Generated)', 'Sold (Stored)', 'Purchased Electricity',
                 'Unsatisfied Reserve', 'Unsatisfied Reserve Cost',
                 'Total NPV', 'Capex',
                 'Fixed Opex', 'Variable Opex',
                 'Fuel Opex', 'Emissions Tax',
                 'Lost Load Cost', 'Excess Load Cost',
                 'Purchased Electricity Cost', 'Electricity Revenues',
                 'Value of Early Retirement', 'Yearly Values'
                 ]

    for key in final_output:
        if key in printable:
            final_output[key].to_excel(
                writer, sheet_name=key, merge_cells=False)
    writer.close()

    final_output['Model Name'] = modelName

    return final_output
