import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd

import os

'''
Functions in this module are used to generate the figures presented in the
paper linked in the GitHub repository.
'''

red = '#D34C4C'
yellow = '#FFE493'
light_blue = '#B0C3E6'
light_green = '#C2DEAF'
light_gray = '#C7C4C4'
light_purple = '#E3CBD7'
orange = '#F8C894'
dark_gray = '#757373'


def plot_NPV(output, out_dir=None, alpha=1, model_type='',
             save=True, show=False):
    '''
    Generates stacked bar plot of NPV based on results of a model.

    Parameters
    ----------
    output : dict
        Output dict containing model results.
    out_dir : str
        Folder where the image will be saved. The default is None.
    alpha : float, optional
        Transparency for plot colors. The default is 1.
    model_type : str, optional
        Type of model results are taken from. Only for file name distinction.
        The default is ''.
    save : bool, optional
        To save or not to save the resulting plot as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plot. The default is False.

    Returns
    -------
    None.

    '''

    # NPV
    name = (
        # out_dir
        model_type
        + '_NPV'
        # + output['Model Name']
        + '.png'
    )

    NPV_df = output['Total NPV']
    fig, ax = plt.subplots(figsize=(2, 8))

    costs = [
        'Capex', 'Fixed Opex', 'Variable Opex', 'Fuel Opex',
        'Emissions Tax', 'Lost Load Cost', 'Excess Load Cost',
        'Purchased Electricity Cost']

    revenues = ['Salvage Value', 'Value of Early Retirement', 'Revenues']
    bar_colors = ['skyblue', '#C2DEAF',
                  '#C7C4C4', '#D34C4C',
                  '#FFE493', '#B0C3E6',
                  '#E3CBD7', '#F8C894',
                  '#757373']
    cols = revenues + costs

    bar = 0
    alpha = 1
    for key in revenues:
        value = NPV_df[key].values
        value = -value
        ax.bar(x=[0], height=value, bottom=bar,
               color='green', alpha=alpha)
        bar += float(value)
        alpha *= 0.8

    bar = 0
    alpha = 1
    color_index = 0
    for key in costs:
        value = NPV_df[key].values
        ax.bar(x=[0], height=value,  bottom=bar,
               color=bar_colors[color_index])
        color_index += 1
        bar += float(value)
        alpha *= 0.8

    ax.set_ylabel('m$')
    ax.set_xticks([y for y in range(-1, 2)])
    ax.set_xticks([])
    ax.set_yticks([x for x in range(-30, 240, 25)])
    plt.legend(title='NPV Breakdown', labels=cols, loc='center',
               bbox_to_anchor=(1.8, 0.5), ncol=1)

    if save:
        file_name = os.path.join(
            out_dir,
            name
        )
        plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_added_retired(output, out_dir=None, alpha=1,
                       model_type='', fs=30,
                       save=True, show=False):
    '''
    Generates stacked bar plot of yearly added and retired capacities.

    Parameters
    ----------
    output : dict
        Output dict containing model results.
    out_dir : str
        Folder where the image will be saved. The default is None.
    alpha : float, optional
        Transparency for plot colors. The default is 1.
    model_type : str, optional
        Type of model results are taken from. Only for file name distinction.
        The default is ''.
    fs : int, optional
        Font size for label and tick text. The default is 30.
    save : bool, optional
        To save or not to save the resulting plot as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plot. The default is False.

    Returns
    -------
    None.

    '''

    name = (
        model_type
        + '_Added Retired'
        + '.png'
    )

    Y = output['Remaining Capacity'].shape[1]
    ys = Y - output['Installed Capacity'].shape[1]
    years = list(range(ys, Y))

    added_cap_df = output['Added Capacity']
    added_diesel = added_cap_df.loc['Diesel'].values
    added_pv = added_cap_df.loc['PV'].values
    added_wind = added_cap_df.loc['Wind'].values

    added_stp_df = output['Added Storage Power']
    added_battery = added_stp_df.loc['Battery'].values

    retired_cap_df = output['Retired Capacity']
    retired_diesel = retired_cap_df.loc['Diesel'].values
    yearly_retired_diesel = [-sum(retired_diesel[i])
                             for i in range(len(retired_diesel))]

    retired_pv = retired_cap_df.loc['PV'].values
    yearly_retired_pv = [-sum(retired_pv[i]) for i in range(len(retired_pv))]

    retired_wind = retired_cap_df.loc['Wind'].values
    yearly_retired_wind = [-sum(retired_wind[i])
                           for i in range(len(retired_wind))]

    retired_stp_df = output['Retired Storage Power']
    retired_battery = retired_stp_df.loc['Battery'].values
    yearly_retired_battery = [-sum(retired_battery[i])
                              for i in range(len(retired_battery))]

    plt.figure(figsize=(30, 12))

    plt.bar(years, added_diesel, color=red,
            label='Added Diesel', alpha=alpha)
    plt.bar(years, added_pv,
            color=yellow, label='Added PV',
            bottom=added_diesel, alpha=alpha)
    plt.bar(years, added_wind, color=light_blue, label='Added Wind',
            bottom=np.array(added_diesel) + np.array(added_pv),
            alpha=alpha)
    plt.bar(years, added_battery, color=light_green, label='Added Battery',
            bottom=np.array(added_diesel) + np.array(added_pv) +
            np.array(added_wind), alpha=alpha)

    plt.bar(years, yearly_retired_diesel,
            color=red, label='Retired Diesel',
            hatch='/', alpha=alpha)

    plt.bar(years, yearly_retired_pv,
            color=yellow, label='Retired PV',
            bottom=yearly_retired_diesel,
            hatch='/',
            alpha=alpha)

    plt.bar(years, yearly_retired_wind,
            color=light_blue, label='Retired Wind',
            bottom=np.array(yearly_retired_diesel) + np.array(
                yearly_retired_pv),
            hatch='/',
            alpha=alpha)

    plt.bar(years, yearly_retired_battery,
            color=light_green, label='Retired Battery',
            bottom=np.array(yearly_retired_diesel) + np.array(
                yearly_retired_pv) + np.array(yearly_retired_wind),
            hatch='/',
            alpha=alpha)

    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Years', fontsize=fs)
    plt.ylabel('Capacity (MW)', fontsize=fs)
    plt.xticks(years, [f'{y}' for y in years], fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.ylim(-30, 70)

    plt.legend(fontsize=fs, loc='center',
               bbox_to_anchor=(0.5, 1.15), ncol=4)

    if save:
        file_name = os.path.join(
            out_dir,
            name
        )
        plt.savefig(file_name, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_installed(output, out_dir=None, alpha=1,
                   model_type='', fs=30,
                   save=True, show=False):
    '''
    Generates stacked bar plot of yearly installed capacities.

    Parameters
    ----------
    output : dict
        Output dict containing model results.
    out_dir : str
        Folder where the image will be saved. The default is None.
    alpha : float, optional
        Transparency for plot colors. The default is 1.
    model_type : str, optional
        Type of model results are taken from. Only for file name distinction.
        The default is ''.
    fs : int, optional
        Font size for label and tick text. The default is 30.
    save : bool, optional
        To save or not to save the resulting plot as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plot. The default is False.

    Returns
    -------
    None.

    '''

    name = (
        model_type
        + '_Installed'
        + '.png'
    )

    Y = output['Remaining Capacity'].shape[1]
    ys = Y - output['Installed Capacity'].shape[1]
    years = list(range(ys, Y))

    installed_capacity_df = output['Installed Capacity']
    installed_diesel = installed_capacity_df.loc['Diesel'].values
    installed_pv = installed_capacity_df.loc['PV'].values
    installed_wind = installed_capacity_df.loc['Wind'].values

    installed_stp_df = output['Installed Storage Power']
    installed_battery = installed_stp_df.loc['Battery'].values
    plt.figure(figsize=(30, 12))
    plt.bar(years, installed_diesel, color=red,
            label='Diesel', alpha=alpha)

    plt.bar(years, installed_pv, color=yellow, label='PV',
            bottom=installed_diesel, alpha=alpha)

    plt.bar(years, installed_wind, color=light_blue, label='Wind',
            bottom=np.array(installed_diesel) + np.array(installed_pv),
            alpha=0.65)

    plt.bar(years, installed_battery, color=light_green, label='Battery',
            bottom=np.array(installed_diesel) + np.array(installed_pv) +
            np.array(installed_wind), alpha=alpha)

    plt.xlabel('Years', fontsize=fs)
    plt.ylabel('Installed Capacity (MW)', fontsize=fs)
    plt.xticks(years, [f'{y}' for y in years], fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.ylim(0, 100)

    plt.legend(fontsize=fs,
               bbox_to_anchor=(0.9, 1.2),
               ncol=4
               )

    if save:
        file_name = os.path.join(
            out_dir,
            name
        )
        plt.savefig(file_name, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_operations(output, out_dir=None, alpha=1,
                    model_type='', save=True, show=False):
    '''
    Generates stacked bar plot of yearly operational decisions.

    Parameters
    ----------
    output : dict
        Output dict containing model results.
    out_dir : str
        Folder where the image will be saved. The default is None.
    alpha : float, optional
        Transparency for plot colors. The default is 1.
    model_type : str, optional
        Type of model results are taken from. Only for file name distinction.
        The default is ''.
    save : bool, optional
        To save or not to save the resulting plot as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plot. The default is False.

    Returns
    -------
    None.

    '''

    name = (
        model_type
        + '_Generation'
        + '.png'
    )

    generation_df = output['Yearly Values']

    bar_colors = [
        '#757373']
    column_colors = {
        'Unserved Demand': 'skyblue',
        'Excess Electricity': orange,
        'Charging': light_green,
        'Discharging': light_purple,
        'Sold (Generated)': light_gray,
        'Sold (Stored)': 'gray',
        'Dispatched Diesel': red,
        'Dispatched PV': yellow,
        'Dispatched Wind': light_blue,
        'Purchased Electricity': dark_gray
    }

    bar_colors = [
        column_colors.get(col, 'purple')
        for col in generation_df.columns
        if col != 'Demand'
    ]

    line_color = 'black'

    columns_to_negate = ['Charging',
                         'Sold (Generated)',
                         'Sold (Stored)', 'Excess Electricity']
    generation_df[columns_to_negate] = generation_df[columns_to_negate] * -1
    upper_bound = generation_df.values.max()*1.5
    upper_bound = 9500
    lower_bound = generation_df[columns_to_negate].values.min()*1.2
    lower_bound = -1000

    fig, ax1 = plt.subplots(figsize=(35, 20))
    ax1 = (
        generation_df.drop(columns=['Demand'])
        # generation_df['Sold (Generated)']
    ).plot(kind='bar',
           stacked=True,
           ax=ax1,
           color=bar_colors,
           alpha=alpha)
    ax1.set_ylabel("Energy (MWh)", fontsize=50)
    ax1.set_xlabel("Years", fontsize=50)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='right')

    plt.legend(loc='upper left', fontsize=40, ncol=2)

    ax2 = ax1.twinx()
    generation_df['Demand'].plot(ax=ax2, color=line_color, marker='o',
                                 linestyle='-', linewidth=2,
                                 markersize=20, alpha=alpha)
    ax2.invert_yaxis()
    ax2.set_yticklabels([])

    ax1.set_ylim(lower_bound, upper_bound)
    ax2.set_ylim(lower_bound, upper_bound)

    plt.subplots_adjust(top=0.9, bottom=0.1)

    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)

    plt.axhline(0, color='black', linestyle='--')

    ax1.tick_params(axis='both', which='major', labelsize=40)
    plt.yticks(fontsize=40)

    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    if save:
        file_name = os.path.join(
            out_dir,
            name
        )
        plt.savefig(file_name, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_energy_mix(output, out_dir=None, alpha=1,
                    model_type=None, save=True, show=False):
    '''
    Generates stacked bar plot of energy mix i.e. percentage of each technology
    in the total installations.

    Parameters
    ----------
    output : dict
        Output dict containing model results.
    out_dir : str
        Folder where the image will be saved. The default is None.
    alpha : float, optional
        Transparency for plot colors. The default is 1.
    model_type : str, optional
        Type of model results are taken from. Only for file name distinction.
        The default is None.
    save : bool, optional
        To save or not to save the resulting plot as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plot. The default is False.

    Returns
    -------
    None.

    '''

    name = (
        model_type
        + '_Energy Mix'
        + '.png'
    )

    Y = output['Remaining Capacity'].shape[1]
    years = list(range(output['ys'], Y))

    diesel = output['Installed Capacity'].loc[['Diesel']].values.flatten()
    pv = output['Installed Capacity'].loc[['PV']].values.flatten()
    wind = output['Installed Capacity'].loc[['Wind']].values.flatten()
    power = output['Installed Storage Power'].values.flatten()

    sum_ = diesel + pv + wind + power

    diesel = diesel/sum_ * 100
    pv = pv/sum_ * 100
    wind = wind/sum_ * 100
    power = power/sum_ * 100

    capacities = [diesel, pv, wind, power]
    fig, ax = plt.subplots(
        figsize=(30, 12)
    )

    color_map = {
        0: '#D34C4C',
        1: '#FFE493',
        2: '#B0C3E6',
        3: '#C2DEAF'
    }

    label_map = {
        0: 'Diesel',
        1: 'PV',
        2: 'Wind',
        3: 'Power'
    }

    bar = np.zeros((len(years)))

    i = 0

    for cap in capacities:
        ax.bar(x=years, height=cap, bottom=bar,
               color=color_map[i], alpha=alpha, label=label_map[i])
        bar += cap
        i += 1

    plt.xlabel('Years', fontsize=40)
    plt.ylabel('%', fontsize=40)
    plt.xticks(years, [f'{y}' for y in years], fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylim(0, 100)

    ax.legend(fontsize=30, loc='center',
              bbox_to_anchor=(0.5, 1.1), ncol=2)

    if save:
        file_name = os.path.join(
            out_dir,
            name
        )
        plt.savefig(file_name, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_all_decisions(output, out_dir=None, alpha=1, model_type='', fs=30,
                       generation=True, save=True, show=False):
    '''
    Generates NPV, installed, added/retired, and operations plots.

    Parameters
    ----------
    output : dict
        Output dict containing model results.
    out_dir : str
        Folder where the image will be saved. The default is None.
    alpha : float, optional
        Transparency for plot colors. The default is 1.
    model_type : str, optional
        Type of model results are taken from. Only for file name distinction.
        The default is ''.
    fs : int, optional
        Font size for label and tick text. The default is 30.
    save : bool, optional
        To save or not to save the resulting plots as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plots. The default is False.

    Returns
    -------
    None.

    '''

    plot_NPV(output, out_dir, alpha, model_type=model_type,
             save=save, show=show)

    plot_installed(output, out_dir, alpha, model_type=model_type,
                   fs=fs, save=save, show=show)

    plot_added_retired(output, out_dir, alpha, model_type=model_type,
                       fs=fs, save=save,
                       show=show)

    if generation:
        plot_operations(output, out_dir, alpha, model_type=model_type,
                        save=save, show=show)


def boxplots_by_parameter(summary, result_name, unit, out_dir=None,
                          save=True, show=False):
    '''
    Generates three boxplots of value indicator grouped by T, P and FiT.

    Parameters
    ----------
    summary : DataFrame
        DF containing values for the indicator.
    result_name : str
        Name of value indicator, used in file name.
    unit : str
        Unit of y axis.
    out_dir : str
        Output folder where the plots are to be saved.
    save : bool, optional
        To save or not to save the resulting plots as a png.
        The default is True.
    show : bool, optional
        To show or not to show the resulting plots. The default is False.

    Returns
    -------
    None.

    '''

    T = summary.index.levels[0]
    FiT = summary.index.levels[1]
    P = summary.index.levels[2]

    # By T
    values_by_t = {}
    for t in T:
        values_by_t[t] = summary.loc[t]

    plt.boxplot([values_by_t[t].values.flatten() for t in T],
                patch_artist=True)
    plt.title(result_name)
    plt.xlabel('T')
    plt.ylabel(unit)
    plt.xticks(ticks=[x+1 for x in range(len(T))],
               labels=T)

    if save:
        fig_name = result_name + ' by T'
        file_name = os.path.join(
            out_dir,
            fig_name
        )
        plt.savefig(file_name)
    if show:
        plt.show()

    # By Tariff
    values_by_p = {}
    resultsByP = summary.swaplevel(i=0, j=2)
    for p in P:
        values_by_p[p] = resultsByP.loc[p]

    plt.boxplot([values_by_p[p].values.flatten()
                 for p in P
                 ],
                patch_artist=True)
    plt.title(result_name)
    plt.xlabel('P')
    plt.xticks(ticks=[x+1 for x in range(len(P))], labels=P)
    plt.ylabel(unit)

    if save:
        fig_name = result_name + ' by P'
        file_name = os.path.join(
            out_dir,
            fig_name
        )
        plt.savefig(file_name)
    if show:
        plt.show()

    # By FiT
    values_by_fit = {}
    resultsByFiT = summary.swaplevel(i=0, j=1)
    for fit in FiT:
        values_by_fit[fit] = resultsByFiT.loc[fit]

    plt.boxplot([values_by_fit[fit].values.flatten() for fit in FiT],
                patch_artist=True)
    plt.title(result_name)
    plt.xlabel('FiT')
    plt.xticks(ticks=[x+1 for x in range(len(FiT))], labels=FiT)
    plt.ylabel(unit)

    if save:
        fig_name = result_name + ' by FiT'
        file_name = os.path.join(
            out_dir,
            fig_name
        )
        plt.savefig(file_name)
    if show:
        plt.show()
    if show:
        plt.show()


def voml_by_P(main_dir, out_dir):
    '''
    Generates boxplot for value of market liquidity for each case of
    discounting considered (30% and 100%).

    Parameters
    ----------
    main_dir : str
        Folder where base case and discount cases are stored.
    out_dir : str
        Folder where the plot is to be saved.

    Returns
    -------
    None.

    '''

    discounts = [
        '30',
        '100'
    ]

    T = [x for x in range(1, 11)]

    FiT = [
        0.03,
        0.08,
        0.15
    ]

    P = [
        0.1,
        0.185,
        0.27
    ]

    tuples = []

    for disc in discounts:
        for t in T:
            for fit in FiT:
                for p in P:
                    if fit < p:
                        tuples.append((disc, t, fit, p))

        tuples.append((disc, 20, 0, 0))

    base_results = pd.read_excel(
        os.path.join(
            main_dir,
            'Base Case',
            'Base Case Results',
            'Summary',
            'NPV_SUmmary.xlsx'
        ),
        sheet_name='Rolling Horizon',
        index_col=[0, 1, 2]
    )

    tdisc_results = pd.read_excel(
        os.path.join(
            main_dir,
            '30% Discount',
            '30% Discount Results',
            'Summary',
            'NPV_SUmmary.xlsx'
        ),
        sheet_name='Rolling Horizon',
        index_col=[0, 1, 2]
    )

    hdisc_results = pd.read_excel(
        os.path.join(
            main_dir,
            '100% Discount',
            '100% Discount Results',
            'Summary',
            'NPV_SUmmary.xlsx'
        ),
        sheet_name='Rolling Horizon',
        index_col=[0, 1, 2]
    )

    t_voc = 100*(tdisc_results - base_results)/abs(tdisc_results)
    h_voc = 100*(hdisc_results - base_results)/abs(hdisc_results)

    allvalues = pd.concat(
        [
            t_voc,
            h_voc
        ],
    )

    allvalues.index = pd.MultiIndex.from_tuples(tuples)
    allvalues.index.names = ['Discount', 'T', 'FiT', 'P']

    colors = ['red', 'firebrick']
    for i in range(3):
        colors += colors
    to_plot = []

    for p in P:
        # for fit in FiT:
        for disc in discounts:
            values = allvalues.loc[
                disc,
                :,
                :,
                # fit,
                # :,
                p
            ].values.flatten()*100
            to_plot.append(values)

    fig, ax = plt.subplots(1, 1)
    bplot = ax.boxplot(to_plot, patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.legend(['30% Discount', '100% Discount'])

    ax.set_xticks(
        ticks=[2*x+1.5 for x in range(3)],
        labels=P,
        # labels=FiT,
        fontsize=10
    )

    ax.set_yticks(
        ticks=[2.5*x for x in range(9)],
        labels=[str(2.5*x) for x in range(9)],
        fontsize=10
    )

    ax.set_xlabel(
        # 'FiT',
        'P',
        fontsize=10
    )

    ax.set_ylabel(
        '%',
        fontsize=10
    )

    plt.savefig(
        os.path.join(
            out_dir,
            'Value of Market Liquidity by P.png'
        )
    )

    plt.show()


def model_comparison(main_dir, out_dir):
    '''
    Generates boxplot for value of certainty for naive deterministic,
    stochastic and rolling horizon models.

    Parameters
    ----------
    main_dir : str
        Folder where results are stored.
    out_dir : str
        Folder where the plot is to be saved.

    Returns
    -------
    None.

    '''

    dvd_data = pd.read_excel(
        os.path.join(
            main_dir,
            'Naive Deterministic Results.xlsx'
        ),
        index_col=[0]
    )

    other_data = pd.read_excel(
        os.path.join(
            main_dir,
            'Base Case Results',
            'Summary',
            'NPV_Summary.xlsx'
        ),
        sheet_name=None,
        index_col=[0, 1, 2]
    )

    svd_data = (
        (other_data['Stochastic'] - other_data['Deterministic'])
        /
        abs(other_data['Stochastic'])
    )

    rhvd_data = other_data['Value of Certainty']

    values = dvd_data.values.flatten()

    fig, ax = plt.subplots()

    ax.boxplot(
        [
            values*100,
            svd_data.values.flatten()*100,
            rhvd_data.values.flatten()*100,
        ],
        patch_artist=True
    )

    plt.xticks(
        ticks=[1, 2, 3],
        labels=[
            'Deterministic',
            'Stochastic',
            'Rolling Horizon',
        ],
        fontsize=12
    )

    plt.ylabel(
        '%'
    )
    plt.yticks(
        fontsize=12)

    ins = ax.inset_axes([0.7, 0.45, 0.25, 0.4])

    ins.boxplot(
        [
            rhvd_data.values.flatten()*100,
        ],
        patch_artist=True
    )

    ins.set_yticks(
        ticks=[x*10e-2 for x in range(4)],
        labels=[str(x*10) + 'e-02' for x in range(4)],
        fontsize=10
    )

    ins.set_xticks(
        ticks=[]
    )

    ax.annotate("", xytext=(3, 3), xy=(3, 29.5),
                arrowprops=dict(arrowstyle="->"),
                fontsize=20)

    plt.savefig(
        os.path.join(
            out_dir,
            'Model Comparison.png'
        )
    )

    plt.show()


def deviation_from_no_grid(main_dir, out_dir):
    '''
    Generates stacked barplot similar to plot_installed but specifically for
    the no grid case, along with lineplot of deviation from the deterministic
    model for the first 10 years.

    Parameters
    ----------
    main_dir : str
        Folder where base case and discount cases are stored.
    out_dir : str
        Folder where the plot is to be saved.

    Returns
    -------
    None.

    '''

    # Baseline (no grid) deterministic result
    d_no_grid_path = os.path.join(
        main_dir,
        'T = Never',
        'D_NoGrid.xlsx'
    )

    # Merge capacity and storage dataframes
    d_no_grid_cap = pd.read_excel(
        d_no_grid_path,
        sheet_name='Installed Capacity',
        index_col=[0]
    )

    d_no_grid_stp = pd.read_excel(
        d_no_grid_path,
        sheet_name='Installed Storage Power',
        index_col=[0]
    )

    d_no_grid_inst = pd.concat(
        [
            d_no_grid_cap,
            d_no_grid_stp
        ]
    )

    # Rolling Horizon result
    rh_no_grid_path = os.path.join(
        main_dir,
        'T = Never',
        'RH_NoGrid.xlsx'
    )

    # Merge capacity and storage dataframes
    rh_no_grid_cap = pd.read_excel(
        rh_no_grid_path,
        sheet_name='Installed Capacity',
        index_col=[0]
    )

    rh_no_grid_stp = pd.read_excel(
        rh_no_grid_path,
        sheet_name='Installed Storage Power',
        index_col=[0]
    )

    rh_no_grid_inst = pd.concat(
        [
            rh_no_grid_cap,
            rh_no_grid_stp
        ]
    )

    alphas = (
        ((d_no_grid_inst - rh_no_grid_inst)**2).sum()**0.5
        / d_no_grid_inst.sum()
    )[[f'Year {y}' for y in range(10)]]

    deviation = np.copy(alphas)

    for i in range(len(deviation)):
        deviation[i] = sum(alphas[:i+1])/(i+1)

    # Plot installations and deviation
    alpha = 1
    fs = 20
    red = '#D34C4C'
    yellow = '#FFE493'
    light_blue = '#B0C3E6'
    light_green = '#C2DEAF'
    dark_gray = '#757373'

    Y = 20
    ys = 0
    years = list(range(ys, Y))

    # Installations
    installed_diesel = rh_no_grid_cap.loc['Diesel'].values
    installed_pv = rh_no_grid_cap.loc['PV'].values
    installed_wind = rh_no_grid_cap.loc['Wind'].values
    installed_battery = rh_no_grid_inst.loc['Battery'].values

    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.bar(years, installed_diesel, color=red,
            label='Diesel', alpha=alpha)
    ax1.bar(years, installed_pv, color=yellow, label='PV',
            bottom=installed_diesel, alpha=alpha)
    ax1.bar(years, installed_wind, color=light_blue, label='Wind',
            bottom=np.array(installed_diesel) + np.array(installed_pv),
            alpha=0.65)
    ax1.bar(years, installed_battery, color=light_green, label='Battery',
            bottom=np.array(installed_diesel) + np.array(installed_pv) +
            np.array(installed_wind), alpha=alpha)

    ax1.set_xlabel('Years', fontsize=fs)
    ax1.set_ylabel('Installed Capacity (MW)', fontsize=fs)
    ax1.set_xticks(years)
    ax1.set_xticklabels([f'{y}' for y in years], fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    ax1.set_ylim(0, 150)

    # % Deviation
    ax2 = ax1.twinx()
    ax2.plot(years[:len(deviation)], deviation*100,
             color=dark_gray, marker='o', linestyle='--', linewidth=2,
             markersize=8, label='Deviation')
    ax2.set_ylabel('Deviation (%)', fontsize=fs)
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.set_ylim(0, 15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=fs,
               bbox_to_anchor=(0.9, 1.2),
               ncol=5
               )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            'Deviation from No Grid.png'
        )
    )

    plt.show()


def grid_yearly_installed(folder, y, xlim):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing stacked barplots
    of installed capacities at a certain year for all scenarios except the no
    grid case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    y : int
        Year of which the installations will be displayed.
    xlim : float
        Limit of x-axis.

    Returns
    -------
    None.

    '''

    all_retired = []

    for p in [
            0.27,
            0.185,
            0.1
    ]:
        for fit in [
                0.03,
                0.08,
                0.15
        ]:
            grids = []
            if fit < p:

                for t in range(1, 11):
                    rh_path = os.path.join(
                        folder,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}',
                        f'RH_Tgrid = {t}_FiT = {fit}_EDLTar = {p}.xlsx'
                    )

                    cap = pd.read_excel(
                        rh_path,
                        sheet_name='Installed Capacity',
                        index_col=[0]
                    )[[f'Year {y}']]

                    strg = pd.read_excel(
                        rh_path,
                        sheet_name='Installed Storage Power',
                        index_col=[0]
                    )[[f'Year {y}']]

                    cap_sum = pd.DataFrame(
                        [cap.loc[['Diesel']].sum().sum(),
                         cap.loc[['PV']].sum().sum(),
                         cap.loc[['Wind']].sum().sum()],
                        index=['Diesel', 'PV', 'Wind']
                    )

                    strg_sum = pd.DataFrame(
                        [strg.loc[['Battery']].sum().sum()],
                        index=['Battery']
                    )

                    this_t = pd.concat(
                        [
                            cap_sum,
                            strg_sum
                        ]
                    )

                    grids.append(this_t)

            all_retired.append(grids)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n in range(len(all_retired)):

        if n != 8:
            retireds = all_retired[n]

            for i in range(len(retireds)):
                thing = retireds[i]

                h = (i+1)/4
                # h = 5

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Diesel'],
                    color=red,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['PV'],
                    left=thing.loc['Diesel'],
                    color=yellow,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Wind'],
                    left=thing.loc['Diesel'] + thing.loc['PV'],
                    color=light_blue,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Battery'],
                    left=thing.loc['Diesel'] +
                    thing.loc['PV'] + thing.loc['Wind'],
                    color=light_green,
                    # hatch= '/'
                )

                if n in [2, 5, 7]:
                    axes.flat[n].tick_params(
                        which='both',
                        left=False,
                        labelleft=False,
                        right=True,
                        labelright=True,
                    )
                else:
                    axes.flat[n].tick_params(
                        which='both',
                        left=False,
                        labelleft=False,
                        right=False,
                        labelright=False
                    )
    for row in range(3):
        col = 2 if row < 2 else 1
        axes[row][col].set_yticks(
            ticks=[(i+1)/4 for i in range(10)],
            labels=range(1, 11),
            fontsize=8
        )
        axes[row][col].set_ylabel('T',
                                  rotation=0,
                                  labelpad=15,
                                  fontsize=10)
        axes[row][col].yaxis.set_label_position("right")
        axes[row][col].yaxis.set_label_coords(x=1.2, y=0.5)

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.185')
    axes[2][0].set_ylabel('0.1')

    axes[2][0].set_xlabel('0.03', fontsize=10, labelpad=10)
    axes[2][0].xaxis.set_label_coords(x=0.5, y=-0.25)

    axes[2][1].set_xlabel('0.08', fontsize=10, labelpad=10)
    axes[2][1].xaxis.set_label_coords(x=0.5, y=-0.25)

    axes[1][2].set_xlabel('0.15', fontsize=10, labelpad=10)
    axes[1][2].xaxis.set_label_coords(x=0.5, y=-0.15)

    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])

    plt.xlim(0, xlim)

    custom_lines = [
        Line2D([0], [0], color=red, lw=4),
        Line2D([0], [0], color=yellow, lw=4),
        Line2D([0], [0], color=light_blue, lw=4),
        Line2D([0], [0], color=light_green, lw=4)
    ]

    plt.legend(custom_lines, [
        'Diesel',
        'PV',
        'Wind',
        'Battery'
    ],
        loc='lower right',
        bbox_to_anchor=(1.1, 0, 1, 1)
    )

    plt.show()


def grid_cumul_installed(folder, xlim):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing stacked barplots
    of cumulative installed capacities for all scenarios except the no grid
    case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    xlim : float
        Limit of x-axis.

    Returns
    -------
    None.

    '''

    all_retired = []

    for p in [0.27, 0.185, 0.1]:
        for fit in [0.03, 0.08, 0.15]:
            grids = []
            if fit < p:

                for t in range(1, 11):
                    rh_path = os.path.join(
                        folder,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}',
                        f'RH_Tgrid = {t}_FiT = {fit}_EDLTar = {p}.xlsx'
                    )

                    cap = pd.read_excel(
                        rh_path,
                        sheet_name='Added Capacity',
                        index_col=[0]
                        # )[[f'Year {y}' for y in range(t, 20)]].mean(axis= 1)
                    )

                    strg = pd.read_excel(
                        rh_path,
                        sheet_name='Added Storage Power',
                        index_col=[0]
                        # )[[f'Year {y}' for y in range(t, 20)]].mean(axis= 1)
                    )

                    cap_sum = pd.DataFrame(
                        [cap.loc[['Diesel']].sum().sum(),
                         cap.loc[['PV']].sum().sum(),
                         cap.loc[['Wind']].sum().sum()],
                        index=['Diesel', 'PV', 'Wind']
                    )

                    strg_sum = pd.DataFrame(
                        [strg.loc[['Battery']].sum().sum()],
                        index=['Battery']
                    )

                    this_t = pd.concat(
                        [
                            cap_sum,
                            strg_sum
                        ]
                    )

                    grids.append(this_t)

            all_retired.append(grids)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n in range(len(all_retired)):

        if n != 8:
            retireds = all_retired[n]

            for i in range(len(retireds)):
                thing = retireds[i]

                h = (i+1)/4
                # h = 5

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Diesel'],
                    color=red,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['PV'],
                    left=thing.loc['Diesel'],
                    color=yellow,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Wind'],
                    left=thing.loc['Diesel'] + thing.loc['PV'],
                    color=light_blue,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Battery'],
                    left=thing.loc['Diesel'] +
                    thing.loc['PV'] + thing.loc['Wind'],
                    color=light_green,
                    # hatch= '/'
                )

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.185')
    axes[2][0].set_ylabel('0.1')

    axes[2][0].set_xlabel('0.03')
    axes[2][1].set_xlabel('0.08')
    axes[1][2].set_xlabel('0.15',
                          # labelpad= 205
                          )

    # fig.suptitle('FiT')
    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])
    plt.yticks(
        ticks=[]
    )

    # plt.xticks(
    #     ticks= []
    #     )

    # plt.xlabel('Capacity (MW)')
    # plt.ylim(4.5, 5.5)
    plt.xlim(0, xlim)
    plt.show()


def grid_cumul_retired(folder, xlim):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing stacked barplots
    of cumulative retired capacities for all scenarios except the no grid
    case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    xlim : float
        Limit of x-axis.

    Returns
    -------
    None.

    '''

    all_retired = []

    for p in [0.27, 0.185, 0.1]:
        for fit in [0.03, 0.08, 0.15]:
            grids = []
            if fit < p:

                for t in range(1, 11):
                    rh_path = os.path.join(
                        folder,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}',
                        f'RH_Tgrid = {t}_FiT = {fit}_EDLTar = {p}.xlsx'
                    )

                    cap = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Capacity',
                        index_col=[0, 1]
                        # )[[f'Year {y}' for y in range(t, 20)]].mean(axis= 1)
                    )

                    strg = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Storage Power',
                        index_col=[0, 1]
                        # )[[f'Year {y}' for y in range(t, 20)]].mean(axis= 1)
                    )

                    cap_sum = pd.DataFrame(
                        [cap.loc[['Diesel']].sum().sum(),
                         cap.loc[['PV']].sum().sum(),
                         cap.loc[['Wind']].sum().sum()],
                        index=['Diesel', 'PV', 'Wind']
                    )

                    strg_sum = pd.DataFrame(
                        [strg.loc[['Battery']].sum().sum()],
                        index=['Battery']
                    )

                    this_t = pd.concat(
                        [
                            cap_sum,
                            strg_sum
                        ]
                    )

                    grids.append(this_t)

            all_retired.append(grids)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n in range(len(all_retired)):

        if n != 8:
            retireds = all_retired[n]

            for i in range(len(retireds)):
                thing = retireds[i]

                h = (i+1)/4
                # h = 5

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Diesel'],
                    color=red,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['PV'],
                    left=thing.loc['Diesel'],
                    color=yellow,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Wind'],
                    left=thing.loc['Diesel'] + thing.loc['PV'],
                    color=light_blue,
                    # hatch= '/'
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Battery'],
                    left=thing.loc['Diesel'] +
                    thing.loc['PV'] + thing.loc['Wind'],
                    color=light_green,
                    # hatch= '/'
                )

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.185')
    axes[2][0].set_ylabel('0.1')

    axes[2][0].set_xlabel('0.03')
    axes[2][1].set_xlabel('0.08')
    axes[1][2].set_xlabel('0.15',
                          # labelpad= 205
                          )

    # fig.suptitle('FiT')
    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])
    plt.yticks(
        ticks=[]
    )

    # plt.xticks(
    #     ticks= []
    #     )

    # plt.xlabel('Capacity (MW)')
    # plt.ylim(4.5, 5.5)
    plt.xlim(0, xlim)
    plt.show()


def percent_retired(folder):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing stacked barplots
    showing how much of each technology's total installed capacity was retired
    in % for all scenarios except the no grid case.

    Parameters
    ----------
    folder : str
        Location of result folder.

    Returns
    -------
    None.

    '''

    all_things = []

    for p in [
            0.27,
            0.1525,
            0.035
    ]:
        for fit in [
                0.03,
                0.08,
                0.15
        ]:
            grids = []
            if fit < p:

                for t in range(1, 11):
                    rh_path = os.path.join(
                        folder,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}',
                        f'RH_Tgrid = {t}_FiT = {fit}_EDLTar = {p}.xlsx'
                    )

                    retcap = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Capacity',
                        index_col=[0, 1]
                    )

                    instcap = pd.read_excel(
                        rh_path,
                        sheet_name='Added Capacity',
                        index_col=[0]
                    )

                    retstrg = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Storage Power',
                        index_col=[0, 1]
                    )

                    inststrg = pd.read_excel(
                        rh_path,
                        sheet_name='Added Storage Power',
                        index_col=[0]
                    )

                    retcap_sum = pd.DataFrame(
                        [retcap.loc[['Diesel']].sum().sum(),
                         retcap.loc[['PV']].sum().sum(),
                         retcap.loc[['Wind']].sum().sum()],
                        index=['Diesel', 'PV', 'Wind']
                    )

                    instcap_sum = pd.DataFrame(
                        [instcap.loc[['Diesel']].sum().sum(),
                         instcap.loc[['PV']].sum().sum(),
                         instcap.loc[['Wind']].sum().sum()],
                        index=['Diesel', 'PV', 'Wind']
                    )

                    retstrg_sum = pd.DataFrame(
                        [retstrg.loc[['Battery']].sum().sum()],
                        index=['Battery']
                    )

                    inststrg_sum = pd.DataFrame(
                        [inststrg.loc[['Battery']].sum().sum()],
                        index=['Battery']
                    )

                    cap_ratio = retcap_sum/instcap_sum
                    strg_ratio = retstrg_sum/inststrg_sum

                    this_t = pd.concat(
                        [
                            cap_ratio,
                            strg_ratio
                        ]
                    )

                    grids.append(this_t)

            all_things.append(grids)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    colors = [red, yellow, light_blue, light_green]
    for n in range(len(all_things)):

        if n != 8:
            retireds = all_things[n]

            for i in range(len(retireds)):
                thing = retireds[i]

                h = (i+1)/4
                # h = 5
                j = 0
                for cap in ['Diesel', 'PV', 'Wind', 'Battery']:
                    # Diesel
                    axes.flat[n].barh(
                        y=h,
                        height=0.2,
                        width=1 - thing.loc[cap],
                        left=j + thing.loc[cap],
                        color=colors[j],
                    )

                    axes.flat[n].barh(
                        y=h,
                        height=0.2,
                        width=thing.loc[cap],
                        left=j,
                        color=colors[j],
                        alpha=0.5,
                        # hatch= '/'
                    )

                    j += 1

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.1525')
    axes[2][0].set_ylabel('0.035')

    axes[2][0].set_xlabel('0.03')
    axes[2][1].set_xlabel('0.08')
    axes[1][2].set_xlabel('0.15',
                          # labelpad= 205
                          )

    # axes[0][0].set_yticks(
    #     ticks= [(t+1)/4 for t in range(10)],
    #     labels= [t+1 for t in range(10)]
    #     )

    # fig.suptitle('FiT')
    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])

    # plt.xticks(
    #     ticks= []
    #     )

    # plt.xlabel('Capacity (MW)')
    # plt.ylim(4.5, 5.5)
    xlim = 4
    plt.xlim(0, xlim)
    plt.show()


def grid_npv(folder, value_name, xlim=50):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing barplots of the
    NPV for a certain objective function component for all scenarios
    except the no grid case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    value_name: str
        Name of NPV component to plot.
    xlim : float
        Limit of x-axis. The default is 50.

    Returns
    -------
    None.

    '''

    all_values = []

    for p in [0.27, 0.185, 0.1]:
        for fit in [0.03, 0.08, 0.15]:
            if fit < p:
                this_t = []
                for t in range(1, 11):

                    rh_path = os.path.join(
                        folder,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}',
                        f'RH_Tgrid = {t}_FiT = {fit}_EDLTar = {p}.xlsx'
                    )

                    npvs = pd.read_excel(
                        rh_path,
                        sheet_name='Total NPV',
                    )

                    value = npvs[value_name].loc[0]
                    this_t.append(value)

            all_values.append(this_t)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n in range(len(all_values)):

        if n != 8:
            values = all_values[n]

            for i in range(len(values)):
                thing = values[i]

                h = (i+1)/4

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing,
                    color='green',
                )

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.185')
    axes[2][0].set_ylabel('0.1')

    axes[2][0].set_xlabel('0.03')
    axes[2][1].set_xlabel('0.08')
    axes[1][2].set_xlabel('0.15',
                          )

    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])
    plt.yticks(
        ticks=[]
    )

    plt.xlim(0, xlim)
    plt.show()


def grid_value(folder, value_name, xlim=50):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing barplots of the
    value indicator (either certainty or waiting) for all scenarios except
    the no grid case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    value_name: str
        Name of NPV component to plot.
    xlim : float
        Limit of x-axis. The default is 50.

    Returns
    -------
    None.

    '''

    all_values = []

    rh_path = os.path.join(
        folder,
        'Summary',
        'NPV_Summary.xlsx'
    )

    rh_values = pd.read_excel(
        rh_path,
        sheet_name=value_name,
        index_col=[0, 1, 2]
    )*100

    for p in [
            0.27,
            0.1525,
            0.035
    ]:
        for fit in [
                0.03,
                0.08,
                0.15
        ]:
            if fit < p:
                this_t = []
                for t in range(1, 11):

                    value = rh_values.loc[(t, fit, p)]
                    this_t.append(value)

            all_values.append(this_t)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n in range(len(all_values)):

        if n != 8:
            values = all_values[n]

            for i in range(len(values)):
                thing = values[i]

                h = (i+1)/4

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing,
                    color='green',
                )

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.1525')
    axes[2][0].set_ylabel('0.035')

    axes[2][0].set_xlabel('0.03')
    axes[2][1].set_xlabel('0.08')
    axes[1][2].set_xlabel('0.15',
                          )

    # fig.suptitle('FiT')
    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])
    plt.yticks(
        ticks=[]
    )

    plt.xlim(0, xlim)
    plt.show()


def grid_ver_per_tech(folder, discount, Y=20, xlim=35):
    '''
    Generates 3 rows of 3,3, and 2 subplots, each containing stacked barplots
    of the revenues obtained from retirement of assets of each technology
    for all scenarios except the no grid case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    discount : float
        Discount % applied on valuation of assets sold.
    Y : int, optional
        Number of years in the model. The default is 20.
    xlim : float, optional
        Limit of the x-axis. The default is 35.

    Returns
    -------
    None.

    '''

    data = pd.read_excel(
        'RollingHorizonData.xlsx',
        sheet_name=None
    )

    generalParameters = data['General Parameters']

    i = generalParameters['Discount Rate'][0]
    gen_unit_capex = data['Generation Unit Capex']
    stp_unit_capex = data['Storage Power Unit Capex']
    ste_unit_capex = data['Storage Energy Unit Capex']
    gen_rem_value = data['Rm Val']
    st_rem_value = data['Rm ValS']
    discount_gen = [max(0.01, discount), discount, discount]
    discount_stp = [discount]
    discount_ste = [discount]

    all_rtrs = []

    for p in [
            0.27,
            0.185,
            0.1
    ]:
        for fit in [
                0.03,
                0.08,
                0.15
        ]:
            this_rtrs = []
            if fit < p:

                for t in range(1, 11):
                    rh_path = os.path.join(
                        folder,
                        f'T = {t}',
                        f'P = {p}',
                        f'FiT = {fit}',
                        f'RH_Tgrid = {t}_FiT = {fit}_EDLTar = {p}.xlsx'
                    )

                    cap = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Capacity',
                        index_col=[0, 1]
                    )

                    strg = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Storage Power',
                        index_col=[0, 1]
                    )

                    ste = pd.read_excel(
                        rh_path,
                        sheet_name='Retired Storage Energy',
                        index_col=[0, 1]
                    )

                    RtR_all = {}
                    for g, tech in enumerate(
                            [
                                'Diesel',
                                'PV',
                                'Wind'
                            ]
                    ):
                        RtR = 0
                        for y in range(Y):
                            for a in range(0, y):
                                RtR += (
                                    (1/(1 + i)**(y))
                                    * cap.loc[tech].iat[y, a]
                                    * gen_rem_value.iloc[g, y-a]
                                    * gen_unit_capex.iat[g, y]
                                    * (1 - discount_gen[g])
                                )

                        RtR_all[tech] = RtR

                        for s, stech in enumerate(
                                [
                                    'Battery'
                                ]
                        ):

                            RtR = 0
                            for y in range(Y):
                                for a in range(0, y):

                                    RtR += (
                                        (1/(1 + i)**(y)) *
                                        (
                                            (strg.loc[stech].iat[y, a]
                                             * stp_unit_capex.iat[s, y]
                                             * (1 - discount_stp[s])
                                             + ste.loc[stech].iat[y, a]
                                             * ste_unit_capex.iat[s, y])
                                            * (st_rem_value.iloc[s, y - a])
                                            * (1 - discount_ste[s])
                                        )
                                    )
                            RtR_all[stech] = RtR

                    RtR_df = pd.DataFrame(
                        RtR_all,
                        index=range(1)
                    ).T
                    this_rtrs.append(RtR_df)

            all_rtrs.append(this_rtrs)

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)

    for n in range(len(all_rtrs)):

        if n != 8:
            retireds = all_rtrs[n]

            for i in range(len(retireds)):
                thing = retireds[i]

                h = (i+1)/4

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Diesel'],
                    color=red,
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['PV'],
                    left=thing.loc['Diesel'],
                    color=yellow,
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Wind'],
                    left=thing.loc['Diesel'] + thing.loc['PV'],
                    color=light_blue,
                )

                axes.flat[n].barh(
                    y=h,
                    height=0.2,
                    width=thing.loc['Battery'],
                    left=thing.loc['Diesel'] +
                    thing.loc['PV'] + thing.loc['Wind'],
                    color=light_green,
                )

                if n in [2, 5, 7]:
                    axes.flat[n].tick_params(
                        which='both',
                        left=False,
                        labelleft=False,
                        right=True,
                        labelright=True,
                    )
                else:
                    axes.flat[n].tick_params(
                        which='both',
                        left=False,
                        labelleft=False,
                        right=False,
                        labelright=False
                    )
    for row in range(3):
        col = 2 if row < 2 else 1
        axes[row][col].set_yticks(
            ticks=[(i+1)/4 for i in range(10)],
            labels=range(1, 11),
            fontsize=8
        )
        axes[row][col].set_ylabel('T',
                                  rotation=0,
                                  labelpad=15,
                                  fontsize=10)
        axes[row][col].yaxis.set_label_position("right")
        axes[row][col].yaxis.set_label_coords(x=1.2, y=0.5)

    axes[0][0].set_ylabel('0.27')
    axes[1][0].set_ylabel('0.185')
    axes[2][0].set_ylabel('0.1')

    axes[2][0].set_xlabel('0.03', fontsize=10, labelpad=10)
    axes[2][0].xaxis.set_label_coords(x=0.5, y=-0.25)

    axes[2][1].set_xlabel('0.08', fontsize=10, labelpad=10)
    axes[2][1].xaxis.set_label_coords(x=0.5, y=-0.25)

    axes[1][2].set_xlabel('0.15', fontsize=10, labelpad=10)
    axes[1][2].xaxis.set_label_coords(x=0.5, y=-0.15)

    # fig.suptitle('FiT')
    fig.supylabel('P',
                  x=0.05)
    fig.supxlabel('FiT',
                  y=-0.03)

    fig.delaxes(axes[2][2])

    plt.xlim(0, xlim)

    custom_lines = [
        Line2D([0], [0], color=red, lw=4),
        Line2D([0], [0], color=yellow, lw=4),
        Line2D([0], [0], color=light_blue, lw=4),
        Line2D([0], [0], color=light_green, lw=4)
    ]

    plt.legend(custom_lines, [
        'Diesel',
        'PV',
        'Wind',
        'Battery'
    ],
        loc='lower right',
        bbox_to_anchor=(1.1, 0, 1, 1)
    )

    plt.show()
