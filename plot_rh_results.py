import plotting_functions as pl
import pandas as pd

import os


def is_inst_or_add(sheet_name):
    value = sheet_name[:9] == 'Installed' or sheet_name[:5] == 'Added'
    return value


def is_rem_or_ret(sheet_name):
    value = sheet_name[:7] == 'Retired' or sheet_name[:9] == 'Remaining'
    return value


def is_yearly_or_npv(sheet_name):
    value = sheet_name == 'Yearly Values' or sheet_name == 'Total NPV'
    return value


def plot_results(folder, fs=30, save=True, show=False,
                 plot_scenarios=True, plot_values=True):
    '''
    Plots all relevant figures for a rolling horizon of a specific input case.

    Parameters
    ----------
    folder : str
        Location of result folder.
    fs : float, optional
        Font size for ticks and labels. The default is 30.
    save : bool, optional
        To save or not to save the plots generated as png files. If true,
        images will be saved in the same folder as the results.
        The default is True.
    show : bool, optional
        Whether to show the plots being generated. The default is False.
    plot_scenarios : bool, optional
        Whether to plot NPV, installed, added/retired, and operations for all
        scenarios. The default is True.
    plot_values : bool, optional
        Whether to plot value indicator boxplots (Total NPVs, Value of
        Certainty, Value of Waiting).
        The default is True.

    Returns
    -------
    None.

    '''

    if plot_scenarios:
        # Stochastic
        stoc_path = os.path.join(
            folder,
            'Stochastic'
        )

        files = os.listdir(stoc_path)

        files = [f for f in files if os.path.isfile(
            os.path.join(stoc_path, f))]
        excel_files = [f for f in files if f.endswith(('.xls', '.xlsx'))]

        for file in excel_files:
            if file[:3] == 'S_C':
                file_path = os.path.join(stoc_path, excel_files[0])

                stoc_file = pd.ExcelFile(
                    file_path,
                    engine='calamine'
                )

            else:
                print("Excel file not found.")

        sheets_dict = {}

        for sheet_name in stoc_file.sheet_names:
            if sheet_name[:9] == 'Installed' or sheet_name[:5] == 'Added':
                df = stoc_file.parse(sheet_name, index_col=[0])
                sheets_dict[sheet_name] = df
            elif sheet_name[:7] == 'Retired' or sheet_name[:9] == 'Remaining':
                df = stoc_file.parse(sheet_name, index_col=[0, 1])
                sheets_dict[sheet_name] = df
            elif sheet_name == 'Yearly Values' or sheet_name == 'Total NPV':
                df = stoc_file.parse(sheet_name, index_col=[0])
                sheets_dict[sheet_name] = df

        pl.plot_all_decisions(
            sheets_dict,
            stoc_path,
            'S',
            save=save,
            show=show,
            fs=fs,
            generation=False
        )

        # Deterministic and Rolling Horizon
        for t in range(1, 11):
            for p in [0.1, 0.185, 0.27]:
                for fit in [0.03, 0.08, 0.15]:
                    if fit < p:
                        drh_path = os.path.join(
                            folder,
                            f'T = {t}',
                            f'P = {p}',
                            f'FiT = {fit}'
                        )

                        files = os.listdir(drh_path)

                        excel_files = [
                            f for f in files if f.endswith(('.xls', '.xlsx'))]

                        for file in excel_files:
                            if file[0] == 'D':
                                det_path = os.path.join(
                                    drh_path, files[files.index(file)])

                                det_file = pd.ExcelFile(det_path)

                            elif file[:1] == 'R':
                                rh_path = os.path.join(
                                    drh_path, files[files.index(file)])

                                rh_file = pd.ExcelFile(rh_path)
                            else:
                                print("Excel file not found.")

                        files = [
                            det_file,
                            rh_file
                        ]

                        for file in files:
                            if file == det_file:
                                model_type = 'D'
                            elif file == rh_file:
                                model_type = 'RH'

                            sheets_dict = {}

                            for sheet_name in file.sheet_names:
                                if is_inst_or_add(sheet_name):
                                    df = file.parse(sheet_name, index_col=[0])
                                    sheets_dict[sheet_name] = df
                                elif is_rem_or_ret(sheet_name):
                                    df = file.parse(
                                        sheet_name, index_col=[0, 1])
                                    sheets_dict[sheet_name] = df
                                elif is_yearly_or_npv(sheet_name):
                                    df = file.parse(sheet_name, index_col=[0])
                                    sheets_dict[sheet_name] = df

                            pl.plot_all_decisions(
                                sheets_dict,
                                drh_path,
                                model_type,
                                save=save,
                                show=show,
                                fs=fs,
                                generation=True
                            )

        no_grid_path = os.path.join(
            folder,
            'T = Never'
        )

        if os.path.exists(no_grid_path):

            files = os.listdir(no_grid_path)

            excel_files = [f for f in files if f.endswith(('.xls', '.xlsx'))]

            for file in excel_files:
                if file[0] == 'D':
                    det_path = os.path.join(
                        no_grid_path, files[files.index(file)])

                    det_file = pd.ExcelFile(det_path)

                elif file[:1] == 'R':
                    rh_path = os.path.join(
                        no_grid_path, files[files.index(file)])

                    rh_file = pd.ExcelFile(rh_path)
                else:
                    print("Excel file not found.")

            files = [
                det_file,
                rh_file
            ]

            for file in files:
                if file == det_file:
                    model_type = 'D'
                elif file == rh_file:
                    model_type = 'RH'

                sheets_dict = {}

                for sheet_name in file.sheet_names:
                    if is_inst_or_add(sheet_name):
                        df = file.parse(sheet_name, index_col=[0])
                        sheets_dict[sheet_name] = df
                    elif is_rem_or_ret(sheet_name):
                        df = file.parse(
                            sheet_name, index_col=[0, 1])
                        sheets_dict[sheet_name] = df
                    elif is_yearly_or_npv(sheet_name):
                        df = file.parse(sheet_name, index_col=[0])
                        sheets_dict[sheet_name] = df

                pl.plot_all_decisions(
                    sheets_dict,
                    no_grid_path,
                    model_type,
                    save=save,
                    show=show,
                    fs=fs,
                    generation=True
                )
    if plot_values:
        summary_path = os.path.join(
            folder,
            'Summary'
        )

        files = os.listdir(summary_path)
        excel_file = [f for f in files if f.endswith(('.xls', '.xlsx'))]

        file_path = os.path.join(summary_path, excel_file[0])

        summary_file = pd.ExcelFile(
            file_path,
            engine='calamine'
        )

        sheets_dict = {}

        for sheet_name in summary_file.sheet_names:
            df = summary_file.parse(sheet_name, index_col=[0, 1, 2])
            df.index = pd.MultiIndex.from_tuples(
                list(df.index[:-1]) + [('Never', 'None', 'None')],
                names=['T', 'FiT', 'P']
            )
            sheets_dict[sheet_name] = df

            result_name = sheet_name + (sheet_name[:5] != 'Value')*(' NPV')

            if sheet_name[:5] != 'Value':
                unit = 'm$'
            else:
                df = df*100
                unit = '%'

            pl.boxplots_by_parameter(
                df,
                out_dir=summary_path,
                result_name=result_name,
                unit=unit,
                save=save,
                show=show
            )


folder = (
    r"C:\Users\User\OneDrive - American University of Beirut"
    r"\Decentralized Planning Grid Uncertainty\Codes - Kareem\GitHub"
    r"\100% Discount\100% Discount Results"
)

path = os.path.join(
    folder,
)

plot_results(
    path,
    fs=50,
    save=True,
    show=True,
    plot_scenarios=False,
    plot_values=True
)
