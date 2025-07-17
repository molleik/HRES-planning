# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:37:42 2025

@author: Kareem Abou Jalad
"""

import gurobipy as gp
from gurobipy import quicksum
from gurobipy import GRB
import pandas as pd
import numpy as np
import math


class DeterministicModel:

    def __init__(self, input_data):
        '''
        Loads input parameters of deterministic model.

        Parameters
        ----------
        input_data : str, dict
            Pass str if reading data directly from .xlsx file. Otherwise, data
            can be passed as dict containing DataFrames.

        Returns
        -------
        None.

        '''

        if isinstance(input_data, dict):
            data = input_data
        else:
            data = pd.read_excel(input_data, sheet_name=None)

        # General Parameters
        generalParameters = data['General Parameters']
        generation = data['Generation']
        storage = data['Storage']

        # Costs
        self.i = generalParameters['Discount Rate'][0]
        self.emu_tax = generalParameters['Emissions Unit Tax'][0]
        self.VOLL = generalParameters['VOLL'][0]
        self.VOEL = generalParameters['VOEL'][0]
        self.VOUR = generalParameters['VOUR'][0]
        self.gen_unit_capex = data['Generation Unit Capex']
        self.gen_unit_fixed_opex = generation['Generation Unit Fixed Opex']
        self.gen_unit_var_opex = generation[
            'Generation Unit Non '
            'Fuel Variable Opex'
        ]
        self.fuel_price = data['Fuel Prices']
        self.stp_unit_capex = data['Storage Power Unit Capex']
        self.ste_unit_capex = data['Storage Energy Unit Capex']
        self.stp_unit_fixed_opex = storage['Storage Power Unit Fixed Opex']
        self.ste_unit_fixed_opex = storage['Storage Energy Unit Fixed Opex']
        self.st_unit_var_opex = storage['Storage Unit Non Fuel Variable Opex']
        self.gen_rem_value = data['Rm Val']
        self.st_rem_value = data['Rm ValS']
        self.discount = generation['Discount']
        self.discount_stp = storage['Storage Power Discount']
        self.discount_ste = storage['Storage Energy Discount']

        # Time
        time_frame = data['Time Frame']
        representative_period = time_frame['Representative Period']
        self.days = len(representative_period)
        self.wd = time_frame['Weight of representative period']
        self.hours = int(time_frame.loc[0, "Subperiod with a period"])
        self.years = int(time_frame.loc[0, "Years"])

        # Limitations
        self.purchase_limit = generalParameters['Purchased Limit'][0]
        self.sold_limit = generalParameters['Sold Limit'][0]
        self.capacity_limit = generation['Installed Limit']
        self.early_retirement = generalParameters['Early Retirement'][0]

        # Demand
        self.hourly_demand = data['Hourly Demand']
        self.demand_forecast = data['Demand Forecast']
        self.demand = []
        self.planningReserve = generalParameters['Planning Reserve'][0]
        self.peak_demand = self.hourly_demand.values.max()*self.demand_forecast

        # Technicalities
        self.heat_rate = generation['Heat Rate']
        self.max_storage = storage['Max Storage']
        self.min_storage = storage['Min Storage']
        self.char_eff = storage['Charging Efficiency']
        self.dischar_eff = storage['Discharging Efficiency']
        self.PEF = generation['Pollutant Emissions']
        self.gen_lifetime = generation['Life Time']
        self.st_lifetime = storage['Life Time ST']
        self.pv_cap_fac = data['PV CF']
        self.wind_cap_fac = data['Wind CF']
        self.gen_prm_factor = generation['Planning Reserve Factor']
        self.stp_prm_factor = storage['Planning Reserve Factor']
        self.max_usage = generation['Maximum Usage']

        # Indices
        self.gen_tech = generation['GT']
        self.gt_index = generation['GT Index']
        self.gen = len(self.gen_tech)
        self.diesel_index = []
        self.pv_index = []
        self.wind_index = []

        for index in self.gt_index:
            if not math.isnan(index):
                if generation['GT'][index] == 'Diesel':
                    self.diesel_index.append(int(index))
                if generation['GT'][index] == 'PV':
                    self.pv_index.append(int(index))
                if generation['GT'][index] == 'Wind':
                    self.wind_index.append(int(index))

        self.st_tech = storage['Type']
        self.st_index = storage['ST Index']
        self.strg = len(self.st_index)

    def input_scenario(self, ys=0, scenario=(0, 0, 0),
                       day_night_fit=False, day_threshold=0.05,
                       fit_factor=1):
        '''
        Sets scenario conditions for the model.

        Parameters
        ----------
        ys : int, optional
            Starting year for the model. The default is 0.
        scenario : tuple, optional
            Grid policy conditions, ordered as (T, P, FiT). The default
            is (0, 0, 0).
        day_night_fit : bool, optional
            Enables/disables variable FiT scheme. The default is False.
        day_threshold : float, optional
            Hours having a PV capacity factor higher than day_threshold are
            defined as day hours. The default is 0.05.
        fit_factor : float, optional
            Factor by which FiT is multiplied during the night. The default
            is 1.

        Returns
        -------
        None.

        '''

        self.ys = ys
        self.t_grid = scenario[0]
        self.feed_in_tar = scenario[1]
        self.purchase_tar = scenario[2]

        for y in range(self.ys, self.years):
            yearly_demand = []
            for d in range(self.days):
                daily_demand = []
                for h in range(self.hours):
                    daily_demand.append(
                        self.demand_forecast[y]
                        * self.hourly_demand.loc[d].values[h])
                yearly_demand.append(daily_demand)
            self.demand.append(yearly_demand)

        if day_night_fit:
            self.hourly_fit = self.pv_cap_fac.mask(self.pv_cap_fac
                                                   < day_threshold,
                                                   self.feed_in_tar*fit_factor)

            self.hourly_fit = self.hourly_fit.mask(
                self.hourly_fit
                != self.feed_in_tar*fit_factor,
                self.feed_in_tar)
        else:
            self.hourly_fit = pd.DataFrame(self.feed_in_tar,
                                           index=range(self.days),
                                           columns=range(self.hours)
                                           )

    def input_installations(self, input_dict):
        '''
        Loads remaining capacity from previous output as input to the model.

        Parameters
        ----------
        input_dict : dict
            Output from previous model. Contains remaining installations
            DataFrames for capacity and storage.

        Returns
        -------
        None.

        '''

        input_rem_cap = input_dict['Remaining Capacity']
        input_rem_stp = input_dict['Remaining Storage Power']
        input_rem_ste = input_dict['Remaining Storage Energy']

        self.input_cap = pd.DataFrame()
        self.input_stp = pd.DataFrame()
        self.input_ste = pd.DataFrame()

        for g in range(self.gen):
            self.input_cap = pd.concat(
                [
                    self.input_cap,
                    input_rem_cap.loc[
                        [
                            (
                                self.gen_tech[g],
                                self.ys - 1
                            )
                        ],
                        :
                    ]
                ]
            )

        for s in range(self.strg):
            self.input_stp = pd.concat(
                [
                    self.input_stp,
                    input_rem_stp.loc[
                        [(self.st_tech[s],
                          self.ys - 1)], :
                    ]
                ])

            self.input_ste = pd.concat([
                self.input_ste,
                input_rem_ste.loc[
                    [(self.st_tech[s],
                      self.ys - 1)], :
                ]
            ])

    def det_solve(self):
        '''
        Solves model.

        Returns
        -------
        None.

        '''

        THOUSAND = 1000
        MILLION = 1000000

        self.m = gp.Model('Installation')
        self.m.Params.DualReductions = 0

        # Decision Variables

        # Installations
        self.InstCap = self.m.addVars(
            [(g, y) for g in range(self.gen)
             for y in range(self.ys, self.years)],
            vtype=gp.GRB.CONTINUOUS,
            lb=0,
            name='Installed Capacity'
        )

        self.AddCap = self.m.addVars([(g, y)
                                      for g in range(self.gen)
                                      for y in range(self.ys, self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Added Capacity')

        self.RemCap = self.m.addVars([(g, y, a)
                                      for g in range(self.gen)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Remaining Capacity')

        self.RetCap = self.m.addVars([(g, y, a)
                                      for g in range(self.gen)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Retired Capacity')

        self.InstStP = self.m.addVars([(s, y)
                                       for s in range(self.strg)
                                       for y in range(self.ys, self.years)
                                       ],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Installed Storage Power')

        self.AddStP = self.m.addVars([(s, y)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Added Storage Power')

        self.RemStP = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Remaining Storage Power')

        self.RetStP = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Retired Storage Power')

        self.InstStE = self.m.addVars([(s, y)
                                       for s in range(self.strg)
                                       for y in range(self.ys, self.years)
                                       ],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Installed Storage Energy')

        self.AddStE = self.m.addVars([(s, y)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Added Storage Energy')

        self.RemStE = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Remaining Storage Energy')

        self.RetStE = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Retired Storage Energy')

        # Dispatching
        self.DispElec = self.m.addVars([(g, y, d, h)
                                        for g in range(self.gen)
                                        for y in range(self.ys, self.years)
                                        for d in range(self.days)
                                        for h in range(self.hours)
                                        ],
                                       vtype=gp.GRB.CONTINUOUS, lb=0,
                                       name='Dispatched Electricity')

        self.USDem = self.m.addVars([(y, d, h)
                                     for y in range(self.ys, self.years)
                                     for d in range(self.days)
                                     for h in range(self.hours)
                                     ],
                                    vtype=gp.GRB.CONTINUOUS, lb=0,
                                    name='Unserved Demand')

        self.Charging = self.m.addVars([(s, y, d, h)
                                        for s in range(self.strg)
                                        for y in range(self.ys, self.years)
                                        for d in range(self.days)
                                        for h in range(self.hours)
                                        ],
                                       vtype=gp.GRB.CONTINUOUS, lb=0,
                                       name='Charging')

        self.SoldGen = self.m.addVars([(g, y, d, h)
                                       for g in range(self.gen)
                                       for y in range(self.ys, self.years)
                                       for d in range(self.days)
                                       for h in range(self.hours)
                                       ],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Sold (Generated)')

        self.SoldStorage = self.m.addVars([(s, y, d, h)
                                           for s in range(self.strg)
                                           for y in range(self.ys, self.years)
                                           for d in range(self.days)
                                           for h in range(self.hours)
                                           ],
                                          vtype=gp.GRB.CONTINUOUS, lb=0,
                                          name='Sold (Stored)')

        self.Discharging = self.m.addVars([(s, y, d, h)
                                           for s in range(self.strg)
                                           for y in range(self.ys, self.years)
                                           for d in range(self.days)
                                           for h in range(self.hours)
                                           ],
                                          vtype=gp.GRB.CONTINUOUS, lb=0,
                                          name='Discharging')

        self.ExElec = self.m.addVars([(y, d, h)
                                      for y in range(self.ys, self.years)
                                      for d in range(self.days)
                                      for h in range(self.hours)
                                      ],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Excess Electricity')

        self.PurElec = self.m.addVars([(y, d, h)
                                       for y in range(self.ys, self.years)
                                       for d in range(self.days)
                                       for h in range(self.hours)
                                       ],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Purchased Electricity')

        self.SoC = self.m.addVars([(s, y, d, h)
                                   for s in range(self.strg)
                                   for y in range(self.ys, self.years)
                                   for d in range(self.days)
                                   for h in range(self.hours)],
                                  vtype=gp.GRB.CONTINUOUS, lb=0,
                                  name='State of Charge')

        self.USRes = self.m.addVars([(y)
                                     for y in range(self.years)
                                     ],
                                    vtype=gp.GRB.CONTINUOUS, lb=0,
                                    name='Unsatisfied Reserve')

        # Objective Function Components
        # Capex
        self.yearly_capex = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_capex[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * quicksum(
                    self.gen_unit_capex.iat[g, y + self.ys]
                    * self.AddCap[g, y + self.ys]
                    for g in range(self.gen)
                )
                + quicksum(
                    self.stp_unit_capex.iat[s, y + self.ys]
                    * self.AddStP[s, y + self.ys]
                    +
                    self.ste_unit_capex.iat[s, y + self.ys]
                    * self.AddStE[s, y + self.ys]
                    for s in range(self.strg)
                )
            )

        self.total_capex = sum(self.yearly_capex)

        # Fixed Opex
        self.yearly_fixed_opex = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_fixed_opex[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * quicksum(
                    self.gen_unit_fixed_opex[g]
                    * self.InstCap[g, y + self.ys]
                    for g in range(self.gen)
                )
                + quicksum(
                    self.stp_unit_fixed_opex[s]
                    * self.InstStP[s, y + self.ys]
                    +
                    self.ste_unit_fixed_opex[s]
                    * self.InstStE[s, y + self.ys]
                    for s in range(self.strg)
                )
            )

        self.total_fixed_opex = sum(self.yearly_fixed_opex)

        # Variable Opex
        self.yearly_var_opex = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_var_opex[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * quicksum(
                    self.gen_unit_var_opex[g]
                    * quicksum(
                        self.wd[d]
                        * quicksum(
                            self.DispElec[g, y + self.ys, d, h]
                            for h in range(self.hours)
                        )
                        for d in range(self.days)
                    )
                    for g in range(self.gen)
                )
                + quicksum(self.st_unit_var_opex[s]
                           * quicksum(
                    self.wd[d]
                    * quicksum(
                        self.Discharging[s, y + self.ys, d, h]
                        for h in range(self.hours)
                    )
                    for d in range(self.days)
                )
                    for s in range(self.strg)
                )
            )

        self.total_var_opex = sum(self.yearly_var_opex)

        # Fuel opex
        self.yearly_fuel_opex = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_fuel_opex[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * quicksum(
                    self.heat_rate[g]
                    * self.fuel_price.iloc[0, y + self.ys]
                    * quicksum(
                        self.wd[d]
                        * quicksum(
                            self.DispElec[g, y + self.ys, d, h]
                            for h in range(self.hours)
                        )
                        for d in range(self.days)
                    )
                    for g in self.diesel_index)
            )/MILLION

        self.total_fuel_opex = sum(self.yearly_fuel_opex)

        # Emissions Tax
        self.yearly_em_tax = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_em_tax[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * quicksum(
                    self.heat_rate[g]
                    * self.PEF[g]
                    * self.emu_tax
                    * quicksum(
                        self.wd[d]
                        * quicksum(
                            self.DispElec[g, y + self.ys, d, h]
                            for h in range(self.hours)
                        )
                        for d in range(self.days)
                    )
                    for g in range(self.gen)
                )
            )/THOUSAND/MILLION

        self.total_em_tax = sum(self.yearly_em_tax)

        # Lost Load
        self.yearly_llc = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_llc[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * self.VOLL
                * quicksum(
                    self.wd[d]
                    * quicksum(
                        self.USDem[y + self.ys, d, h]
                        for h in range(self.hours)
                    )
                    for d in range(self.days)
                )
            )

        self.total_llc = sum(self.yearly_llc)

        # Excess load
        self.yearly_elc = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_elc[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * self.VOEL
                * quicksum(
                    self.wd[d]
                    * quicksum(
                        self.ExElec[y + self.ys, d, h]
                        for h in range(self.hours)
                    )
                    for d in range(self.days)
                )
            )/MILLION

        self.total_elc = sum(self.yearly_elc)

        # Purchased Electricity
        self.yearly_pec = [0 for y in range(self.ys, self.years)]

        for y in range(self.years - self.ys):
            self.yearly_pec[y] = (
                (1/(1 + self.i)**(y + self.ys))
                * self.purchase_tar
                * quicksum(
                    self.wd[d]
                    * quicksum(
                        self.PurElec[y + self.ys, d, h]
                        for h in range(self.hours)
                    )
                    for d in range(self.days)
                )
            )/THOUSAND

        self.total_pec = sum(
            self.yearly_pec
        )

        # Electricity Revenues
        self.yearly_elr = [0 for y in range(self.ys, self.years)]

        for y in range(self.years - self.ys):
            self.yearly_elr[y] = (
                (1/(1+self.i)**(y + self.ys))
                * (
                    quicksum(
                        quicksum(
                            self.wd[d] * quicksum(
                                self.SoldGen[g, y + self.ys, d, h]
                                * self.hourly_fit.iat[d, h]
                                for h in range(self.hours)
                            )
                            for d in range(self.days)
                        )
                        for g in range(self.gen)
                    )
                    +
                    quicksum(
                        quicksum(
                            self.wd[d] * quicksum(
                                self.SoldStorage[s, y + self.ys, d, h]
                                * self.hourly_fit.iat[d, h]
                                for h in range(self.hours)
                            )
                            for d in range(self.days)
                        )
                        for s in range(self.strg)
                    )
                )
            ) / THOUSAND

        self.total_elr = sum(self.yearly_elr)

        # Retirement Revenues
        self.yearly_rtr = [0 for y in range(self.ys, self.years)]

        if self.early_retirement == 1:
            for y in range(self.years - self.ys):
                self.yearly_rtr[y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * (quicksum(
                        quicksum(
                            self.RetCap[g, y + self.ys, a]
                            * self.gen_rem_value.iloc[g, y + self.ys - a]
                            * self.gen_unit_capex.iat[g, y + self.ys]
                            * (1 - self.discount.iloc[g])
                            for a in range(0, y + self.ys)
                        )
                        for g in range(self.gen)
                    )
                        + quicksum(
                            quicksum(
                                (self.RetStP[s, y + self.ys, a]
                                 * self.stp_unit_capex.iat[s, y + self.ys]
                                 * (1 - self.discount_stp.iloc[s])
                                 + self.RetStE[s, y + self.ys, a]
                                 * self.ste_unit_capex.iat[s, y + self.ys])
                                * (self.st_rem_value.iloc[s, y + self.ys - a])
                                * (1 - self.discount_ste.iloc[s])
                                for a in range(0, y + self.ys)
                            )
                            for s in range(self.strg)
                    )
                    )
                )

        self.total_rtr = sum(self.yearly_rtr)

        # Salvage Value
        self.yearly_sv = [0 for y in range(self.ys, self.years)]

        self.yearly_sv[-1] = (
            (1/((1 + self.i)**(self.years)))
            * (
                quicksum(
                    quicksum(
                        self.RemCap[g, self.years-1, a]
                        * (self.gen_rem_value.iloc[g, self.years-a])
                        * (self.gen_unit_capex.iat[g, self.years])
                        * (1 - self.discount.iloc[g])
                        for a in range(0, self.years)
                    )
                    for g in range(self.gen)
                )
                + quicksum(
                    (self.RemStP[s, self.years-1, a]
                     * self.stp_unit_capex.iat[s, self.years]
                     * (1 - self.discount_stp.iloc[s])
                     + self.RemStE[s, self.years-1, a]
                     * self.ste_unit_capex.iat[s, self.years]
                     * (1 - self.discount_ste.iloc[s])
                     )
                    * self.st_rem_value.iloc[s, self.years-a]
                    for a in range(0, self.years)
                    for s in range(self.strg)
                )
            )
        )

        self.total_sv = sum(self.yearly_sv)

        # Unsatisfied Reserve
        self.yearly_urc = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_urc[y] = (
                (1/((1 + self.i)**(y + self.ys)))
                * self.USRes[y + self.ys]
                * self.VOUR
            )

        self.total_urc = sum(self.yearly_urc)

        # NPV
        self.yearly_npv = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.yearly_npv[y] = (
                self.yearly_capex[y]
                + self.yearly_fixed_opex[y]
                + self.yearly_var_opex[y]
                + self.yearly_fuel_opex[y]
                + self.yearly_em_tax[y]
                + self.yearly_llc[y]
                + self.yearly_elc[y]
                + self.yearly_pec[y]
                + self.yearly_urc[y]
                - (
                    self.yearly_elr[y]
                    + self.yearly_sv[y]
                    + self.yearly_rtr[y]
                )
            )

        # Constraints

        # Installed Capacity Initial Constraint
        if self.ys == 0:
            self.m.addConstrs(
                (
                    self.InstCap[g, self.ys]
                    == self.AddCap[g, self.ys]
                    for g in range(self.gen)
                ),
                name='Installed Capacity Initial Constraint'
            )
        else:
            self.m.addConstrs(
                (
                    self.InstCap[g, self.ys]
                    == quicksum(
                        self.input_cap.iat[g, a]
                        for a in range(0, self.ys)
                    )
                    + self.AddCap[g, self.ys]
                    - quicksum(
                        self.RetCap[g, self.ys, a]
                        for a in range(0, self.ys)
                    )
                    for g in range(self.gen)
                ),
                name='Installed Capacity Initial Constraint'
            )

        # Installed Capacity Constraint
        self.m.addConstrs(
            (
                self.InstCap[g, y]
                == self.InstCap[g, y-1]
                + self.AddCap[g, y]
                - (
                    quicksum(
                        self.RetCap[g, y, a]
                        for a in range(0, y)
                    )
                )
                for g in range(self.gen)
                for y in range(self.ys+1, self.years)
            ),
            name='Installed Capacity Constraint'
        )

        # Installed Capacity Limit Constraint
        self.m.addConstrs(
            (
                self.InstCap[g, y]
                <= self.capacity_limit[g]
                for g in range(self.gen)
                for y in range(self.ys, self.years)
            ),
            name='Installed Capacity Limit'
        )

        # Remaining Capacity Constraints
        self.m.addConstrs(
            (
                self.RemCap[g, self.ys, a]
                == self.input_cap.iat[g, a]
                - self.RetCap[g, self.ys, a]
                for g in range(self.gen)
                for a in range(0, self.ys)
            ),
            name='Remaining Capacity Initialization Constraint'
        )

        self.m.addConstrs(
            (
                self.RemCap[g, y, a]
                == self.RemCap[g, y-1, a]
                - self.RetCap[g, y, a]
                for g in range(self.gen)
                for y in range(self.ys+1, self.years)
                for a in range(0, y)
            ),
            name='Remaining Capacity Calculation Constraint'
        )

        self.m.addConstrs(
            (
                self.RemCap[g, y, y]
                == self.AddCap[g, y]
                for g in range(self.gen)
                for y in range(self.ys, self.years)
            ),
            name='Remaining Capacity Logical Constraint 1'
        )

        self.m.addConstrs(
            (
                self.RemCap[g, y, a]
                == 0
                for g in range(self.gen)
                for y in range(self.ys, self.years)
                for a in range(y+1, self.years)
            ),
            name='Remaining Capacity Logical Constraint 2'
        )

        # Retirement Initial Constraint
        self.m.addConstrs(
            (
                self.RetCap[g, y, a] == 0
                for g in range(self.gen)
                for y in range(self.ys, self.years)
                for a in range(y, self.years)),
            name='Retirement Initial Constraint'
        )

        # Retirement Capacity Constraints
        self.m.addConstrs(
            (
                self.RemCap[g, y, y-self.gen_lifetime[g]] == 0
                for g in range(self.gen)
                for y in range(max(self.gen_lifetime[g], self.ys), self.years)
            ),
            name='Retirement End of Lifetime Constraint'
        )

        if self.early_retirement == 0:
            self.m.addConstrs(
                (
                    self.RetCap[g, y, y - self.gen_lifetime[g]]
                    == self.AddCap[g, y - self.gen_lifetime[g]]
                    for g in range(self.gen)
                    for y in range(self.ys + self.gen_lifetime[g], self.years)
                ),
                name='Disposal of Newly Added Capacity'
            )

            self.m.addConstrs(
                (
                    self.RetCap[g, y, y - self.gen_lifetime[g]]
                    == self.input_cap.iat[g, y - self.gen_lifetime[g]]
                    for g in range(self.gen)
                    for y in range(
                        max(self.gen_lifetime[g], self.ys),
                        min(self.ys
                            + self.gen_lifetime[g],
                            self.years
                            ),
                    )
                ),
                name='Disposal of Pre-Installed Capacity'
            )

        # Storage Power

        # Installed Storage Power Initial Constraint
        if self.ys == 0:
            self.m.addConstrs(
                (
                    self.InstStP[s, self.ys]
                    == self.AddStP[s, self.ys]
                    for s in range(self.strg)
                ),
                name='Installed Storage Power Initial Constraint'
            )

        else:

            self.m.addConstrs(
                (
                    self.InstStP[s, self.ys]
                    ==
                    quicksum(
                        self.input_stp.iat[s, a]
                        for a in range(0, self.ys))
                    + self.AddStP[s, self.ys]
                    - (
                        quicksum(
                            self.RetStP[s, self.ys, a]
                            for a in range(0, self.ys)
                        )
                    )
                    for s in range(self.strg)
                ),
                name='Installed Storage Power Constraint ys'
            )

        # Installed Storage Power Constraint
        self.m.addConstrs((self.InstStP[s, y]
                           == self.InstStP[s, y-1]
                           + self.AddStP[s, y]
                           - (quicksum(self.RetStP[s, y, a]
                                       for a in range(0, y)))
                           for s in range(self.strg)
                           for y in range(self.ys+1, self.years)),
                          name='Installed Storage Power Constraint')

        # Installed Storage Power Limit Constraint
        self.m.addConstrs(
            (
                self.InstStP[s, y]
                <= 1000
                for s in range(self.strg)
                for y in range(self.ys, self.years)
            ),
            name='Installed Storage Power Limit'
        )

        # Remaining Storage Power Constraints
        self.m.addConstrs((self.RemStP[s, self.ys, a]
                           == self.input_stp.iat[s, a]
                           - self.RetStP[s, self.ys, a]
                           for s in range(self.strg)
                           for a in range(0, self.ys)),
                          name='Remaining Storage Power Constraint 1')

        self.m.addConstrs((self.RemStP[s, y, a]
                           == self.RemStP[s, y-1, a]
                           - self.RetStP[s, y, a]
                           for s in range(self.strg)
                           for y in range(self.ys+1, self.years)
                           for a in range(0, y)),
                          name='Remaining Storage Power Constraint 2')

        self.m.addConstrs((self.RemStP[s, y, y]
                           == self.AddStP[s, y]
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)),
                          name='Remaining Storage Power Constraint 3')

        self.m.addConstrs((self.RemStP[s, y, a]
                           == 0
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for a in range(y+1, self.years)),
                          name='Remaining Storage Power Constraint 4')

        # Retirement Storage Power Initial Constraint
        self.m.addConstrs((self.RetStP[s, y, a] == 0
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for a in range(y, self.years)),
                          name='Retirement Storage Power Initial Constraint')

        # Retirement Storage Power Constraints
        self.m.addConstrs(
            (self.RemStP[s, y, y-self.st_lifetime[s]] == 0
             for s in range(self.strg)
             for y in range(max(self.st_lifetime[s], self.ys), self.years)
             ),
            name='Retirement End of Lifetime Constraint StP')

        if self.early_retirement == 0:
            self.m.addConstrs((self.RetStP[s, y, y - self.st_lifetime[s]]
                               == self.AddStP[s, y - self.st_lifetime[s]]
                               for s in range(self.strg)
                               for y in range(self.ys + self.st_lifetime[s],
                                              self.years
                                              )
                               ),
                              name='Disposal of Newly Added Storage Power'
                              )

            self.m.addConstrs((self.RetStP[s, y, y - self.st_lifetime[s]]
                               == self.input_stp.iat[s,
                                                     y - self.st_lifetime[s]
                                                     ]
                               for s in range(self.strg)
                               for y in range(max(self.st_lifetime[s],
                                                  self.ys
                                                  ),
                                              min(self.ys
                                                  + self.st_lifetime[s],
                                                  self.years
                                                  ),
                                              )
                               ),
                              name='Disposal of Pre-Installed Storage Power'
                              )

        # Storage Energy

        # Installed Storage Energy Initial Constraint
        if self.ys == 0:
            self.m.addConstrs(
                (self.InstStE[s, self.ys]
                 == self.AddStE[s, self.ys]
                 for s in range(self.strg)),
                name='Installed Storage Energy Initial Constraint')
        else:
            self.m.addConstrs((self.InstStE[s, self.ys]
                               ==
                               quicksum(self.input_ste.iat[s, a]
                                        for a in range(0, self.ys))
                               + self.AddStE[s, self.ys]
                               - (quicksum(self.RetStE[s, self.ys, a]
                                           for a in range(0, self.ys)))
                               for s in range(self.strg)),
                              name='Installed Storage Energy Constraint')

        # Installed Storage Energy Constraint
        self.m.addConstrs((self.InstStE[s, y]
                           == self.InstStE[s, y-1]
                          + self.AddStE[s, y]
                          - (quicksum(self.RetStE[s, y, a]
                                      for a in range(0, y)))
                           for s in range(self.strg)
                           for y in range(self.ys+1, self.years)),
                          name='Installed Storage Energy Constraint')

        # Installed Storage Energy Limit Constraint
        self.m.addConstrs(
            (
                self.InstStE[s, y]
                <= 1000
                for s in range(self.strg)
                for y in range(self.ys, self.years)
            ),
            name='Installed Storage Energy Limit'
        )

        # Remaining Storage Energy Constraints
        self.m.addConstrs((self.RemStE[s, self.ys, a]
                           == self.input_ste.iat[s, a]
                           - self.RetStE[s, self.ys, a]
                           for s in range(self.strg)
                           for a in range(0, self.ys)),
                          name='Remaining Storage Energy Constraint 1')

        self.m.addConstrs((self.RemStE[s, y, a]
                           == self.RemStE[s, y-1, a]
                           - self.RetStE[s, y, a]
                           for s in range(self.strg)
                           for y in range(self.ys+1, self.years)
                           for a in range(0, y)),
                          name='Remaining Storage Energy Constraint 2')

        self.m.addConstrs((self.RemStE[s, y, y]
                           == self.AddStE[s, y]
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)),
                          name='Remaining Storage Energy Constraint 3')

        self.m.addConstrs((self.RemStE[s, y, a]
                           == 0
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for a in range(y+1, self.years)),
                          name='Remaining Storage Energy Constraint 4')

        # Retirement Storage Energy Initial Constraint
        self.m.addConstrs((self.RetStE[s, y, a] == 0
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for a in range(y, self.years)),
                          name='Retirement Storage Energy Initial Constraint')

        # Retirement Storage Energy Constraints
        self.m.addConstrs(
            (self.RemStE[s, y, y-self.st_lifetime[s]]
             == 0
             for s in range(self.strg)
             for y in range(max(self.st_lifetime[s], self.ys), self.years)),
            name='Retirement End of Lifetime Constraint StE')

        if self.early_retirement == 0:
            self.m.addConstrs((self.RetStE[s, y, y - self.st_lifetime[s]]
                               == self.AddStE[s, y - self.st_lifetime[s]]
                               for s in range(self.strg)
                               for y in range(self.ys + self.st_lifetime[s],
                                              self.years
                                              )
                               ),
                              name='Disposal of Newly Added Storage Energy'
                              )

            self.m.addConstrs((self.RetStE[s, y, y - self.st_lifetime[s]]
                               == self.input_ste.iat[s,
                                                     y - self.st_lifetime[s]
                                                     ]
                               for s in range(self.strg)
                               for y in range(max(self.st_lifetime[s],
                                                  self.ys
                                                  ),
                                              min(self.ys
                                                  + self.st_lifetime[s],
                                                  self.years
                                                  ),
                                              )
                               ),
                              name='Disposal of Pre-Installed Storage Energy'
                              )

        # Operational Constraints

        # Purchased Electricity Pre-Grid Constraint
        self.m.addConstrs((self.PurElec[y, d, h] == 0
                           for y in range(self.ys, self.t_grid)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Purchased Electricity')

        # Sold Electricity Pre-Grid Constraints
        self.m.addConstrs((self.SoldGen[g, y, d, h] == 0
                           for g in range(self.gen)
                           for y in range(self.ys, self.t_grid)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Sold Electricity')

        self.m.addConstrs((self.SoldStorage[s, y, d, h] == 0
                           for s in range(self.strg)
                           for y in range(self.ys, self.t_grid)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Sold Electricity')

        # Selling Post-Grid Constraint
        self.m.addConstrs((
            self.SoldGen[g, y, d, h] <= self.DispElec[g, y, d, h]
            for g in range(self.gen)
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)),
            name='Sold Generated Electricity'
        )

        self.m.addConstrs((
            self.SoldStorage[s, y, d, h] <= self.Discharging[s, y, d, h]
            for s in range(self.strg)
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)),
            name='Sold Stored Electricity'
        )

        # Sold Electricity Limit
        self.m.addConstrs((
            quicksum(self.SoldGen[g, y, d, h] for g in range(self.gen))
            + quicksum(self.SoldStorage[s, y, d, h] for s in range(self.strg))
            <= self.sold_limit
            for y in range(max(self.t_grid, self.ys), self.years)
            for d in range(self.days)
            for h in range(self.hours)),
            name='Sold Electricity Amount')

        # Purchased Electricity Limit
        self.m.addConstrs((self.PurElec[y, d, h]
                           <= self.purchase_limit
                           for y in range(max(
                               self.t_grid, self.ys), self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Purchased Electricity Amount')

        # Energy Balance Constraint
        self.m.addConstrs(((self.demand_forecast.iloc[0, y]
                            * self.hourly_demand.iloc[d, h])
                           - self.USDem[y, d, h]
                           + quicksum(self.Charging[s, y, d, h]
                                      for s in range(self.strg))
                           + quicksum(self.SoldGen[g, y, d, h]
                                      for g in range(self.gen))
                           + quicksum(self.SoldStorage[s, y, d, h]
                                      for s in range(self.strg))
                           == quicksum(self.DispElec[g, y, d, h]
                                       for g in range(self.gen))
                           + quicksum(self.Discharging[s, y, d, h]
                                      for s in range(self.strg))
                           - self.ExElec[y, d, h]
                           + self.PurElec[y, d, h]
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Energy Balance Equation Constraint')

        # Maximum Usage
        self.m.addConstrs((quicksum(self.wd[d]
                                    * quicksum(self.DispElec[g, y, d, h]
                                               for h in range(self.hours))
                                    for d in range(self.days))
                           <= self.max_usage.loc[g]*self.InstCap[g, y]
                           * 365*24
                           for g in range(self.gen)
                           for y in range(self.ys, self.years)
                           ),
                          name='Maximum Usage Constraint')

        # Dispacthing
        self.m.addConstrs((self.DispElec[g, y, d, h]
                           <= self.InstCap[g, y]
                           for g in self.diesel_index
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Dispatched Diesel Capacity')

        self.m.addConstrs((self.DispElec[g, y, d, h]
                           <= self.InstCap[g, y] * self.pv_cap_fac.iloc[d, h]
                           for g in self.pv_index
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Dispatched PV Capacity')

        self.m.addConstrs((self.DispElec[g, y, d, h]
                           <= self.InstCap[g, y] * self.wind_cap_fac.iloc[d, h]
                           for g in self.wind_index
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Dispatched Wind Capacity')

        # Electricity Sold from Diesel Constraint
        self.m.addConstrs((self.SoldGen[g, y, d, h] == 0
                           for g in self.diesel_index
                           for y in range(max(
                               self.t_grid, self.ys), self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Sold Electricity Diesel Constraint')

        # Unserved Demand Constraint
        self.m.addConstrs((
            self.USDem[y, d, h] <= (self.demand_forecast.iloc[0, y]
                                    * self.hourly_demand.iloc[d, h])
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)),
            name='Unserved Demand Limit'
        )

        # Max Charging
        self.m.addConstrs((self.Charging[s, y, d, h]
                           <= self.InstStP[s, y]
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Max Charging')

        # Max Discharging
        self.m.addConstrs((self.Discharging[s, y, d, h]
                           <= self.InstStP[s, y]
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Max Discharging')

        # Max Storage
        self.m.addConstrs((self.SoC[s, y, d, h]
                           <= self.InstStE[s, y]*self.max_storage[s]
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Max Storage')

        # Min Storage
        self.m.addConstrs((self.SoC[s, y, d, h]
                           >= self.InstStE[s, y]*self.min_storage[s]
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Min Storage')

        # State of Charge
        self.m.addConstrs((self.SoC[s, y, d, (h)]
                           == self.SoC[s, y, d, h-1]
                           + (
                               (
                                   self.Charging[s, y, d, h-1]*self.char_eff[s]
                               )
                               - (
                                   self.Discharging[s, y, d, h-1]
                                   / self.dischar_eff[s]
                               )
        )
            for s in range(self.strg)
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(1, self.hours)),
            name='State of Charge')

        self.m.addConstrs((self.SoC[s, y, d, 0]
                           == self.SoC[s, y, d, 23]
                           + (self.Charging[s, y, d, 23]*self.char_eff[s])
                           - (self.Discharging[s, y, d,
                              23]/self.dischar_eff[s])
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)),
                          name='State of Charge Balance')

        # Planning Reserve
        if self.ys < self.t_grid:
            self.m.addConstrs(
                (quicksum(self.InstCap[g, y]*self.gen_prm_factor[g]
                          for g in range(self.gen))
                 + quicksum(self.InstStP[s, y]*self.stp_prm_factor[s]
                            for s in range(self.strg))
                 + self.USRes[y]
                 >= (1 + self.planningReserve)
                 * self.peak_demand.iloc[0, y]
                 for y in range(self.ys, self.t_grid)
                 ),
                name='Planning Reserve Constraint pre-grid')

        self.m.addConstrs((quicksum(self.InstCap[g, y]*self.gen_prm_factor[g]
                                    for g in range(self.gen))
                           + quicksum(self.InstStP[s, y]*self.stp_prm_factor[s]
                                      for s in range(self.strg))
                           + self.USRes[y]
                           + self.purchase_limit
                           >= (1 + self.planningReserve)
                           * self.peak_demand.iloc[0, y]
                           for y in range(max(
                               self.t_grid, self.ys), self.years)
                           ),
                          name='Planning Reserve Constraint post-grid')

        # Objective Function
        self.m.setObjective(sum(self.yearly_npv),
                            GRB.MINIMIZE)

        self.m.optimize()

    def det_get_npv(self, yl):
        '''
        Helper function for det_store_decisions. Stores yearly and total NPV
        over years ys to yl.

        Parameters
        ----------
        yl : int
            NPV is stored up to year yl.

        Returns
        -------
        yearly_npv_df : DataFrame
            Yearly NPV of each objective function component.
        total_npv_df : DataFrame
            Total NPV of each objective function component.

        '''

        # Total NPV
        if yl == self.years:
            salvage = self.yearly_sv[:yl-self.ys -
                                     1] + [self.yearly_sv[-1].getValue()]
        else:
            salvage = self.yearly_sv[:yl-self.ys]

        if self.early_retirement == 1:
            earlyRet = [self.yearly_rtr[y].getValue()
                        for y in range(yl - self.ys)]
        else:
            earlyRet = [0 for y in range(yl - self.ys)]

        yearly_npv_data = {
            "Total Value": [self.yearly_npv[y].getValue()
                            for y in range(yl - self.ys)],
            "Capex": [self.yearly_capex[y].getValue()
                      for y in range(yl - self.ys)],
            "Fixed Opex": [self.yearly_fixed_opex[y].getValue()
                           for y in range(yl - self.ys)],
            "Variable Opex": [self.yearly_var_opex[y].getValue()
                              for y in range(yl - self.ys)],
            "Fuel Opex": [self.yearly_fuel_opex[y].getValue()
                          for y in range(yl - self.ys)],
            "Emissions Tax": [self.yearly_em_tax[y].getValue()
                              for y in range(yl - self.ys)],
            "Lost Load Cost": [self.yearly_llc[y].getValue()
                               for y in range(yl - self.ys)],
            "Excess Load Cost": [self.yearly_elc[y].getValue()
                                 for y in range(yl - self.ys)],
            "Purchased Electricity Cost": [self.yearly_pec[y].getValue()
                                           for y in range(yl - self.ys)],
            'Unsatisfied Reserve Cost': [self.yearly_urc[y].getValue()
                                         for y in range(yl - self.ys)],
            "Electricity Revenues": [self.yearly_elr[y].getValue()
                                     for y in range(yl - self.ys)],
            "Retirement Revenues": earlyRet,
            "Salvage Value": salvage
        }

        yearly_npv_df = pd.DataFrame(yearly_npv_data)

        if self.early_retirement == 1:
            totalEarlyRet = sum(earlyRet)
        else:
            totalEarlyRet = 0

        total_npv_data = {
            "Total Value": [sum(
                self.yearly_npv[:yl - self.ys]).getValue()],
            "Capex": [sum(
                self.yearly_capex[:yl - self.ys]).getValue()],
            "Fixed Opex": [sum(
                self.yearly_fixed_opex[:yl - self.ys]).getValue()],
            "Variable Opex": [sum(
                self.yearly_var_opex[:yl - self.ys]).getValue()],
            "Fuel Opex": [sum(
                self.yearly_fuel_opex[:yl - self.ys]).getValue()],
            "Emissions Tax": [sum(
                self.yearly_em_tax[:yl - self.ys]).getValue()],
            "Lost Load Cost": [sum(
                self.yearly_llc[:yl - self.ys]).getValue()],
            "Excess Load Cost": [sum(
                self.yearly_elc[:yl - self.ys]).getValue()],
            "Purchased Electricity Cost": [sum(
                self.yearly_pec[:yl - self.ys]).getValue()],
            'Unsatisfied Reserve Cost': [sum(
                self.yearly_urc[y].getValue() for y in range(yl - self.ys))],
            "Electricity Revenues": [sum(
                self.yearly_elr[:yl - self.ys]).getValue()],
            "Retirement Revenues": totalEarlyRet,
            "Salvage Value": sum(
                salvage)
        }

        total_npv_df = pd.DataFrame(total_npv_data)

        return yearly_npv_df, total_npv_df

    def det_store_decisions(self, output, yl=20):
        '''
        Stores decisions and results in dictionary up until a certain year.

        Parameters
        ----------
        yl : int, optional
            Year up until which decisions are stored. The default is 20.

        Returns
        -------
        None.

        '''

        # Scenario
        output['ys'] = self.ys

        scenario_data = {
            'T': [self.t_grid],
            'FiT': [self.feed_in_tar],
            'P': [self.purchase_tar]
        }

        scenario_df = pd.DataFrame(scenario_data)

        output['Scenario'] = scenario_df

        gy_index = pd.MultiIndex.from_product([self.gen_tech,
                                               range(self.ys, yl)])

        sy_index = pd.MultiIndex.from_product([self.st_tech,
                                               range(self.ys, yl)])

        gyd_index = pd.MultiIndex.from_product([self.gen_tech,
                                                range(self.ys, yl),
                                                range(self.days)],
                                               names=['Generation Technology',
                                                      'Year',
                                                      'Day']
                                               )

        syd_index = pd.MultiIndex.from_product([self.st_tech,
                                                range(self.ys, yl),
                                                range(self.days)],
                                               names=['Generation Technology',
                                                      'Year',
                                                      'Day']
                                               )

        yd_index = pd.MultiIndex.from_product([range(self.ys, yl),
                                               range(self.days)],
                                              names=['Year', 'Day']
                                              )

        y_columns = ['Year ' + str(y) for y in range(self.ys, yl)]
        a_columns = ['Year ' + str(a) for a in range(self.years)]
        hour_columns = ['Hour ' + str(h) for h in range(self.hours)]

        # Capacity
        instCap_data = np.zeros((self.gen, yl - self.ys))
        addCap_data = np.copy(instCap_data)

        # Installed Capacity
        for g in range(self.gen):
            for y in range(self.ys, yl):
                instCap_data[g, y - self.ys] = self.InstCap[g, y].X

        instCap_df = pd.DataFrame(
            instCap_data,
            index=self.gen_tech,
            columns=y_columns
        )

        # Added Capacity
        for g in range(self.gen):
            for y in range(self.ys, yl):
                addCap_data[g, y - self.ys] = self.AddCap[g, y].X

        addCap_df = pd.DataFrame(
            addCap_data,
            index=self.gen_tech,
            columns=y_columns
        )

        retCap_data = np.zeros((self.gen, yl - self.ys, self.years))
        remCap_data = np.copy(retCap_data)

        # Retired Capacity
        for g in range(self.gen):
            for y in range(self.ys, yl):
                for a in range(self.years):
                    retCap_data[g, y - self.ys, a] = self.RetCap[g, y, a].X

        retCap_df = pd.DataFrame(
            retCap_data.reshape(-1, self.years),
            index=gy_index,
            columns=a_columns
        )

        # Remaining Capacity
        for g in range(self.gen):
            for y in range(self.ys, yl):
                for a in range(self.years):
                    remCap_data[g, y - self.ys, a] = self.RemCap[g, y, a].X

        remCap_df = pd.DataFrame(
            remCap_data.reshape(-1, self.years),
            index=gy_index,
            columns=a_columns
        )

        output['Installed Capacity'] = instCap_df
        output['Added Capacity'] = addCap_df
        output['Retired Capacity'] = retCap_df
        output['Remaining Capacity'] = remCap_df

        # Storage Power
        instStP_data = np.zeros((self.strg, yl - self.ys))
        addStP_data = np.copy(instStP_data)

        # Installed Storage Power
        for s in range(self.strg):
            for y in range(self.ys, yl):
                instStP_data[s, y - self.ys] = self.InstStP[s, y].X

        instStP_df = pd.DataFrame(
            instStP_data,
            index=self.st_tech,
            columns=y_columns
        )

        # Added Storage Power
        for s in range(self.strg):
            for y in range(self.ys, yl):
                addStP_data[s, y - self.ys] = self.AddStP[s, y].X

        addStP_df = pd.DataFrame(
            addStP_data,
            index=self.st_tech,
            columns=y_columns
        )

        retStP_data = np.zeros((self.strg, yl - self.ys, self.years))
        remStP_data = np.copy(retStP_data)

        # Retired Storage Power
        for s in range(self.strg):
            for y in range(self.ys, yl):
                for a in range(self.years):
                    retStP_data[s, y - self.ys, a] = self.RetStP[s, y, a].X

        retStP_df = pd.DataFrame(
            retStP_data.reshape(-1, self.years),
            index=sy_index,
            columns=a_columns
        )

        # Remaining Storage Power
        for s in range(self.strg):
            for y in range(self.ys, yl):
                for a in range(self.years):
                    remStP_data[s, y - self.ys, a] = self.RemStP[s, y, a].X

        remStP_df = pd.DataFrame(
            remStP_data.reshape(-1, self.years),
            index=sy_index,
            columns=a_columns
        )

        output['Installed Storage Power'] = instStP_df
        output['Added Storage Power'] = addStP_df
        output['Retired Storage Power'] = retStP_df
        output['Remaining Storage Power'] = remStP_df

        # Storage Energy
        # Installed Storage Energy
        instStE_data = np.zeros((self.strg, yl - self.ys))
        addStE_data = np.copy(instStE_data)

        # Installed Storage Energy
        for s in range(self.strg):
            for y in range(self.ys, yl):
                instStE_data[s, y - self.ys] = self.InstStE[s, y].X

        instStE_df = pd.DataFrame(
            instStE_data,
            index=self.st_tech,
            columns=y_columns
        )

        # Added Storage Energy
        for s in range(self.strg):
            for y in range(self.ys, yl):
                addStE_data[s, y - self.ys] = self.AddStE[s, y].X

        addStE_df = pd.DataFrame(
            addStE_data,
            index=self.st_tech,
            columns=y_columns
        )

        retStE_data = np.zeros((self.strg, yl - self.ys, self.years))
        remStE_data = np.copy(retStE_data)

        # Retired Storage Energy
        for s in range(self.strg):
            for y in range(self.ys, yl):
                for a in range(self.years):
                    retStE_data[s, y - self.ys, a] = self.RetStE[s, y, a].X

        retStE_df = pd.DataFrame(
            retStE_data.reshape(-1, self.years),
            index=sy_index,
            columns=a_columns
        )

        # Remaining Storage Energy
        for s in range(self.strg):
            for y in range(self.ys, yl):
                for a in range(self.years):
                    remStE_data[s, y - self.ys, a] = self.RemStE[s, y, a].X

        remStE_df = pd.DataFrame(
            remStE_data.reshape(-1, self.years),
            index=sy_index,
            columns=a_columns
        )

        output['Installed Storage Energy'] = instStE_df
        output['Added Storage Energy'] = addStE_df
        output['Retired Storage Energy'] = retStE_df
        output['Remaining Storage Energy'] = remStE_df

        # Operations
        # Dispatching
        DispElec_data = np.zeros((self.gen, yl - self.ys,
                                  self.days, self.hours))

        for g in range(self.gen):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        DispElec_data[g, y - self.ys,
                                      d, h] = self.DispElec[g, y, d, h].X

        DispElec_df = pd.DataFrame(
            DispElec_data.reshape(-1, self.hours),
            index=gyd_index,
            columns=hour_columns
        )

        # Charging
        Charging_data = np.zeros((self.strg, yl - self.ys,
                                  self.days, self.hours))

        for s in range(self.strg):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        Charging_data[s, y - self.ys,
                                      d, h] = self.Charging[s, y, d, h].X

        Charging_df = pd.DataFrame(
            Charging_data.reshape(-1, self.hours),
            index=syd_index,
            columns=hour_columns
        )

        # Discharging
        Discharging_data = np.zeros((self.strg, yl - self.ys,
                                     self.days, self.hours))

        for s in range(self.strg):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        Discharging_data[s, y - self.ys,
                                         d, h] = self.Discharging[s, y, d, h].X

        Discharging_df = pd.DataFrame(
            Discharging_data.reshape(-1, self.hours),
            index=syd_index,
            columns=hour_columns
        )

        # State of Charge
        SoC_data = np.zeros((self.strg, yl - self.ys,
                             self.days, self.hours))

        for s in range(self.strg):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        SoC_data[s, y - self.ys, d, h] = self.SoC[s, y,
                                                                  d, h].X

        SoC_df = pd.DataFrame(
            SoC_data.reshape(-1, self.hours),
            index=syd_index,
            columns=hour_columns
        )

        # Unserved Demand
        USDem_data = np.zeros((yl - self.ys, self.days, self.hours))

        for y in range(self.ys, yl):
            for d in range(self.days):
                for h in range(self.hours):
                    USDem_data[y - self.ys, d, h] = self.USDem[y, d, h].X

        USDem_df = pd.DataFrame(
            USDem_data.reshape(-1, self.hours),
            index=yd_index,
            columns=hour_columns
        )

        # Excess Electricity
        ExElec_data = np.zeros((yl - self.ys, self.days, self.hours))

        for y in range(self.ys, yl):
            for d in range(self.days):
                for h in range(self.hours):
                    ExElec_data[y - self.ys, d, h] = self.ExElec[y, d, h].X

        ExElec_df = pd.DataFrame(
            ExElec_data.reshape(-1, self.hours),
            index=yd_index,
            columns=hour_columns
        )

        # Sold Electricity
        SoldGen_data = np.zeros((self.gen, yl - self.ys,
                                 self.days, self.hours))
        for g in range(self.gen):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        SoldGen_data[g, y - self.ys,
                                     d, h] = self.SoldGen[g, y, d, h].X

        SoldGen_df = pd.DataFrame(
            SoldGen_data.reshape(-1, self.hours),
            index=gyd_index,
            columns=hour_columns
        )

        SoldStorage_data = np.zeros((self.strg, yl - self.ys,
                                     self.days, self.hours))
        for s in range(self.strg):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        SoldStorage_data[s, y - self.ys,
                                         d, h] = self.SoldStorage[s, y, d, h].X

        SoldStorage_df = pd.DataFrame(
            SoldStorage_data.reshape(-1, self.hours),
            index=syd_index,
            columns=hour_columns
        )

        # Purchased Electricity
        PurElec_data = np.zeros((yl - self.ys, self.days, self.hours))

        for y in range(self.ys, yl):
            for d in range(self.days):
                for h in range(self.hours):
                    PurElec_data[y - self.ys, d, h] = self.PurElec[y, d, h].X

        PurElec_df = pd.DataFrame(
            PurElec_data.reshape(-1, self.hours),
            index=yd_index,
            columns=hour_columns
        )

        # Unsatisfied Reserve
        USRes_data = np.zeros((yl - self.ys))
        for y in range(self.ys, yl):
            USRes_data[y - self.ys] = self.USRes[y].X

        USRes_df = pd.DataFrame(USRes_data.reshape(-1, yl - self.ys),
                                index=['Value'],
                                columns=y_columns[:yl]
                                )

        output['Dispatched Electricity'] = DispElec_df
        output['Charging'] = Charging_df
        output['Discharging'] = Discharging_df
        output['State of Charge'] = SoC_df
        output['Unserved Demand'] = USDem_df
        output['Excess Electricity'] = ExElec_df
        output['Sold (Generated)'] = SoldGen_df
        output['Sold (Stored)'] = SoldStorage_df
        output['Purchased Electricity'] = PurElec_df
        output['Unsatisfied Reserve'] = USRes_df

        # Cash Flows
        output['Yearly NPV'], output['Total NPV'] = self.det_get_npv(yl)

        # Yearly Values
        Y_Dem_values = []
        Y_USDem_values = []
        Y_ExElec_values = []
        Y_PurElec_values = []
        Y_Charging_values = []
        Y_Discharging_values = []
        Y_SoldGen_values = []
        Y_SoldStorage_values = []
        Y_DispElec_Diesel_values = []
        Y_DispElec_PV_values = []
        Y_DispElec_Wind_values = []

        for y in range(self.ys, yl):
            Y_Dem_values.append(
                gp.quicksum(
                    self.demand_forecast.iloc[0, y]
                    * self.hourly_demand.iloc[d, h]
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_USDem_values.append(
                gp.quicksum(
                    self.USDem[y, d, h]
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_ExElec_values.append(
                gp.quicksum(
                    self.ExElec[y, d, h]
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_PurElec_values.append(
                gp.quicksum(
                    self.PurElec[y, d, h]
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_Charging_values.append(
                gp.quicksum(
                    self.Charging[s, y, d, h]
                    for s in range(self.strg)
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_Discharging_values.append(
                gp.quicksum(
                    self.Discharging[s, y, d, h]
                    for s in range(self.strg)
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_SoldGen_values.append(
                gp.quicksum(
                    self.SoldGen[g, y, d, h]
                    for g in range(self.gen)
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_SoldStorage_values.append(
                gp.quicksum(
                    self.SoldStorage[s, y, d, h]
                    for s in range(self.strg)
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_DispElec_Diesel_values.append(
                gp.quicksum(
                    self.DispElec[g, y, d, h]
                    for g in self.diesel_index
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_DispElec_PV_values.append(
                gp.quicksum(
                    self.DispElec[g, y, d, h]
                    for g in self.pv_index
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

            Y_DispElec_Wind_values.append(
                gp.quicksum(
                    self.DispElec[g, y, d, h]
                    for g in self.wind_index
                    for d in range(self.days)
                    for h in range(self.hours)
                ).getValue()
            )

        yearly_data = {
            'Demand': Y_Dem_values,
            'Unserved Demand': Y_USDem_values,
            'Excess Electricity': Y_ExElec_values,
            'Charging': Y_Charging_values,
            'Discharging': Y_Discharging_values,
            'Sold (Generated)': Y_SoldGen_values,
            'Sold (Stored)': Y_SoldStorage_values,
            'Dispatched Diesel': Y_DispElec_Diesel_values,
            'Dispatched PV': Y_DispElec_PV_values,
            'Dispatched Wind': Y_DispElec_Wind_values,
            'Purchased Electricity': Y_PurElec_values
        }

        yearly_df = pd.DataFrame(yearly_data)
        output['Yearly Values'] = yearly_df

        if self.t_grid != self.years:
            output['Model Name'] = ('T = ' + str(self.t_grid)
                                    + ', P = ' + str(self.purchase_tar)
                                    + ', FiT = ' + str(self.feed_in_tar))
        else:
            output['Model Name'] = 'No Grid'
