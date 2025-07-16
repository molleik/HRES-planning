# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:37:52 2025

@author: Kareem Abou Jalad
"""

import os
import math
import numpy as np
import pandas as pd


class StochasticModel:

    def __init__(self, input_data):
        '''
        Loads input parameters of stochastic model.

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
            'Generation Unit Non Fuel Variable Opex']
        self.fuel_price = data['Fuel Prices']
        self.stp_unit_capex = data['Storage Power Unit Capex']
        self.ste_unit_capex = data['Storage Energy Unit Capex']
        self.stp_unit_fixed_opex = storage['Storage Power Unit Fixed Opex']
        self.ste_unit_fixed_opex = storage['Storage Energy Unit Fixed Opex']
        self.st_unit_var_opex = storage['Storage Unit Non Fuel Variable Opex']
        self.gen_rem_value = data['Rm Val']
        self.st_rem_value = data['Rm ValS']
        self.discount_g = generation['Discount']
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

    def input_scenarios(self, scenarios, probabilities, ys=0,
                        day_night_fit=False, day_threshold=0,
                        fit_factor=1):
        '''
        Loads scenarios and respective probabilities to model.

        Parameters
        ----------
        scenarios : list
            All possible scenarios for the model to consider.
        probabilities : list
            Probabilities for each scenario. Should have the same order as the
            scenarios list.
        ys : int, optional
            Starting year for the model. The default is 0.
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
        self.Scenarios = scenarios
        self.num_scenarios = len(scenarios)
        self.P = probabilities

        self.hourly_fits = []

        for c in range(self.num_scenarios):
            if day_night_fit:
                hourly_fit = self.pv_cap_fac.mask(self.pv_cap_fac
                                                  < day_threshold,
                                                  self.Scenarios[c][1]
                                                  * fit_factor
                                                  )

                hourly_fit = hourly_fit.mask(
                    hourly_fit
                    != self.Scenarios[c][1]*fit_factor,
                    self.Scenarios[c][1]
                )

            else:
                hourly_fit = pd.DataFrame(self.Scenarios[c][1],
                                          index=range(self.days),
                                          columns=range(self.hours)
                                          )

            self.hourly_fits.append(hourly_fit)

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
            self.input_cap = pd.concat([
                self.input_cap,
                input_rem_cap.loc[
                    [(self.gen_tech[g],
                      self.ys - 1)], :
                ]
            ])

        for s in range(self.strg):
            self.input_stp = pd.concat([
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

    def stoch_solve(self):
        '''
        Solves model.

        Returns
        -------
        None.

        '''

        import gurobipy as gp
        from gurobipy import quicksum
        from gurobipy import GRB

        THOUSAND = 1000
        MILLION = 1000000

        self.m = gp.Model('Installation')
        self.m.Params.DualReductions = 0

        # Variables

        # Installation
        self.InstCap = self.m.addVars([(g, y)
                                       for g in range(self.gen)
                                       for y in range(self.ys, self.years)],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Installed Capacity')

        self.AddCap = self.m.addVars([(g, y)
                                      for g in range(self.gen)
                                      for y in range(self.ys, self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Added Capacity')

        self.RemCap = self.m.addVars([(g, y, a)
                                      for g in range(self.gen)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Remaining Capacity')

        self.RetCap = self.m.addVars([(g, y, a)
                                      for g in range(self.gen)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Retired Capacity')

        self.InstStP = self.m.addVars([(s, y)
                                       for s in range(self.strg)
                                       for y in range(self.ys, self.years)],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Installed Storage Power')

        self.AddStP = self.m.addVars([(s, y)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Added Storage Power')

        self.RemStP = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Remaining Storage Power')

        self.RetStP = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Retired Storage Power')

        self.InstStE = self.m.addVars([(s, y)
                                       for s in range(self.strg)
                                       for y in range(self.ys, self.years)],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Installed Storage Energy')

        self.AddStE = self.m.addVars([(s, y)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Added Storage Energy')

        self.RemStE = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Remaining Storage Energy')

        self.RetStE = self.m.addVars([(s, y, a)
                                      for s in range(self.strg)
                                      for y in range(self.ys, self.years)
                                      for a in range(self.years)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Retired Storage Energy')

        self.DispElec = self.m.addVars([(c, g, y, d, h)
                                        for c in range(self.num_scenarios)
                                        for g in range(self.gen)
                                        for y in range(self.ys, self.years)
                                        for d in range(self.days)
                                        for h in range(self.hours)],
                                       vtype=gp.GRB.CONTINUOUS, lb=0,
                                       name='Dispatched Electricity')

        self.USDem = self.m.addVars([(c, y, d, h)
                                     for c in range(self.num_scenarios)
                                     for y in range(self.ys, self.years)
                                     for d in range(self.days)
                                     for h in range(self.hours)],
                                    vtype=gp.GRB.CONTINUOUS, lb=0,
                                    name='Unserved Demand')

        self.Charging = self.m.addVars([(c, s, y, d, h)
                                        for c in range(self.num_scenarios)
                                        for s in range(self.strg)
                                        for y in range(self.ys, self.years)
                                        for d in range(self.days)
                                        for h in range(self.hours)],
                                       vtype=gp.GRB.CONTINUOUS, lb=0,
                                       name='Charging')

        self.SoldGen = self.m.addVars([(c, g, y, d, h)
                                       for c in range(self.num_scenarios)
                                       for g in range(self.gen)
                                       for y in range(self.ys, self.years)
                                       for d in range(self.days)
                                       for h in range(self.hours)],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Sold (Generated)')

        self.SoldStorage = self.m.addVars([(c, s, y, d, h)
                                           for c in range(self.num_scenarios)
                                           for s in range(self.strg)
                                           for y in range(self.ys, self.years)
                                           for d in range(self.days)
                                           for h in range(self.hours)
                                           ],
                                          vtype=gp.GRB.CONTINUOUS, lb=0,
                                          name='Sold (Stored)')

        self.Discharging = self.m.addVars([(c, s, y, d, h)
                                           for c in range(self.num_scenarios)
                                           for s in range(self.strg)
                                           for y in range(self.ys, self.years)
                                           for d in range(self.days)
                                           for h in range(self.hours)],
                                          vtype=gp.GRB.CONTINUOUS, lb=0,
                                          name='Discharging')

        self.ExElec = self.m.addVars([(c, y, d, h)
                                      for c in range(self.num_scenarios)
                                      for y in range(self.ys, self.years)
                                      for d in range(self.days)
                                      for h in range(self.hours)],
                                     vtype=gp.GRB.CONTINUOUS, lb=0,
                                     name='Excess Electricity')

        self.PurElec = self.m.addVars([(c, y, d, h)
                                       for c in range(self.num_scenarios)
                                       for y in range(self.ys, self.years)
                                       for d in range(self.days)
                                       for h in range(self.hours)],
                                      vtype=gp.GRB.CONTINUOUS, lb=0,
                                      name='Purchased Electricity')

        self.SoC = self.m.addVars([(c, s, y, d, h)
                                   for c in range(self.num_scenarios)
                                   for s in range(self.strg)
                                   for y in range(self.ys, self.years)
                                   for d in range(self.days)
                                   for h in range(self.hours)],
                                  vtype=gp.GRB.CONTINUOUS, lb=0,
                                  name='State of Charge')

        self.USRes = self.m.addVars([(c, y)
                                    for y in range(self.years)
                                    for c in range(self.num_scenarios)],
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
                    + self.ste_unit_capex.iat[s, y + self.ys]
                    * self.AddStE[s, y + self.ys]
                    for s in range(self.strg)
                )
            )

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
                    + self.ste_unit_fixed_opex[s]
                    * self.InstStE[s, y + self.ys]
                    for s in range(self.strg)
                )
            )

        # Variable Opex
        self.yearly_var_opex_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_var_opex_c[c][y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * quicksum(
                        self.gen_unit_var_opex[g]
                        * quicksum(
                            self.wd[d]
                            * quicksum(
                                self.DispElec[c, g, y + self.ys, d, h]
                                for h in range(self.hours)
                            )
                            for d in range(self.days)
                        )
                        for g in range(self.gen)
                    )

                    + quicksum(
                        self.st_unit_var_opex[s]
                        * quicksum(
                            self.wd[d]
                            * quicksum(
                                self.Discharging[c, s, y + self.ys, d, h]
                                for h in range(self.hours)
                            )
                            for d in range(self.days)
                        )
                        for s in range(self.strg)
                    )
                )

        # Fuel opex
        self.yearly_fuel_opex_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_fuel_opex_c[c][y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * quicksum(
                        self.heat_rate[g]
                        * self.fuel_price.iloc[0, y + self.ys]
                        * quicksum(
                            self.wd[d]
                            * quicksum(
                                self.DispElec[c, g, y + self.ys, d, h]
                                for h in range(self.hours)
                            )
                            for d in range(self.days)
                        )
                        for g in self.diesel_index)
                )/MILLION

        # Emissions Tax
        self.yearly_em_tax_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_em_tax_c[c][y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * quicksum(
                        self.heat_rate[g]
                        * self.PEF[g]
                        * self.emu_tax
                        * quicksum(
                            self.wd[d]
                            * quicksum(
                                self.DispElec[c, g, y + self.ys, d, h]
                                for h in range(self.hours)
                            )
                            for d in range(self.days)
                        )
                        for g in range(self.gen)
                    )
                )/THOUSAND/MILLION

        # Lost Load
        self.yearly_llc_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_llc_c[c][y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * self.VOLL
                    * quicksum(
                        self.wd[d]
                        * quicksum(
                            self.USDem[c, y + self.ys, d, h]
                            for h in range(self.hours)
                        )
                        for d in range(self.days)
                    )
                )

        # Excess load
        self.yearly_elc_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_elc_c[c][y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * self.VOEL
                    * quicksum(
                        self.wd[d]
                        * quicksum(
                            self.ExElec[c, y + self.ys, d, h]
                            for h in range(self.hours)
                        )
                        for d in range(self.days)
                    )
                )/MILLION

        # Purchased Electricity
        self.yearly_pec_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_pec_c[c][y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * self.Scenarios[c][2]
                    * quicksum(
                        self.wd[d]
                        * quicksum(
                            self.PurElec[c, y + self.ys, d, h]
                            for h in range(self.hours)
                        )
                        for d in range(self.days)
                    )
                )/THOUSAND

        # Electricity Revenues
        self.yearly_elr_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):

                self.yearly_elr_c[c][y] = (
                    (1/(1+self.i)**(y + self.ys))
                    * (
                        quicksum(
                            quicksum(
                                self.wd[d]
                                * quicksum(
                                    self.SoldGen[c, g, y + self.ys, d, h]
                                    * self.hourly_fits[c].iat[d, h]
                                    for h in range(self.hours)
                                )
                                for d in range(self.days)
                            )
                            for g in range(self.gen)
                        )
                        +
                        quicksum(
                            quicksum(
                                self.wd[d]
                                * quicksum(
                                    self.SoldStorage[c, s, y + self.ys, d, h]
                                    * self.hourly_fits[c].iat[d, h]
                                    for h in range(self.hours)
                                )
                                for d in range(self.days)
                            )
                            for s in range(self.strg)
                        )
                    )
                ) / THOUSAND

        # Retirement Revenues
        self.yearly_rtr = [0 for y in range(self.ys, self.years)]
        self.newEarly = [0 for y in range(self.ys, self.years)]
        if self.early_retirement == 1:
            for y in range(self.years - self.ys):
                self.yearly_rtr[y] = (
                    (1/(1 + self.i)**(y + self.ys))
                    * (quicksum(
                        quicksum(
                            self.RetCap[g, y + self.ys, a]
                            * self.gen_rem_value.iloc[g, y + self.ys - a]
                            * self.gen_unit_capex.iat[g, y + self.ys]
                            * (1 - self.discount_g.iloc[g])
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
                        * (1 - self.discount_g.iloc[g])
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

        # Unsatisfied Reserve
        self.yearly_urc_c = [
            [0 for y in range(self.ys, self.years)]
            for c in range(self.num_scenarios)
        ]

        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.yearly_urc_c[c][y] = ((1/(1 + self.i)**(y + self.ys))
                                           * self.USRes[c, y + self.ys]
                                           * self.VOUR
                                           )

        # NPV
        self.NPVCY = [[0 for y in range(self.ys, self.years)]
                      for c in range(self.num_scenarios)]
        for c in range(self.num_scenarios):
            for y in range(self.years - self.ys):
                self.NPVCY[c][y] = (self.yearly_capex[y]
                                    + self.yearly_fixed_opex[y]
                                    + self.yearly_var_opex_c[c][y]
                                    + self.yearly_fuel_opex_c[c][y]
                                    + self.yearly_em_tax_c[c][y]
                                    + self.yearly_llc_c[c][y]
                                    + self.yearly_elc_c[c][y]
                                    + self.yearly_pec_c[c][y]
                                    + self.yearly_urc_c[c][y]
                                    - (self.yearly_elr_c[c][y]
                                       + self.yearly_sv[y]
                                       + self.yearly_rtr[y])
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

        # Demand
        self.m.addConstrs((
            (self.demand_forecast.iloc[0, y]
             * self.hourly_demand.iloc[d, h])
            - self.USDem[c, y, d, h]
            + quicksum(self.Charging[c, s, y, d, h]
                       for s in range(self.strg))
            + quicksum(self.SoldGen[c, g, y, d, h]
                       for g in range(self.gen))
            + quicksum(self.SoldStorage[c, s, y, d, h]
                       for s in range(self.strg))
            == quicksum(self.DispElec[c, g, y, d, h]
                        for g in range(self.gen))
            + quicksum(self.Discharging[c, s, y, d, h]
                       for s in range(self.strg))
            - self.ExElec[c, y, d, h]
            + self.PurElec[c, y, d, h]
            for c in range(self.num_scenarios)
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)),
            name='Energy Balance Equation Constraint')

        # Planning Reserve
        self.m.addConstrs((quicksum(self.InstCap[g, y]*self.gen_prm_factor[g]
                                    for g in range(self.gen))
                           + quicksum(self.InstStP[s, y]*self.stp_prm_factor[s]
                                      for s in range(self.strg))
                           + self.USRes[c, y]
                           >= (1 + self.planningReserve)
                           * self.peak_demand.iloc[0, y]
                           for y in range(self.ys, self.Scenarios[c][0])
                           for c in range(self.num_scenarios)
                           ),
                          name='Planning Reserve Constraint pre-grid')

        self.m.addConstrs((quicksum(self.InstCap[g, y]*self.gen_prm_factor[g]
                                    for g in range(self.gen))
                           + quicksum(self.InstStP[s, y]*self.stp_prm_factor[s]
                                      for s in range(self.strg))
                           + self.USRes[c, y]
                           + self.purchase_limit
                           >= (1 + self.planningReserve)
                           * self.peak_demand.iloc[0, y]
                           for y in range(self.Scenarios[c][0], self.years)
                           for c in range(self.num_scenarios)
                           ),
                          name='Planning Reserve Constraint post-grid')

        # Maximum Usage
        self.m.addConstrs((quicksum(self.wd[d]
                                    * quicksum(self.DispElec[c, g, y, d, h]
                                               for h in range(self.hours))
                                    for d in range(self.days))
                           <= self.max_usage.loc[g]*self.InstCap[g, y]
                           * 365*24
                           for g in range(self.gen)
                           for y in range(self.ys, self.years)
                           for c in range(self.num_scenarios)
                           ),
                          name='Maximum Usage Constraint')

        # Dispacthing
        self.m.addConstrs(
            (self.DispElec[c, g, y, d, h]
             <= self.InstCap[g, y]
             for c in range(self.num_scenarios)
             for g in self.diesel_index
             for y in range(self.ys, self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Dispatched Diesel Capacity')

        self.m.addConstrs(
            (self.DispElec[c, g, y, d, h]
             <= self.InstCap[g, y] * self.pv_cap_fac.iloc[d, h]
             for c in range(self.num_scenarios)
             for g in self.pv_index
             for y in range(self.ys, self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Dispatched PV Capacity')

        self.m.addConstrs(
            (self.DispElec[c, g, y, d, h]
             <= self.InstCap[g, y] * self.wind_cap_fac.iloc[d, h]
             for c in range(self.num_scenarios)
             for g in self.wind_index
             for y in range(self.ys, self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Dispatched Wind Capacity')

        # Max Charging
        self.m.addConstrs(
            (self.Charging[c, s, y, d, h] <= self.InstStP[s, y]
             for c in range(self.num_scenarios)
             for s in range(self.strg)
             for y in range(self.ys, self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Max Charging')

        # Max Discharging
        self.m.addConstrs(
            (self.Discharging[c, s, y, d, h]
             <= self.InstStP[s, y]
             for c in range(self.num_scenarios)
             for s in range(self.strg)
             for y in range(self.ys, self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Max Discharging')

        # Max Storage
        self.m.addConstrs((self.SoC[c, s, y, d, h] <=
                           self.InstStE[s, y]*self.max_storage[s]
                           for c in range(self.num_scenarios)
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Max Storage')

        # Min Storage
        self.m.addConstrs((self.SoC[c, s, y, d, h] >=
                           self.InstStE[s, y]*self.min_storage[s]
                           for c in range(self.num_scenarios)
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Min Storage')

        # State of Charge
        self.m.addConstrs((self.SoC[c, s, y, d, h]
                           == self.SoC[c, s, y, d, h-1]
                           + ((self.Charging[c, s, y, d, h-1]
                               * self.char_eff[s])
                           - (self.Discharging[c, s, y, d, h-1]
                              / self.dischar_eff[s]))
                           for c in range(self.num_scenarios)
                           for s in range(self.strg)
                           for y in range(self.ys, self.years)
                           for d in range(self.days)
                           for h in range(1, self.hours)),
                          name='State of charge')

        self.m.addConstrs(
            (self.SoC[c, s, y, d, 0]
             == self.SoC[c, s, y, d, 23]
             + (self.Charging[c, s, y, d, 23]
                * self.char_eff[s])
             - (self.Discharging[c, s, y, d, 23]
                / self.dischar_eff[s])
             for c in range(self.num_scenarios)
             for s in range(self.strg)
             for y in range(self.ys, self.years)
             for d in range(self.days)),
            name='State of charge initial')

        # Purchased Electricity Constraint
        self.m.addConstrs(
            (self.PurElec[c, y, d, h] == 0
             for c in range(self.num_scenarios)
             for y in range(self.ys, self.Scenarios[c][0])
             for d in range(self.days)
             for h in range(self.hours)),
            name='Purchased Electricity')

        # Purchased Electricity Amount Constraint
        self.m.addConstrs(
            (self.PurElec[c, y, d, h]
             <= self.purchase_limit
             for c in range(self.num_scenarios)
             for y in range(max(self.Scenarios[c][0], self.ys), self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Purchased Electricity Amount')

        # Sold Electricity Constraints
        self.m.addConstrs(
            (self.SoldGen[c, g, y, d, h] == 0
             for c in range(self.num_scenarios)
             for g in range(self.gen)
             for y in range(self.ys, self.Scenarios[c][0])
             for d in range(self.days)
             for h in range(self.hours)),
            name='Sold Electricity')

        self.m.addConstrs((self.SoldStorage[c, s, y, d, h] == 0
                           for c in range(self.num_scenarios)
                           for s in range(self.strg)
                           for y in range(self.ys, self.Scenarios[c][0])
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Sold Electricity')

        self.m.addConstrs((
            self.SoldGen[c, g, y, d, h] <= self.DispElec[c, g, y, d, h]
            for g in range(self.gen)
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)
            for c in range(self.num_scenarios)),
            name='Sold Generated Electricity'
        )

        self.m.addConstrs((
            self.SoldStorage[c, s, y, d, h] <= self.Discharging[c, s, y, d, h]
            for s in range(self.strg)
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)
            for c in range(self.num_scenarios)),
            name='Sold Stored Electricity'
        )

        self.m.addConstrs((
            self.USDem[c, y, d, h] <= (self.demand_forecast.iloc[0, y]
                                       * self.hourly_demand.iloc[d, h])
            for y in range(self.ys, self.years)
            for d in range(self.days)
            for h in range(self.hours)
            for c in range(self.num_scenarios)),
            name='Unserved Demand Limit'
        )

        self.m.addConstrs(
            (self.SoldGen[c, g, y, d, h] == 0
             for c in range(self.num_scenarios)
             for g in self.diesel_index
                for y in range(max(self.Scenarios[c][0], self.ys), self.years)
             for d in range(self.days)
             for h in range(self.hours)),
            name='Sold Electricity Diesel Constraint')

        self.m.addConstrs((
            quicksum(self.SoldGen[c, g, y, d, h] for g in range(self.gen))
            + quicksum(self.SoldStorage[c, s, y, d, h]
                       for s in range(self.strg))
            <= self.sold_limit
            for c in range(self.num_scenarios)
            for y in range(max(self.Scenarios[c][0],
                               self.ys),
                           self.years)
            for d in range(self.days)
            for h in range(self.hours)
        ),
            name='Sold Electricity Amount')

        # Objective Function
        self.m.setObjective(
            sum(
                sum(self.NPVCY[c])*self.P[c]
                for c in range(self.num_scenarios)
            ),
            GRB.MINIMIZE
        )

        self.m.optimize()

    def stoch_get_npv(self, yl):
        '''
        Helper function for stoch_store_decisions. Stores yearly and total NPV
        over years ys to yl.

        Parameters
        ----------
        yl : int
            NPV is stored up to year yl.

        Returns
        -------
        yearly_npv_c : dict
            Yearly NPV of each objective function component, per scenario.
        total_npv_df : DataFrame
            Total (expected) NPV of each objective function component.

        '''

        import pandas as pd
        import numpy as np

        yearly_npv_c = {}
        years = ['Year ' + str(y) for y in range(self.ys, yl)]

        def extract_values(expression_list):
            return [expr.getValue() for expr in expression_list]

        # Capex
        total_capex = sum(self.yearly_capex[:yl - self.ys]).getValue()

        yearly_npv_c['Capex'] = pd.DataFrame(
            data=[extract_values(self.yearly_capex[:yl - self.ys])],
            columns=years
        )

        # Fixed Opex
        total_fixed_opex = sum(
            self.yearly_fixed_opex[:yl - self.ys]
        ).getValue()

        yearly_npv_c['Fixed Opex'] = pd.DataFrame(
            data=[extract_values(self.yearly_fixed_opex[:yl - self.ys])],
            columns=years
        )

        # Variable Opex
        total_var_opex = sum(
            sum(self.yearly_var_opex_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Variable Opex'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_var_opex_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Fuel Opex
        total_fuel_opex = sum(
            sum(self.yearly_fuel_opex_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Fuel Opex'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_fuel_opex_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Emissions Tax
        total_em_tax = sum(
            sum(self.yearly_em_tax_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Emissions Tax'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_em_tax_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Lost Load Cost
        total_llc = sum(
            sum(self.yearly_llc_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Lost Load Cost'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_llc_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Excess Load Cost
        total_elc = sum(
            sum(self.yearly_elc_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Excess Load Cost'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_elc_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Purchased Electricity Cost
        total_pec = sum(
            sum(self.yearly_pec_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Purchased Electricity Cost'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_pec_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Electricity Revenues
        total_elr = sum(
            sum(self.yearly_elr_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Electricity Revenues'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_elr_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Unsatisfied Reserve Cost
        total_urc = sum(
            sum(self.yearly_urc_c[c][:yl - self.ys]) * self.P[c]
            for c in range(self.num_scenarios)
        ).getValue()

        yearly_npv_c['Unsatisfied Reserve Cost'] = pd.DataFrame(
            data=[
                extract_values(self.yearly_urc_c[c][:yl - self.ys])
                for c in range(self.num_scenarios)
            ],
            columns=years
        )

        # Retirement Revenues
        if self.early_retirement == 1:
            total_rtr = sum(self.yearly_rtr[:yl - self.ys]).getValue()
            rtr_data = extract_values(self.yearly_rtr[:yl - self.ys])
        else:
            total_ver = 0
            rtr_data = [0] * (yl - self.ys)

        yearly_npv_c['Retirement Revenues'] = pd.DataFrame(
            data=[rtr_data],
            columns=years
        )

        # Salvage Value
        if yl == self.years:
            total_sv = self.yearly_sv[-1].getValue()
        else:
            total_sv = 0

        # NPV
        NPV = (
            total_capex
            + total_fixed_opex
            + total_var_opex
            + total_fuel_opex
            + total_em_tax
            + total_llc
            + total_elc
            + total_pec
            - (
                total_elr
                + total_rtr
                + total_sv
            )
        )

        self.NPVC = [sum(self.NPVCY[c]).getValue()
                     for c in range(self.num_scenarios)]

        total_npv_data = {
            "Total Value": [NPV],
            "Capex": [total_capex],
            "Fixed Opex": [total_fixed_opex],
            "Variable Opex": [total_var_opex],
            "Fuel Opex": [total_fuel_opex],
            "Emissions Tax": [total_em_tax],
            "Lost Load Cost": [total_llc],
            "Excess Load Cost": [total_elc],
            "Purchased Electricity Cost": [total_pec],
            'Unsatisfied Reserve Cost': [total_urc],
            "Electricity Revenues": [total_elr],
            "Retirement Revenues": [total_rtr],
            "Salvage Value": [total_sv]
        }

        total_npv_df = pd.DataFrame(total_npv_data)

        return total_npv_df, yearly_npv_c

    def stoch_store_decisions(self, output, yl=20):
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

        import gurobipy as gp
        import pandas as pd

        gy_index = pd.MultiIndex.from_product([self.gen_tech,
                                               range(self.ys, yl)])

        sy_index = pd.MultiIndex.from_product([self.st_tech,
                                               range(self.ys, yl)])

        cgyd_index = pd.MultiIndex.from_product([range(self.num_scenarios),
                                                 self.gen_tech,
                                                 range(self.ys, yl),
                                                 range(self.days)],
                                                names=['Scenario',
                                                       'Generation Technology',
                                                       'Year',
                                                       'Day']
                                                )

        csyd_index = pd.MultiIndex.from_product([range(self.num_scenarios),
                                                 self.st_tech,
                                                 range(self.ys, yl),
                                                 range(self.days)],
                                                names=['Scenario',
                                                       'Generation Technology',
                                                       'Year',
                                                       'Day']
                                                )

        cyd_index = pd.MultiIndex.from_product([range(self.num_scenarios),
                                                range(self.ys, yl),
                                                range(self.days)],
                                               names=['Scenario',
                                                      'Year',
                                                      'Day']
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
        DispElec_data = np.zeros((self.num_scenarios, self.gen, yl - self.ys,
                                  self.days, self.hours))

        for c in range(self.num_scenarios):
            for g in range(self.gen):
                for y in range(self.ys, yl):
                    for d in range(self.days):
                        for h in range(self.hours):
                            DispElec_data[c, g, y - self.ys, d, h] = self.DispElec[c, g, y,
                                                                                   d, h].X

        DispElec_df = pd.DataFrame(
            DispElec_data.reshape(-1, self.hours),
            index=cgyd_index,
            columns=hour_columns
        )

        # Charging
        Charging_data = np.zeros((self.num_scenarios, self.strg, yl - self.ys,
                                  self.days, self.hours))

        for c in range(self.num_scenarios):
            for s in range(self.strg):
                for y in range(self.ys, yl):
                    for d in range(self.days):
                        for h in range(self.hours):
                            Charging_data[c, s, y - self.ys, d, h] = self.Charging[c, s, y,
                                                                                   d, h].X

        Charging_df = pd.DataFrame(
            Charging_data.reshape(-1, self.hours),
            index=csyd_index,
            columns=hour_columns
        )

        # Discharging
        Discharging_data = np.zeros((self.num_scenarios, self.strg, yl - self.ys,
                                     self.days, self.hours))

        for c in range(self.num_scenarios):
            for s in range(self.strg):
                for y in range(self.ys, yl):
                    for d in range(self.days):
                        for h in range(self.hours):
                            Discharging_data[c, s, y - self.ys, d, h] = self.Discharging[c, s, y,
                                                                                         d, h].X

        Discharging_df = pd.DataFrame(
            Discharging_data.reshape(-1, self.hours),
            index=csyd_index,
            columns=hour_columns
        )

        # State of Charge
        SoC_data = np.zeros((self.num_scenarios, self.strg, yl - self.ys,
                             self.days, self.hours))

        for c in range(self.num_scenarios):
            for s in range(self.strg):
                for y in range(self.ys, yl):
                    for d in range(self.days):
                        for h in range(self.hours):
                            SoC_data[c, s, y - self.ys, d, h] = self.SoC[c, s, y,
                                                                         d, h].X

        SoC_df = pd.DataFrame(
            SoC_data.reshape(-1, self.hours),
            index=csyd_index,
            columns=hour_columns
        )

        # Unserved Demand
        USDem_data = np.zeros(
            (self.num_scenarios, yl - self.ys, self.days, self.hours))

        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        USDem_data[c, y - self.ys, d,
                                   h] = self.USDem[c, y, d, h].X

        USDem_df = pd.DataFrame(
            USDem_data.reshape(-1, self.hours),
            index=cyd_index,
            columns=hour_columns
        )

        # Excess Electricity
        ExElec_data = np.zeros(
            (self.num_scenarios, yl - self.ys, self.days, self.hours))

        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        ExElec_data[c, y - self.ys, d,
                                    h] = self.ExElec[c, y, d, h].X

        ExElec_df = pd.DataFrame(
            ExElec_data.reshape(-1, self.hours),
            index=cyd_index,
            columns=hour_columns
        )

        # Sold Electricity
        SoldGen_data = np.zeros((self.num_scenarios, self.gen, yl - self.ys,
                                 self.days, self.hours))

        for c in range(self.num_scenarios):
            for g in range(self.gen):
                for y in range(self.ys, yl):
                    for d in range(self.days):
                        for h in range(self.hours):
                            SoldGen_data[c, g, y - self.ys, d, h] = self.SoldGen[c, g, y,
                                                                                 d, h].X

        SoldGen_df = pd.DataFrame(
            SoldGen_data.reshape(-1, self.hours),
            index=cgyd_index,
            columns=hour_columns
        )

        SoldStorage_data = np.zeros((self.num_scenarios, self.strg, yl - self.ys,
                                     self.days, self.hours))

        for c in range(self.num_scenarios):
            for s in range(self.strg):
                for y in range(self.ys, yl):
                    for d in range(self.days):
                        for h in range(self.hours):
                            SoldStorage_data[c, s, y - self.ys, d, h] = self.SoldStorage[c, s, y,
                                                                                         d, h].X

        SoldStorage_df = pd.DataFrame(
            SoldStorage_data.reshape(-1, self.hours),
            index=csyd_index,
            columns=hour_columns
        )

        # Purchased Electricity
        PurElec_data = np.zeros(
            (self.num_scenarios, yl - self.ys, self.days, self.hours))

        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        PurElec_data[c, y - self.ys, d,
                                     h] = self.PurElec[c, y, d, h].X

        PurElec_df = pd.DataFrame(
            PurElec_data.reshape(-1, self.hours),
            index=cyd_index,
            columns=hour_columns
        )

        # Unsatisfied Reserve
        USRes_data = np.zeros((self.num_scenarios, yl - self.ys))

        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                USRes_data[c, y - self.ys] = self.USRes[c, y].X

        USRes_df = pd.DataFrame(USRes_data.reshape(-1, yl - self.ys),
                                index=range(self.num_scenarios),
                                columns=y_columns[:yl])
        USRes_df.index.name = 'Scenario'

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

        # Yearly values
        YearlyValues = {}
        yearlyNames = ['Unserved Demand', 'Charging',
                       'Sold (Generated)', 'Sold (Stored)',
                       'Dispatched Diesel', 'Dispatched PV',
                       'Dispatched Wind', 'Discharging',
                       'Excess Electricity', 'Purchased Electricity']

        # Demand
        demandY = np.zeros((yl - self.ys))
        for y in range(self.ys, yl):
            for d in range(self.days):
                for h in range(self.hours):
                    demandY[y - self.ys] += (self.demand_forecast.iloc[0, y]
                                             * self.hourly_demand.iloc[d, h])
        demandY_df = pd.DataFrame(demandY, columns=['Demand'])

        # Unserved Demand
        USDemYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                USDemYC[c, y - self.ys] = gp.quicksum(self.USDem[c, y,
                                                                 d, h]
                                                      for d in range(self.days)
                                                      for h in range(self.hours)).getValue()
        USDem_df = pd.DataFrame(USDemYC)
        YearlyValues['Unserved Demand'] = USDem_df

        # Charging
        chargingYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                chargingYC[c, y - self.ys] = gp.quicksum(self.Charging[c, s,
                                                                       y, d, h]
                                                         for s in range(self.strg)
                                                         for d in range(self.days)
                                                         for h in range(self.hours)).getValue()
        charging_df = pd.DataFrame(chargingYC)
        YearlyValues['Charging'] = charging_df

        # Sold (Generated)
        soldGenYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                soldGenYC[c, y - self.ys] = gp.quicksum(self.SoldGen[c, g,
                                                                     y, d, h]
                                                        for g in range(self.gen)
                                                        for d in range(self.days)
                                                        for h in range(self.hours)).getValue()
        soldGen_df = pd.DataFrame(soldGenYC)
        YearlyValues['Sold (Generated)'] = soldGen_df

        # Sold (Stored)
        soldStorageYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                soldStorageYC[c, y - self.ys] = gp.quicksum(self.SoldStorage[c, s,
                                                                             y, d, h]
                                                            for s in range(self.strg)
                                                            for d in range(self.days)
                                                            for h in range(self.hours)).getValue()
        soldStorage_df = pd.DataFrame(soldStorageYC)
        YearlyValues['Sold (Stored)'] = soldStorage_df

        # Diesel
        dispElecDieselYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                dispElecDieselYC[c, y - self.ys] = gp.quicksum(self.DispElec[c, g,
                                                                             y, d, h]
                                                               for g in self.diesel_index
                                                               for d in range(self.days)
                                                               for h in range(self.hours)).getValue()
        dispElecDiesel_df = pd.DataFrame(dispElecDieselYC)
        YearlyValues['Dispatched Diesel'] = dispElecDiesel_df

        # PV
        dispElecPVYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                dispElecPVYC[c, y - self.ys] = gp.quicksum(self.DispElec[c, g,
                                                                         y, d, h]
                                                           for g in self.pv_index
                                                           for d in range(self.days)
                                                           for h in range(self.hours)).getValue()
        dispElecPV_df = pd.DataFrame(dispElecPVYC)
        YearlyValues['Dispatched PV'] = dispElecPV_df

        # Wind
        dispElecWindYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                dispElecWindYC[c, y - self.ys] = gp.quicksum(self.DispElec[c, g,
                                                                           y, d, h]
                                                             for g in self.wind_index
                                                             for d in range(self.days)
                                                             for h in range(self.hours)).getValue()
        dispElecWind_df = pd.DataFrame(dispElecWindYC)
        YearlyValues['Dispatched Wind'] = dispElecWind_df

        # Discharging
        dischargingYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                dischargingYC[c, y - self.ys] = gp.quicksum(self.Discharging[c, s,
                                                                             y, d, h]
                                                            for s in range(self.strg)
                                                            for d in range(self.days)
                                                            for h in range(self.hours)).getValue()
        discharging_df = pd.DataFrame(dischargingYC)
        YearlyValues['Discharging'] = discharging_df

        # Excess Electricity
        ExDemYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                ExDemYC[c, y - self.ys] = gp.quicksum(self.ExElec[c, y,
                                                                  d, h]
                                                      for d in range(self.days)
                                                      for h in range(self.hours)).getValue()
        ExDem_df = pd.DataFrame(ExDemYC)
        YearlyValues['Excess Electricity'] = ExDem_df

        # Purchased Electricity
        PurElecYC = np.zeros((self.num_scenarios, yl - self.ys))
        for c in range(self.num_scenarios):
            for y in range(self.ys, yl):
                PurElecYC[c, y - self.ys] = gp.quicksum(self.PurElec[c, y,
                                                                     d, h]
                                                        for d in range(self.days)
                                                        for h in range(self.hours)).getValue()
        PurElec_df = pd.DataFrame(PurElecYC)
        YearlyValues['Purchased Electricity'] = PurElec_df

        yearlyIndex = pd.MultiIndex.from_product([range(self.num_scenarios), yearlyNames],
                                                 names=['Scenario', 'Value'])
        yearlyColumns = range(self.ys, yl)

        yearly_df = pd.DataFrame()
        for c in range(self.num_scenarios):
            for key in YearlyValues:
                newValue = YearlyValues[key]
                yearly_df = pd.concat([yearly_df, newValue.loc[[c]]])

        yearly_df.index = yearlyIndex
        yearly_df.columns = yearlyColumns

        output['Yearly Demand'] = demandY_df
        output['Yearly Values'] = yearly_df

        # Cash Flows
        output['Total NPV'], yearly_npv_c = self.stoch_get_npv(yl)
        for key in yearly_npv_c:
            output[key] = yearly_npv_c[key]

        output['Model Name'] = 'C=' + \
            str(self.num_scenarios) + '_ys = ' + str(self.ys)
        output['Scenarios'] = self.Scenarios
        output['ys'] = self.ys


# model = StochasticModel(
#     r"C:\Users\User\OneDrive - American University of Beirut\Decentralized Planning Grid Uncertainty\Codes - Kareem\RollingHorizonData.xlsx")

# model.input_scenarios([(1, 0.1, 0.03), (1, 0.1, 0.03)], [0.5, 0.5])

# model.stoch_solve()

# output = model.stoch_store_decisions()


# def print_excel(results, file_path, model_type):
#     import pandas as pd

#     model_name = model_type + '_' + results['Model Name'] + '.xlsx'

#     file_name = os.path.join(
#         file_path,
#         model_name
#     )

#     writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
#     for key in results:
#         if type(results[key]) is pd.core.frame.DataFrame:
#             results[key].to_excel(
#                 writer,
#                 sheet_name=key,
#                 merge_cells=False
#             )

#     writer.close()


# print_excel(output, r"C:\Users\User\OneDrive - American University of Beirut\Decentralized Planning Grid Uncertainty\Codes - Kareem", 'blabla')
