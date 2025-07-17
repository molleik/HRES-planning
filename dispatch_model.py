import gurobipy as gp
from gurobipy import quicksum
from gurobipy import GRB
import pandas as pd
import numpy as np
import math
import time


class DispatchModel:

    def __init__(self, input_data):
        '''
        Loads input parameters of dispatch model.

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
                       day_night_fit=False, day_threshold=0,
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

            self.hourly_fit = self.hourly_fit.mask(self.hourly_fit
                                                   > day_threshold,
                                                   self.feed_in_tar)
        else:
            self.hourly_fit = pd.DataFrame(self.feed_in_tar,
                                           index=range(self.days),
                                           columns=range(self.hours)
                                           )

    def input_installations(self, input_dict):
        '''
        Loads installed capacity and fixed costs from previous output
        as input to the model.

        Parameters
        ----------
        input_dict : dict
            Output from previous model. Contains remaining installations
            DataFrames for capacity and storage.

        Returns
        -------
        None.

        '''

        self.InstCap = input_dict['Installed Capacity'].values
        self.InstStP = input_dict['Installed Storage Power'].values
        self.InstStE = input_dict['Installed Storage Energy'].values
        self.FixedNPV = (
            input_dict['Total NPV']['Capex'].loc[0]
            + input_dict['Total NPV']['Fixed Opex'].loc[0]
            - input_dict['Total NPV']['Retirement Revenues'].loc[0]
            - input_dict['Total NPV']['Salvage Value'].loc[0]
        )

    def disp_solve(self):
        '''
        Solves model.

        Returns
        -------
        None.

        '''

        thousand = 1000
        million = 1000000

        start = time.time()

        self.m = gp.Model('Installation')
        self.m.Params.DualReductions = 0
        self.m.Params.LogToConsole = 0

        # Variables
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
        # Variable Opex
        self.VariableOpexY = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.VariableOpexY[y] = (
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

        self.TotalVariableOpex = sum(self.VariableOpexY)

        # Fuel opex
        self.FuelOpexY = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.FuelOpexY[y] = (
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
            )/million

        self.TotalFuelOpex = sum(self.FuelOpexY)

        # Emissions Tax
        self.EmissionsTaxY = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.EmissionsTaxY[y] = (
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
            )/thousand/million

        self.TotalEmissionsTax = sum(self.EmissionsTaxY)

        # Lost Load
        self.LostLoadCostY = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.LostLoadCostY[y] = (
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

        self.TotalLostLoadCost = sum(self.LostLoadCostY)

        # Excess load
        self.ExcessLoadCostY = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.ExcessLoadCostY[y] = (
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
            )/million

        self.TotalExcessLoadCost = sum(self.ExcessLoadCostY)

        # Purchased Electricity
        self.PurchasedElectricityCostY = [0 for y in range(self.ys,
                                                           self.years)]
        for y in range(self.years - self.ys):
            self.PurchasedElectricityCostY[y] = (
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
            )/thousand

        self.TotalPurchasedElectricityCost = sum(
            self.PurchasedElectricityCostY
        )

        # Revenues
        self.RevenuesY = [0 for y in range(self.ys, self.years)]

        for y in range(self.years - self.ys):
            self.RevenuesY[y] = (
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
            ) / thousand

        self.TotalRevenues = sum(self.RevenuesY)

        # Unsatisfied Reserve
        self.USResCostY = [0 for y in range(self.ys, self.years)]
        for y in range(self.years - self.ys):
            self.USResCostY[y] = (
                (1/((1 + self.i)**(y + self.ys)))
                * self.USRes[y + self.ys]
                * self.VOUR
            )

        self.TotalUSResCost = sum(self.USResCostY)

        # NPV
        self.NPVY = [0 for y in range(self.ys, self.years)]

        for y in range(self.years - self.ys):
            self.NPVY[y] = (
                self.VariableOpexY[y]
                + self.FuelOpexY[y]
                + self.EmissionsTaxY[y]
                + self.LostLoadCostY[y]
                + self.ExcessLoadCostY[y]
                + self.PurchasedElectricityCostY[y]
                + self.USResCostY[y]
                - (
                    self.RevenuesY[y]
                )
            )

        # Constraints
        # Demand
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

        # Selling Constraint
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
                           for y in range(max(self.t_grid, self.ys), self.years
                                          )
                           ),
                          name='Planning Reserve Constraint post-grid')

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
                           + ((self.Charging[s, y, d, h-1]*self.char_eff[s])
                              - (self.Discharging[s, y, d, h-1]
                                 / self.dischar_eff[s]))
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

        # Purchased Electricity Constraint
        self.m.addConstrs((self.PurElec[y, d, h] == 0
                           for y in range(self.ys, self.t_grid)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Purchased Electricity')

        # Purchased Electricity Amount Constraint
        self.m.addConstrs((self.PurElec[y, d, h]
                           <= self.purchase_limit
                           for y in range(max(self.t_grid, self.ys), self.years
                                          )
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Purchased Electricity Amount')

        # Sold Electricity Constraints
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

        self.m.addConstrs((self.SoldGen[g, y, d, h] == 0
                           for g in self.diesel_index
                           for y in range(max(self.t_grid, self.ys),
                                          self.years)
                           for d in range(self.days)
                           for h in range(self.hours)),
                          name='Sold Electricity Diesel Constraint')

        self.m.addConstrs((
            quicksum(self.SoldGen[g, y, d, h] for g in range(self.gen))
            + quicksum(self.SoldStorage[s, y, d, h] for s in range(self.strg))
            <= self.sold_limit
            for y in range(max(self.t_grid, self.ys), self.years)
            for d in range(self.days)
            for h in range(self.hours)),
            name='Sold Electricity Amount')

        # Objective Function
        self.m.setObjective(sum(self.NPVY),
                            GRB.MINIMIZE)

        self.m.optimize()

        end = time.time()

        self.solve_time = end - start

    def disp_get_npv(self, yl):
        '''
        Helper function for disp_store_decisions. Stores yearly and total NPV
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
        yearly_npv_data = {
            "Dispatching Value": [
                self.NPVY[y].getValue() for y in range(yl - self.ys)],
            "Variable Opex": [
                self.VariableOpexY[y].getValue() for y in range(yl - self.ys)],
            "Fuel Opex": [
                self.FuelOpexY[y].getValue() for y in range(yl - self.ys)],
            "Emissions Tax": [
                self.EmissionsTaxY[y].getValue() for y in range(yl - self.ys)],
            "Lost Load Cost": [
                self.LostLoadCostY[y].getValue() for y in range(yl - self.ys)],
            "Excess Load Cost": [
                self.ExcessLoadCostY[y].getValue()
                for y in range(yl - self.ys)],
            "Purchased Electricity Cost": [
                self.PurchasedElectricityCostY[y].getValue()
                for y in range(yl - self.ys)],
            'Unsatisfied Reserve Cost': [
                self.USResCostY[y].getValue()
                for y in range(yl - self.ys)],
            "Electricity Revenues": [
                self.RevenuesY[y].getValue() for y in range(yl - self.ys)],
        }

        yearly_npv_df = pd.DataFrame(yearly_npv_data)

        total_npv_data = {
            "Total Value": [self.FixedNPV
                            + sum(self.NPVY[:yl - self.ys]).getValue()],
            "Variable Opex": [sum(
                self.VariableOpexY[:yl - self.ys]).getValue()],
            "Fuel Opex": [sum(
                self.FuelOpexY[:yl - self.ys]).getValue()],
            "Emissions Tax": [sum(
                self.EmissionsTaxY[:yl - self.ys]).getValue()],
            "Lost Load Cost": [sum(
                self.LostLoadCostY[:yl - self.ys]).getValue()],
            "Excess Load Cost": [sum(
                self.ExcessLoadCostY[:yl - self.ys]).getValue()],
            "Purchased Electricity Cost": [sum(
                self.PurchasedElectricityCostY[:yl - self.ys]).getValue()],
            'Unsatisfied Reserve Cost': [sum(
                self.USResCostY[y].getValue() for y in range(yl - self.ys))],
            "Electricity Revenues": [sum(
                self.RevenuesY[:yl - self.ys]).getValue()],
        }

        total_npv_df = pd.DataFrame(total_npv_data)

        return yearly_npv_df, total_npv_df

    def disp_store_decisions(self, output, yl=20):
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
            'Tgrid': [self.t_grid],
            'FiT': [self.feed_in_tar],
            'EDLTar': [self.purchase_tar]
        }

        scenario_df = pd.DataFrame(scenario_data)

        output['Scenario'] = scenario_df

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
        hour_columns = ['Hour ' + str(h) for h in range(self.hours)]

        # Operations
        # Dispatching
        DispElec_data = np.zeros((self.gen, yl - self.ys,
                                  self.days, self.hours))

        for g in range(self.gen):
            for y in range(self.ys, yl):
                for d in range(self.days):
                    for h in range(self.hours):
                        DispElec_data[g, y - self.ys, d, h] = self.DispElec[g, y,
                                                                            d, h].X

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
                        Charging_data[s, y - self.ys, d, h] = self.Charging[s, y,
                                                                            d, h].X

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
                        Discharging_data[s, y - self.ys, d, h] = self.Discharging[s, y,
                                                                                  d, h].X

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
                        SoldGen_data[g, y - self.ys, d, h] = self.SoldGen[g, y,
                                                                          d, h].X

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
                        SoldStorage_data[s, y - self.ys, d, h] = self.SoldStorage[s, y,
                                                                                  d, h].X

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
                                columns=y_columns)

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
        output['Yearly NPV'], output['Total NPV'] = self.disp_get_npv(yl)

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
                    * self.wd[d]
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
            output['Model Name'] = ('Tgrid = ' + str(self.t_grid)
                                    + '_FiT = ' + str(self.feed_in_tar)
                                    + '_EDLTar = ' + str(self.purchase_tar))
        else:
            output['Model Name'] = 'NoGrid'
