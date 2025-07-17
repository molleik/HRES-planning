# HRES-planning

This repository contains data, code, and result files for a research paper entitled: [Planning hybrid renewable energy systems under grid
uncertainties](ADD LINK).

## Table of Contents
1. [Description](#description)
2. [Project Structure](#project-structure)
3. [License](#license)

## Description

The project introduces a rolling-horizon modeling framework for Generation Expansion Planning that considers uncertainties regarding the time of availability of the grid, the electricity price and the feed-in tariff. The framework makes use of a tree of stochastic and deterministic linear programs. 
The framework is applied on a community in the Higher Matn, Lebanon, as a case study. Sensitivity analysis is conducted on valuation of assets and installation limits. All output files are provided in their respective folder (see [Project Structure](#project-structure)).

To cite this work: (ADD CITATION)

## Project Structure
```markdown
│   ├── rolling_horizon.py                 # Rolling Horizon model formulation
│   ├── merge_functions.py                 # Helper functions for rolling_horizon.py
│   ├── deterministic_model.py             # Deterministic model formulation
│   ├── stochastic_model.py                # Stochastic model formulation
│   ├── decoupled_model.py                 # Decoupled model formulation
│   └── naive_deterministic.py             # Naive deterministic model formulation
├── Plotting/                              # Codes used to generate figures from output files
│   ├── plotting_functions.py              # Code containing various functions used to visualize different metrics
│   └── plot_rh_results.py                 # Code that automatically plots all relevant results of a rolling horizon run                           
├── Results/                               # Results and output files
│   ├── Base Case/                         # Input file and results for the base case
│   ├── 30% Discount/                      # Input file and results for the case of 30% discount on asset valuation
│   ├── 100% Discount/                     # Input file and results for the case of 100% discount on asset valuation
│   └── Increased Capacity Limit/          # Input file and results for a case of increased capacity limit
└── README.md                              # This README file
```
## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
