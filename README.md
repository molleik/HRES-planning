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
├── Clustering/                            # Source codes and results for the generation of representative days
    └── Data/                              # Data files used to generate the representative days       
├── Model/                                 # Model files
│   ├── rolling_horizon.py                 # Main script
│   ├── deterministic_model.py             # Deterministic model formulation
│   ├── stochastic_model.py                # Stochastic model formulation
│   └── input_file_v5.0.xlsx               # Input file for the model
├── Results/                               # Results and output files
│   ├── Base Case/    # Output files for the base case
│   ├── 30% Discount/               # Output files for a 30% discount on asset valuation
│   ├── 100% Discount/               # Output files for a 100% discount on asset valuation
│   └── Increased Capacity Limit/                           # Output files for an increased capacity limit
└── README.md                              # This README file
```
## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
