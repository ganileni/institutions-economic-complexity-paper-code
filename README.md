# Institutions in Economic Complexity: Enhancing Growth Predictions and Theoretical Understanding

This repository contains the code and data for reproducing the analysis in the paper:

> Angelini, O., Tacchella, A., Pietronero, L., & Di Matteo, T. *Institutions in Economic Complexity: Enhancing Growth Predictions and Theoretical Understanding.*


## Repository Structure

```
institutions-economic-complexity-paper-code/
├── data/                      # Input datasets
│   ├── COMTRADE_reconciled_hmm/
│   │   └── fitness.csv        # Economic Fitness data
│   ├── WDI/
│   │   └── NY.GDP.PCAP.KD.csv # GDP per capita data
│   ├── POLITY_V/
│   │   └── polity2.csv        # Polity scores
│   └── PATSTAT_tech_fitness/
│       └── tech_fitness_8dig.csv # Technological Fitness data
├── output/                    # Generated outputs (gitignored)
│   ├── predictions/           # Model predictions CSV
│   └── plots/                 # Generated figures
├── src/
│   ├── predictions.py         # Prediction utilities
│   ├── plotting.py            # Plotting utilities
│   ├── run_predictions.py     # Prediction generation script
│   ├── generate_plots.py      # Plot generation script
│   └── generate_stats.py      # Statistical analysis script
├── run_all.py                 # Main entry point
├── Pipfile                    # Pipenv dependencies
├── requirements.txt           # Pip dependencies (alternative)
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- [pipenv](https://pipenv.pypa.io/en/latest/) (recommended) or pip

### Setup with Pipenv (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/ganileni/institutions-economic-complexity-paper-code.git
cd institutions-economic-complexity-paper-code
```

2. Install dependencies using pipenv:
```bash
pipenv install
```

3. Activate the virtual environment:
```bash
pipenv shell
```

### Alternative Setup with pip

If you prefer not to use pipenv:

1. Clone and enter the repository:
```bash
git clone https://github.com/ganileni/institutions-economic-complexity-paper-code.git
cd institutions-economic-complexity-paper-code
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running All Analyses

To run the complete analysis pipeline (predictions, plots, and statistics):

**Preferred method (using pipenv):**
```bash
pipenv run python run_all.py
```

**Note:** The bootstrap analysis can be skipped for faster execution:

```bash
pipenv run python run_all.py --skip-bootstrap
```

### Running Individual Components

Run only predictions:
```bash
python run_all.py --predictions
```

Generate plots only (requires predictions to exist):
```bash
python run_all.py --plots
```

Run statistical analysis only (requires predictions to exist):
```bash
python run_all.py --stats
```

Or run scripts directly from the src directory:
```bash
cd src
python run_predictions.py
python generate_plots.py
python generate_stats.py
```

## Data Sources

- **Economic Fitness**: Derived from COMTRADE export data using the Fitness-Complexity algorithm
- **GDP per capita**: World Development Indicators (World Bank)
- **Polity**: Polity V dataset measuring regime characteristics
- **Technological Fitness**: Derived from PATSTAT patent data

Please check the paper for the cited sources.


## Output

The analysis generates:

1. **Predictions CSV**: `output/predictions/polity-short-4d-backfill.csv`
   - Contains predictions for all model configurations
   - Includes groundtruth values for comparison

2. **Plots**: `output/plots/`
   - Error comparison bar charts
   - Country-level error analysis
   - Trajectory plots in Fitness-GDP space
   - Marginal error distributions

3. **Statistical Analysis**: `output/plots/si/`
   - Bootstrap comparison figures
   - Model summary statistics

## Citation

If you use this code, please cite:

```bibtex
@article{angelini2024institutions,
  title={Institutions in Economic Complexity: Enhancing Growth Predictions and Theoretical Understanding},
  author={Angelini, Orazio and Tacchella, Andrea and Pietronero, Luciano and Di Matteo, T.},
  journal={[Journal TBD]},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions about the code, please contact: ganileni@gmail.com
