# CSP stands for cumulative survival profiling
Validation and detection of heteroresistance and related outcomes in staphylococci using cumulative-survival-profiling. CSP is a population-analysis-profile-based (PAP-based) method.
# Raw data format
For heteroresistance validation and detection via csp, we introduce a new format for isolate's data. In the new format, the isolate data is entered in tabular form such that each row represents a distinct PAP and the columns represent the antibiotic-gradient concentrations (in ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20g/mL)). Thus, the entries in the table would be the raw bacterial counts (in CFU/mL) of the corresponding PAP at each of the antibiotic concentration values.    
If an isolate has multiple measured PAPs each PAP would contribute a row and the names of the multiple PAPs (the row names) would be distinct and indicate that they are for the same isolate (for e.g. iso1_pap1, iso1_pap2, ..etc).
Manipulation of pre-cleaned raw PAP data by averaging, replacing missing data by arbitrary values ..etc must be avoided. Averaging tends to underestimate quantities that has missing values and thus introduces artifacts in data. For example averaging the two PAPs for isos1 below

|          |0 |1  |2  |3  |4  |6|8|
|----------|--|---|---|---|---|---|-|
|iso1_pap1|1e8|9e7|5e7|   |1e3|2e2|3|
|iso1_pap2|   |8e7|   |2e5|   |1e2||5|
|iso2_pap1|2.3e8|3e7|   | |   | | | |

would halve the initial inoculum (counts at 0 ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20g/mL)) and the counts at 3 and 4 ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20g/mL).
# Clean data
The first step after formatting the raw data as described above is to keep clean data and drop PAPs for which we cannot calculate the area under the counts curve in a reliable way. For the reasons mentioned above, we strongly suggest dropping PAPs with a missing initial inoculum and PAPs with measured counts at fewer than three concentrations. For example iso1_pap2 in the table above should be dropped from the analyses for missing the initial inoculum and iso2_pap1 should be dropped for having less than three points. The main algorithm in the script "csp_main.py" does this cleaning step for both the reference and the isolates if both are supplied.
# Separate files for reference and isolates
Separate data files for reference strain and test isolates should be arragned in the above tabular format. To profile a reference strain (e.g. Mu3) and use it to classify isolates, multiple PAPs of the reference spanning wide range of initial inoculum values must be available.
# Profiling reference and classifying isolates
The python script "csp_main.py" encompass the main algorithm of CSP by which the reference strain is profiled and its cumulative survival profile is used to classify a number of isolates. 
To display what inputs this scripts takes and what some inputs default values are use: 

``` 
./csp_main.py -h 
usage: csp_main.py [-h] --refPAPdata REFPAPDATA [--IsoPAPdata ISOPAPDATA]
                   [--lowestDetectableCFU LOWESTDETECTABLECFU]
                   [--regselect REGSELECT] [--xvfrac XVFRAC] [--xviter XVITER]

This program is built to analyze a set of PAPs to profile a reference strain
(e.g. Mu3) and then use this profiling to predict heteroresistance in a set of
clinical isolates

optional arguments:
  -h, --help            show this help message and exit
  --refPAPdata REFPAPDATA, -m REFPAPDATA
                        a CSV file encompassing population analysis profile data
                        for Mu3 with specific format: rows named by the
                        isolates, columns named by the antimicrobial
                        concentration (mic-g/mL) in ascending order starting
                        from 0 and entries are the counts in (CFU/mL)
  --IsoPAPdata ISOPAPDATA, -i ISOPAPDATA
                        a CSV file having population analysis profiling data
                        for isolates with specific format: rows named by the
                        isolates, columns named by the antimicrobial
                        concentration (mic-g/mL) in ascending order starting
                        from 0 and entries are the CFUs in (cells/mL)
  --lowestDetectableCFU LOWESTDETECTABLECFU, -b LOWESTDETECTABLECFU
                        lowest detectable CFU that will be used to replace any
                        fewer counts (default=0.0)
  --regselect REGSELECT, -s REGSELECT
                        select whether regression is in "Linear" or in "Log10"
                        space (default)
  --xvfrac XVFRAC, -xf XVFRAC
                        select fraction of reference PAPs for training set in
                        cross-validation (default=0.8)
  --xviter XVITER, -xi XVITER
                        select number of iterations for cross-validation (default=10)
                        
```
Running csp_main.py using reference and isolates data gives the following outputs:
```
csp_run.log: detailed log file for the run
crossValidation_CSP.csv: detailed cross-validation output
crossValidation_CSP_summary.txt: a summary for the cross-validation output
regFit_CSP_Log10.png: a plot showing the regression fit to data
refTrace.csv: the MCMC trace for the posterior samples 
ref_and_isolates_paps.csv: the reference and isolates' PAPs saved appended togother with added features
```
# Installing required packages 

Required Python packages to run the csp_main.py script are listed in the `requirements.txt` file. The script requires **PyMC 5.28.4** and Python 3.11+.

## Option 1: Using Python Virtual Environment (venv) - Recommended

This is the recommended approach for this project. Create a Python virtual environment and install packages using pip:

```bash
# Create a virtual environment named .myenv
python3 -m venv .myenv

# Activate the virtual environment
source .myenv/bin/activate  # On Windows: .myenv\Scripts\activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

After installation, verify that PyMC 5.28.4 is installed:

```bash
python -c "import pymc as pm; print(f'PyMC version: {pm.__version__}')"
```

## Option 2: Using Conda

Alternatively, you can use conda to create an environment and install packages:

```bash
# Create a new conda environment named csp (optional)
conda create -n csp python=3.11

# Activate the environment
conda activate csp

# Install packages from requirements.txt
conda install --yes --file requirements.txt

```

## Option 3: Using Mamba (Faster Alternative to Conda)

For faster package resolution and installation, use mamba:

```bash
# Create environment with mamba
mamba create -n csp -c conda-forge --yes --file requirements.txt

# Activate the environment
mamba activate csp
```

## Activating the Environment

Once installed, activate the environment before running the script:

```bash
# For venv:
source .myenv/bin/activate  # On Windows: .myenv\Scripts\activate

# For conda/mamba:
conda activate csp          # or: mamba activate csp
```

## Verifying Installation

To verify that all packages are correctly installed:

```bash
# Check PyMC version
python -c "import pymc as pm; print(f'PyMC: {pm.__version__}')"

# Check key dependencies
python -c "import numpy, scipy, pandas, matplotlib, seaborn, statsmodels, sklearn, arviz; print('All packages imported successfully!')"
```

## Package Information

- **PyMC**: 5.28.4 (Bayesian modeling and probabilistic programming)
- **PyTensor**: 2.38.2 (Tensor computation framework, successor to Theano)
- **ArviZ**: 0.23.4 (Posterior analysis visualization)
- **NumPy**: 2.4.4 (Numerical computing)
- **SciPy**: 1.17.1 (Scientific computing)
- **Pandas**: 3.0.2 (Data manipulation)
- **Matplotlib**: 3.10.8 (Visualization)
- **Seaborn**: 0.13.2 (Statistical visualization)
- **Statsmodels**: 0.14.6 (Statistical modeling)
- **Scikit-learn**: 1.8.0 (Machine learning)
# Reference
Ramzi A. Alsallaq, Tina Dao, Jason W. Rosch, Elisa Margolis "Cumulative survival profiling: a new PAP-based method for detecting heteroresistance in staphylococcal clinical isolates" https://www.medrxiv.org/content/10.1101/2020.08.10.20148502v1 
