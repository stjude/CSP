# CSP stands for cumulative survival profiling
Validation and detection of heteroresistance and related outcomes using cumulative-survival-profiling
# Raw data format
For heteroresistance validation and detection via csp, we introduce a new format for isolate's data. In the new format, the isolate data is entered in tabular form such that each row represents a distinct PAP and the columns represent the antibiotic-gradient concentrations (in ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20g/mL)). Thus, the entries in the table would be the raw bacterial counts (in CFU/mL) of the corresponding PAP at each of the antibiotic concentration values.    
If an isolate has multiple measured PAPs each PAP would contribute a row and the names of the multiple PAPs (the row names) would be distinct and indicate that they are for the same isolate (for e.g. iso1_pap1, iso1_pap2, ..etc).
Manipulation of pre-cleaned raw PAP data by averaging, replacing missing data by arbitrary values ..etc must be avoided. Averaging tends to underestimate quantities that has missing values and thus introduces artifacts in data. For example averaging the two PAPs for isos1 below

|          |0 |1  |2  |3  |4  |6|8|
|----------|--|---|---|---|---|---|-|
|iso1_pap1|1e8|9e7|5e7|   |1e3|2e2|3|
|iso1_pap2|   |8e7|   |2e5|   |1e2||5|
|iso2_pap1|2.3e8|3e7|   | |   | | | |

would half the initial inoculum (counts at 0 ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20g/mL)) and the counts at 3 and 4 ![equation](https://latex.codecogs.com/gif.latex?%5Cmu%20g/mL).
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
