# Early Weaning Analysis Pipeline

Analysis code for *Wealth, education and urbanization as predictors of early
cessation of any breastfeeding in a multi-country survival analysis of 96 low-
and middle-income countries*, using DHS
(Demographic and Health Survey) data. The pipeline estimates the duration of
**any** breastfeeding across 96 low- and middle-income countries with
survey-weighted survival methods, and models the association of residence,
household wealth and maternal education with cessation.

**No data are included in this repository.** DHS files must be obtained from
[dhsprogram.com](https://dhsprogram.com/) under their terms of use. All outputs
(CSV, XLSX, figures, logs) are reproducible from the scripts and are not tracked.

## Project structure

```
early-weaning-analysis/
├── 00_check_members.py                 # quick scan of DHS zip files
├── 01_scan_to_excel.py                 # readiness assessment of every KR dataset
├── 02_extract_breastfeeding_data.py    # build the child-level survival dataset
├── 03_survival_analysis.py             # survival curves, forest plot, early cessation
├── 04_comparative_analysis.py          # country rankings, regions, inequalities
├── analysis/
│   ├── 05_cox_retrospective_surveys.py       # primary Cox model, PSU-clustered SEs
│   ├── 06_cox_sensitivity_checks.py          # never-breastfed, PH test, temporal subsets
│   ├── 07_turnbull_descriptive_summaries.py  # Turnbull vs Kaplan-Meier comparison
│   └── 08_pooled_and_residence_estimates.py  # pooled and urban/rural estimates
├── diagnostics/                        # one-off checks kept for the record
├── .gitignore
└── README.md
```

## Handling of the DHS variable m4

`m4` records the duration of any breastfeeding, and its special codes decide the
type of observation. Reading them correctly matters: an earlier version of this
pipeline treated code 93 as a duration of 93 months, which silently dropped
those children.

| m4       | Meaning                                        | Treatment                          |
| -------- | ---------------------------------------------- | ---------------------------------- |
| 0–92     | Completed months of breastfeeding              | exact duration, cessation event    |
| 93       | Ever breastfed, not currently, duration unknown | **left-censored** at current age   |
| 94       | Never breastfed                                | cessation event at time zero       |
| 95       | Still breastfeeding                            | right-censored at current age      |
| 96–99    | Died while breastfeeding, inconsistent, unknown | excluded                          |

Code 93 is common in more recent surveys: 93 of 303 country–survey datasets
record no retrospective durations at all. The analytic dataset therefore carries
three kinds of observation, flagged in `censor_type` with bounds in `t_lower`
and `t_upper`.

## Estimators

- **Descriptive results** use the weighted **Turnbull** non-parametric maximum
  likelihood estimator wherever left-censored children are present, and the
  ordinary **Kaplan–Meier** estimator otherwise. Cessation before six months is
  reported as 1 − S(6) from the same curve, not as a count of recorded durations
  under six months.
- **Cox proportional-hazards models** cannot use left-censored observations, so
  they are restricted to the surveys that recorded retrospective durations, with
  standard errors clustered on the survey PSU (`v021`) and stratification by
  country and survey year.

## Quick start

```bash
pip install pandas numpy matplotlib seaborn openpyxl xlsxwriter
pip install lifelines scipy pyreadstat
```

```bash
# 1. inventory and readiness
python 00_check_members.py ".\data"
python 01_scan_to_excel.py ".\data" --out "Bfeed_scan.xlsx"

# 2. build the analytic dataset
python 02_extract_breastfeeding_data.py ".\data" --scan "Bfeed_scan.xlsx" --out "02_extract"

# 3. primary Cox model (add --fast for model-based SEs while testing)
python analysis\05_cox_retrospective_surveys.py --data ".\02_extract\combined_breastfeeding_data.csv" ^
       --out "05_cox_model"

# 4. curves, figures and comparative outputs
python 03_survival_analysis.py --data ".\02_extract\combined_breastfeeding_data.csv" ^
       --out "03_survival" --cox-results ".\05_cox_model\cox_exact_results.xlsx"
python 04_comparative_analysis.py --data ".\02_extract\combined_breastfeeding_data.csv" --out "04_comparative"

# 5. supporting estimates
python analysis\07_turnbull_descriptive_summaries.py --data ".\02_extract\combined_breastfeeding_data.csv" ^
       --out "07_turnbull"
python analysis\08_pooled_and_residence_estimates.py --data ".\02_extract\combined_breastfeeding_data.csv" ^
       --out "08_key_estimates"
python analysis\06_cox_sensitivity_checks.py --data ".\02_extract\combined_breastfeeding_data.csv" ^
       --out "06_sensitivities"
```

## Key variables

| Variable      | Use                                             |
| ------------- | ----------------------------------------------- |
| `m4`          | breastfeeding duration and status (see above)   |
| `v005`        | survey weight, applied as v005/1,000,000        |
| `v021`        | primary sampling unit, used for clustered SEs   |
| `v022`/`v023` | sampling strata                                 |
| `v025`        | urban/rural residence                           |
| `v190`        | household wealth quintile                       |
| `v106`        | maternal education                              |
| `b19`, `b5`   | child's age in months, survival status          |

## Statistical methods

1. Weighted Turnbull and Kaplan–Meier estimators for duration and continuation
   at 6, 12 and 24 months
2. Cox proportional-hazards models stratified by country and survey year, with
   PSU-clustered sandwich standard errors, adjusted for residence, wealth
   quintile and maternal education
3. Scaled Schoenfeld residual tests of the proportional-hazards assumption
4. Log-rank tests for group comparisons
5. Sensitivity analyses: excluding never-breastfed children, a model without the
   wealth index that retains surveys lacking `v190`, and refits on recent
   surveys only

## Requirements

Python 3.9+

```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
lifelines>=0.27.0
scipy>=1.10.0
openpyxl>=3.1.0
xlsxwriter>=3.0.0
pyreadstat>=1.2.0
```

## Citation

```
Behnamian, S., & Fogh, F. Wealth, education and urbanization as predictors of
early cessation of any breastfeeding in a multi-country survival analysis of
96 low- and middle-income countries.

GitHub repository: https://github.com/sarabehnamian/early-weaning-analysis
```

Please also cite the DHS Program:

```
ICF. [Year]. Demographic and Health Surveys (various) [Datasets].
Funded by USAID. Rockville, Maryland: ICF [Distributor].
```

## Contact

**Sara Behnamian** — Department of Biology, Lund University, Sweden; Section for
GeoGenetics, Globe Institute, University of Copenhagen, Denmark
<sara.behnamian@biol.lu.se> · <sara.behnamian@sund.ku.dk>

**Fatemeh Fogh** — Department of Electrical Engineering and Computer Science,
Florida Atlantic University, USA
<ffogh2021@fau.edu>

## License

Available for research and educational purposes. Use of DHS data must comply
with the DHS Program's terms of use.
