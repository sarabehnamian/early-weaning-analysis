# Early Weaning Analysis Pipeline

Analysis code for *Wealth, education and urbanization as predictors of the
duration of any breastfeeding in a multi-country survival analysis of 74 low-
and middle-income countries*, using DHS
(Demographic and Health Survey) data. The pipeline estimates the duration of
**any** breastfeeding in 2,467,589 children from 278 country–survey datasets in
74 low- and middle-income countries with survey-weighted survival methods, and
models the association of residence, household wealth and maternal education
with cessation.

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
│   ├── 08_pooled_and_residence_estimates.py  # pooled and urban/rural estimates
│   ├── 09_check_country_codes.py             # zip prefix vs the country code in the data (v000)
│   ├── 10_country_map.py                     # world map of children per country
│   ├── 11_key_numbers.py                     # every number quoted in the manuscript, one table
│   ├── 12_figures_and_tables.py              # collects the figures, rebuilds Tables S1 and S2
│   ├── 13_ph_checks.py                       # Schoenfeld test (full sample), time-split Cox model
│   ├── 14_sensitivity_models.py              # never-breastfed, no-wealth and recent-survey models
│   ├── 15_bootstrap_ci.py                    # PSU cluster-bootstrap CIs for medians and S(t)
│   ├── 16_svycoxph_india.py / .R             # lifelines vs R survey::svycoxph, India NFHS-5
│   ├── 17_ever_breastfed_logistic.py         # weighted logistic model of ever breastfeeding
│   ├── 18_km_figures.py                      # redraws Figures 5-7 (24-month reference line only)
│   ├── 19_cox_exclusions_table.py            # Table S8: surveys excluded from the Cox model
│   └── cox_common.py                         # Cox sample and model shared by 13, 14, 16, 17, 19
├── diagnostics/                        # one-off checks kept for the record
├── .gitignore
└── README.md
```

## Country codes and excluded files

Country is taken from the DHS country code the data carry (the first two letters
of `v000`), **not** from the zip-file prefix. Codes are DHS codes, not ISO codes
(e.g. MD = Madagascar, MB = Moldova, NI = Niger, BU = Burundi). An earlier version
used the prefix, which mislabelled files; `analysis/09_check_country_codes.py` lists every
file whose prefix differs from `v000`.

Two groups of files are excluded in `02_extract_breastfeeding_data.py` and written
to separate CSVs so they can be inspected:

| Files | Why excluded | Children | Output |
| ----- | ------------ | -------- | ------ |
| 26 NFHS-2 (India 1998–99) state files, e.g. KE = Kerala, MZ = Mizoram, TN = Tamil Nadu, TR = Tripura | `v000` = IA3; record-for-record subsets of the national IAKR42 survey | 30,879 | `india_state_files_excluded.csv`, `india_state_overlap_check.csv` |
| DRKR5A (2007) and DRKR6A (2013) | Dominican Republic batey special surveys: rural-only samples of a few provinces, not national | 1,743 | `dr_special_surveys_excluded.csv` |

Survey years are converted from raw `v007` (kept as `v007_raw`) to four-digit
Gregorian years: two-digit years 90–99 → 1990s and 0–30 → 2000s; Nepal
(Bikram Sambat) minus 57 years; Ethiopia (Ethiopian calendar) plus 8 years.
The conversion table is written to `survey_year_conversion.csv`.

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

Code 93 is common in more recent surveys: 93 of 278 country–survey datasets
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
python analysis\10_country_map.py --data ".\02_extract\combined_breastfeeding_data.csv" --out "r30_country_map"
```

Scripts 11–19 read fixed paths: `02_extract_v25`, `03_survival_v25`,
`04_comparative_v25`, `r27_key_numbers_v25` (output of script 08) and
`r26_cox_exact_v25` (output of script 05). They write to `v25_extras` and
`figures_v25`:

```bash
python analysis\11_key_numbers.py         # uses the PSU-clustered Cox fit if present, else marks CIs provisional
python analysis\12_figures_and_tables.py
python analysis\13_ph_checks.py
python analysis\14_sensitivity_models.py
python analysis\15_bootstrap_ci.py --reps 200 --workers 10
python analysis\16_svycoxph_india.py      # needs R with the survival and survey packages
python analysis\17_ever_breastfed_logistic.py
python analysis\18_km_figures.py
python analysis\19_cox_exclusions_table.py
```

Several of these fit Cox models on about one million children. Run them one at a
time on a machine with 16 GB of memory.

## Key variables

| Variable      | Use                                             |
| ------------- | ----------------------------------------------- |
| `m4`          | breastfeeding duration and status (see above)   |
| `v005`        | survey weight, applied as v005/1,000,000        |
| `v000`        | DHS country code and phase; defines `country`   |
| `v007`        | year of interview, converted to Gregorian       |
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
3. Scaled Schoenfeld residual tests of the proportional-hazards assumption, and
   a time-split Cox model with interval-specific hazard ratios for residence and
   the richest wealth quintile
4. Log-rank tests for group comparisons
5. Sensitivity analyses: excluding never-breastfed children (m4 = 94), a model
   without the wealth index that retains surveys lacking `v190`, refits on
   recent surveys only, and a weighted logistic model of ever breastfeeding
6. PSU cluster-bootstrap confidence intervals for regional and country medians
   and continuation at 6 and 24 months
7. A design-based check of the Cox model (R `survey::svycoxph`) on India NFHS-5

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
statsmodels>=0.14.0   # 17
geopandas>=0.14.0     # 10
```

Optional, for 16: R ≥ 4.3 with the `survival` and `survey` packages.

## Citation

```
Behnamian, S., & Fogh, F. Wealth, education and urbanization as predictors of
the duration of any breastfeeding in a multi-country survival analysis of
74 low- and middle-income countries.

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
