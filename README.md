# Early Weaning Analysis Pipeline

Comprehensive analysis pipeline for global breastfeeding duration patterns using DHS (Demographic and Health Survey) data. This project performs survival analysis on breastfeeding duration across multiple countries, examining early weaning patterns, geographic variations, and socioeconomic inequalities.

## 📋 Overview

This pipeline processes DHS survey data to analyze breastfeeding duration patterns globally. It uses proper survey weights, handles complex survey designs, and produces publication-ready outputs including survival curves, statistical models, and inequality analyses.

## 🔬 Key Features

- **Automated DHS data scanning** - Validates and prepares datasets for analysis
- **Weighted survival analysis** - Uses proper DHS survey weights (v005) 
- **Kaplan-Meier curves** - Visualizes breastfeeding duration by country, region, and demographics
- **Cox proportional hazards models** - Identifies risk factors for early weaning
- **Inequality analysis** - Examines urban-rural, wealth, and education disparities
- **Geographic comparisons** - Regional and country-level rankings
- **Publication-ready outputs** - Excel tables, high-resolution plots (300 DPI), and comprehensive logs

## 📁 Project Structure

```
early-weaning-analysis/
├── 00_check_members.py           # Quick scan of DHS zip files
├── 01_scan_to_excel.py           # Comprehensive readiness assessment
├── 02_extract_breastfeeding_data.py  # Extract and process survival data
├── 03_survival_analysis.py       # Kaplan-Meier and Cox regression
├── 04_comparative_analysis.py    # Geographic comparisons and inequalities
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
# Required packages
pip install pandas numpy matplotlib seaborn openpyxl xlsxwriter
pip install lifelines scipy pyreadstat
```

### Running the Pipeline

**Step 1: Quick check of your DHS data**
```bash
python 00_check_members.py ".\ZIPS"
```

**Step 2: Scan all datasets for breastfeeding readiness**
```bash
python 01_scan_to_excel.py ".\ZIPS" --out "Bfeed_scan.xlsx"
```

**Step 3: Extract breastfeeding data from ready datasets**
```bash
python 02_extract_breastfeeding_data.py ".\ZIPS" --scan "Bfeed_scan.xlsx" --out "02_extract_breastfeeding_data"
```

**Step 4: Run survival analysis**
```bash
python 03_survival_analysis.py --data ".\02_extract_breastfeeding_data\combined_breastfeeding_data.csv" --out "03_survival_analysis"
```

**Step 5: Perform comparative geographic analysis**
```bash
python 04_comparative_analysis.py --data ".\02_extract_breastfeeding_data\combined_breastfeeding_data.csv" --out "04_comparative_analysis"
```

## 📊 Outputs

### From Step 2 (Scanning)
- `Bfeed_scan.xlsx` - Multi-sheet workbook with:
  - Inventory of all .dta files found
  - KR_Readiness sheet (which datasets are ready for analysis)
  - Column preview for each dataset
  - Bad zips flagged

### From Step 3 (Extraction)
- `combined_breastfeeding_data.csv` - All countries merged
- `by_country/` - Individual country CSV files
- `country_summaries.xlsx` - Key statistics by country and survey year
- `extraction_log.txt` - Detailed processing log

### From Step 4 (Survival Analysis)
- `survival_curves/` - Kaplan-Meier plots (country, region, urban/rural)
- `cox_models/` - Forest plots of hazard ratios
- `tables/` 
  - `survival_results.xlsx` - Overall, country, and regional summaries
  - `early_weaning_by_country.xlsx` - Countries with highest early weaning
  - Early weaning visualization plots

### From Step 5 (Comparative Analysis)
- `rankings/` - Country rankings by median duration
- `inequalities/` - Urban-rural, wealth, education gaps
- `regional/` - Regional comparison tables and plots
- `final_tables/Summary_Tables.xlsx` - Publication-ready tables
- `analysis_summary.txt` - Executive summary

## 🔬 Methodology

### Data Sources
- **DHS Surveys**: Children's Recode (KR) files containing child-level breastfeeding data

### Key Variables
- **m4**: Breastfeeding status/duration
  - 0-93: Months of breastfeeding (stopped)
  - 94: Never breastfed
  - 95: Still breastfeeding (censored)
  - 96-99: Invalid/missing (excluded)
- **v005**: Survey weights (used as v005/1,000,000)
- **v021**: Primary sampling unit (PSU)
- **v022/v023**: Strata
- **b19** or **v008-b3**: Child's age in months

### Statistical Methods
1. **Kaplan-Meier Estimator** - Non-parametric survival curves with proper survey weights
2. **Cox Proportional Hazards** - Stratified by country and survey year, adjusted for:
   - Urban/rural residence
   - Wealth quintiles
   - Mother's education
3. **Log-rank Tests** - Group comparisons (urban vs rural, regional)
4. **Descriptive Statistics** - Weighted percentages and medians

### Handling Complex Survey Design
- Survey weights (v005) used in all analyses
- Stratification by country and survey year in Cox models
- Proper handling of censored observations (m4=95)

## 📈 Key Findings Structure

The analysis produces:
1. **Geographic variation** - Median duration ranges from ~3 to >36 months across countries
2. **Regional patterns** - South Asia and East Africa show longer durations
3. **Early weaning burden** - Proportion weaned before WHO's 6-month recommendation
4. **Never breastfed** - Countries where breastfeeding initiation is low
5. **Inequalities** - Within-country disparities by wealth, education, and residence

## 🛠️ Advanced Options

### Test with subset of data
```bash
# Process only first 10 zip files (fast check)
python 01_scan_to_excel.py ".\ZIPS" --out "test_scan.xlsx" --peek 10

# Process only specific countries
python 01_scan_to_excel.py ".\ZIPS" --include "AF*DT.zip" --include "BD*DT.zip"
```

## ⚠️ Important Notes

1. **DHS Data Access**: You need to register and download DHS datasets from [dhsprogram.com](https://dhsprogram.com/)
2. **Survey Weights**: All statistics use proper DHS weights (v005/1,000,000)
3. **Ethical Use**: DHS data has terms of use - results should be used for research/public health purposes


## 📝 Requirements

### Python Version
- Python 3.9+

### Core Dependencies
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

## 📚 Citation

If you use this pipeline in your research, please cite:

```
Early Weaning Analysis Pipeline
URL: https://github.com/sarabehnamian/early-weaning-analysis
```

And cite the DHS Program:
```
ICF. [Year]. Demographic and Health Surveys (various) [Datasets].
Funded by USAID. Rockville, Maryland: ICF [Distributor].
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaboration inquiries, please open an issue on this repository.

## 📄 License

This project is available for research and educational purposes. DHS data usage must comply with DHS terms of use.

## 🙏 Acknowledgments

- **DHS Program** for providing high-quality survey data
- **lifelines** package for survival analysis tools
- All contributors to open-source scientific Python packages

---

**Note**: This pipeline outputs data files in XLSX format for better compatibility and formatting options.
