# 02_extract_breastfeeding_data.py
# python 02_extract_breastfeeding_data.py ".\ZIPS" --scan ".\Bfeed_scan.xlsx" --out ".\02_extract_breastfeeding_data"

# Extracts breastfeeding survival data from all "ready" KR datasets identified in scan
# Creates individual country files and a combined dataset for survival analysis
# Uses proper DHS survey weights (v005/1e6) for all statistics

import argparse
import pandas as pd
import numpy as np
import zipfile
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
REQUIRED_VARS = ['m4', 'b19', 'v008', 'b3', 'b5', 'v005', 'v021', 'v024', 'v025']
OPTIONAL_VARS = ['v022', 'v023', 'v012', 'v106', 'v190', 'b4', 'v102',
                 'v201', 'v404', 'v501', 'm14', 'm15', 'm17', 'm19', 'h10', 'v007']
EXTRACT_VARS = REQUIRED_VARS + OPTIONAL_VARS

# -------------------- Helper Functions --------------------

def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    log_file = output_dir / "extraction_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def read_scan_results(scan_file: Path) -> pd.DataFrame:
    """Read the scan results to identify ready datasets."""
    df = pd.read_excel(scan_file, sheet_name='KR_Readiness')
    ready = df[df['status'] == 'ready'].copy()
    return ready

def extract_country_code(zip_name: str) -> str:
    """Extract country code from zip filename (first 2 letters)."""
    return zip_name[:2].upper()

def extract_survey_year(zip_name: str, df_data: pd.DataFrame = None) -> int:
    """Extract survey year from zip filename or data."""
    # If we have data with v007 (year of interview), use it
    if df_data is not None and 'v007' in df_data.columns:
        year = df_data['v007'].mode()
        if len(year) > 0:
            return int(year.iloc[0])

    # Otherwise parse from filename
    try:
        round_num = int(zip_name[4:6])
        if round_num >= 70:
            return 2015 + (round_num - 70) * 5
        elif round_num >= 60:
            return 2010 + (round_num - 60) * 5
        elif round_num >= 50:
            return 2005 + (round_num - 50) * 5
        elif round_num >= 40:
            return 2000 + (round_num - 40) * 5
        else:
            return 1990 + round_num * 5
    except:
        return 2010  # Default if parsing fails

def read_dta_from_zip(zip_path: Path, member: str, columns=None, nrows=None) -> pd.DataFrame:
    """Extract and read a DTA file from zip."""
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as td:
        temp_path = Path(td) / Path(member).name
        zf.extract(member, path=td)

        # Try to read with specified columns
        try:
            # Try pyreadstat first (faster for large files)
            import pyreadstat
            df, meta = pyreadstat.read_dta(
                str(temp_path),
                usecols=columns if columns else None,
                row_limit=nrows if nrows else None
            )
            return df
        except:
            # Fallback to pandas
            try:
                if nrows is not None:
                    df = pd.read_stata(temp_path, convert_categoricals=False, nrows=nrows)
                else:
                    df = pd.read_stata(temp_path, convert_categoricals=False)
                if columns:
                    keep_cols = [c for c in columns if c in df.columns]
                    df = df[keep_cols]
                return df
            except Exception as e:
                raise Exception(f"Failed to read {member}: {e}")

def process_kr_file(zip_path: Path, dta_name: str, country_code: str,
                    survey_year: int, logger) -> pd.DataFrame:
    """Process a single KR file and extract breastfeeding data."""

    logger.info(f"Processing {country_code} - {dta_name}")

    # Determine which variables are actually available
    try:
        # First, read just one row to get column names (FASTER)
        df_peek = read_dta_from_zip(zip_path, dta_name, columns=None, nrows=1)
        available_cols = list(df_peek.columns)

        # Find which of our desired variables are available
        cols_to_read = [c for c in EXTRACT_VARS if c in available_cols]

        # Read the full data with only needed columns
        df = read_dta_from_zip(zip_path, dta_name, columns=cols_to_read)

    except Exception as e:
        logger.error(f"Failed to read {dta_name}: {e}")
        return pd.DataFrame()

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get better survey year if v007 available
    if 'v007' in df.columns:
        year_from_data = df['v007'].mode()
        if len(year_from_data) > 0 and not np.isnan(year_from_data.iloc[0]):
            survey_year = int(year_from_data.iloc[0])

    # Calculate age in months
    if 'b19' in df.columns:
        df['age_months'] = df['b19']
    elif all(c in df.columns for c in ['v008', 'b3']):
        df['age_months'] = df['v008'] - df['b3']
    else:
        logger.warning(f"{country_code}: No age variable available")
        return pd.DataFrame()

    # Filter to eligible children (alive, under 60 months)
    if 'b5' in df.columns:
        eligible = (df['b5'] == 1) & (df['age_months'] >= 0) & (df['age_months'] < 60)
    else:
        eligible = (df['age_months'] >= 0) & (df['age_months'] < 60)

    df_eligible = df[eligible].copy()

    if len(df_eligible) == 0:
        logger.warning(f"{country_code}: No eligible children after filtering")
        return pd.DataFrame()

    # Create survival analysis variables
    df_eligible['country'] = country_code
    df_eligible['survey_year'] = survey_year

    # PROPER HANDLING OF M4 CODES (DHS STANDARD - CORRECTED)
    # Valid durations: 0-93 months (actual reported duration)
    # Never breastfed: 94 (event at duration 0)
    # Censored: 95 (still breastfeeding - use age as duration)
    # Invalid: 96, 97, 98, 99 (drop these)

    mask_real_duration = (df_eligible['m4'] >= 0) & (df_eligible['m4'] <= 93)
    mask_never_bf = (df_eligible['m4'] == 94)
    mask_censored = (df_eligible['m4'] == 95)
    mask_valid = mask_real_duration | mask_never_bf | mask_censored

    # Keep only valid observations
    df_final = df_eligible[mask_valid].copy()

    if len(df_final) == 0:
        logger.warning(f"{country_code}: No valid m4 codes")
        return pd.DataFrame()

    # Set duration and event indicators using proper masking
    mask_real = (df_final['m4'] >= 0) & (df_final['m4'] <= 93)
    mask_never = (df_final['m4'] == 94)
    mask_cens = (df_final['m4'] == 95)

    # Use np.where for cleaner assignment
    df_final['duration_months'] = np.where(
        mask_real, df_final['m4'],      # Use m4 value for real durations
        np.where(mask_never, 0,          # Never breastfed = 0 duration
                df_final['age_months'])   # Still BF = use current age
    )
    df_final['event'] = (mask_real | mask_never).astype(int)  # Both are events

    # Final validity check on duration
    valid_duration = (df_final['duration_months'] >= 0) & (df_final['duration_months'] <= 60)
    df_final = df_final[valid_duration].copy()

    # Add urban/rural if available
    if 'v025' in df_final.columns:
        df_final['urban'] = (df_final['v025'] == 1).astype(int)

    # Add wealth quintile if available
    if 'v190' in df_final.columns:
        df_final['wealth_quintile'] = df_final['v190']

    # Add mother's education if available
    if 'v106' in df_final.columns:
        df_final['mother_educ'] = df_final['v106']

    # Add region if available
    if 'v024' in df_final.columns:
        df_final['region'] = df_final['v024']

    # Keep only necessary columns for final dataset
    keep_cols = ['country', 'survey_year', 'duration_months', 'event', 'age_months',
                 'v005', 'v021']  # Always keep weights and PSU

    # Add optional columns if they exist
    for col in ['v022', 'v023', 'v024', 'v025', 'urban', 'wealth_quintile',
                'mother_educ', 'region', 'b4', 'v201', 'm14', 'm15']:
        if col in df_final.columns:
            keep_cols.append(col)

    df_output = df_final[keep_cols].copy()

    # Log never-breastfed count if m4 is available
    if 'm4' in df_final.columns:
        n_never = int((df_final['m4'] == 94).sum())
        logger.info(f"  Never-breastfed (m4=94): {n_never}")

    logger.info(f"  Extracted {len(df_output)} children from {country_code}")

    return df_output

def create_summary_statistics(all_data: pd.DataFrame, output_dir: Path):
    """Create enhanced summary statistics by country AND survey year with proper weighting."""

    summary_list = []
    logger = logging.getLogger(__name__)

    # Group by BOTH country and survey_year (don't pool across years)
    for (country, survey_year), country_data in all_data.groupby(['country', 'survey_year']):

        country_data = country_data.copy()

        # Calculate basic statistics
        n_children = len(country_data)
        n_events = country_data['event'].sum()
        n_censored = n_children - n_events
        
        # Count never-breastfed children
        n_never = int(((country_data['duration_months'] == 0) & 
                      (country_data['event'] == 1)).sum())

        # Calculate survey weights (v005 divided by 1,000,000 - DHS standard)
        weights = (country_data['v005'] / 1e6).values

        # Guard against invalid weights
        if np.any(weights <= 0) or np.any(np.isnan(weights)):
            logger.warning(f"{country}-{survey_year}: Invalid weights detected, using uniform weights")
            weights = np.ones(len(country_data))

        # Calculate weighted overall percentage weaned
        pct_weaned_weighted = 100.0 * np.sum(weights * country_data['event'].values) / np.sum(weights)
        pct_weaned_unweighted = n_events / n_children * 100

        # Try to calculate weighted Kaplan-Meier statistics
        try:
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter()

            # Guard against edge cases
            if country_data['event'].sum() == 0:
                raise ValueError("No events in data")

            # Fit with weights - CRUCIAL for DHS data
            kmf.fit(country_data['duration_months'].values,
                   event_observed=country_data['event'].values,
                   weights=weights)

            # Handle median duration properly
            median_dur = kmf.median_survival_time_
            if np.isinf(median_dur) or median_dur > 60:
                median_dur_display = ">60"
                median_dur_value = np.nan
            else:
                median_dur_display = f"{median_dur:.1f}"
                median_dur_value = float(median_dur)

            # Calculate percentage weaned at key timepoints (weighted)
            try:
                pct_weaned_6mo = (1 - kmf.survival_function_at_times(6).values[0]) * 100
                pct_weaned_12mo = (1 - kmf.survival_function_at_times(12).values[0]) * 100
                pct_weaned_24mo = (1 - kmf.survival_function_at_times(24).values[0]) * 100
            except:
                pct_weaned_6mo = pct_weaned_12mo = pct_weaned_24mo = np.nan

            # Also calculate unweighted for comparison
            kmf_unweighted = KaplanMeierFitter()
            kmf_unweighted.fit(country_data['duration_months'].values,
                             event_observed=country_data['event'].values)
            median_dur_unweighted = kmf_unweighted.median_survival_time_
            if np.isinf(median_dur_unweighted):
                median_dur_unweighted = np.nan

        except Exception as e:
            # Handle any KM fitting errors
            logger.warning(f"{country}-{survey_year}: KM fitting failed - {str(e)}")
            median_dur_value = np.nan
            median_dur_display = "NA"
            median_dur_unweighted = country_data['duration_months'].median()

            # Simple percentages (unweighted fallback)
            pct_weaned_6mo = ((country_data['duration_months'] <= 6) &
                            (country_data['event'] == 1)).sum() / n_children * 100
            pct_weaned_12mo = ((country_data['duration_months'] <= 12) &
                             (country_data['event'] == 1)).sum() / n_children * 100
            pct_weaned_24mo = ((country_data['duration_months'] <= 24) &
                             (country_data['event'] == 1)).sum() / n_children * 100

        # Calculate weighted and unweighted means
        weighted_mean = np.average(country_data['duration_months'].values, weights=weights)
        unweighted_mean = country_data['duration_months'].mean()

        summary_list.append({
            'country': country,
            'survey_year': survey_year,
            'n_children': n_children,
            'n_events': n_events,
            'n_censored': n_censored,
            'n_never_bf': n_never,
            'pct_weaned_weighted': pct_weaned_weighted,
            'pct_weaned_unweighted': pct_weaned_unweighted,
            'median_duration_weighted': median_dur_value,
            'median_display': median_dur_display,
            'median_duration_unweighted': median_dur_unweighted,
            'mean_duration_weighted': weighted_mean,
            'mean_duration_unweighted': unweighted_mean,
            'pct_weaned_6mo': pct_weaned_6mo,
            'pct_weaned_12mo': pct_weaned_12mo,
            'pct_weaned_24mo': pct_weaned_24mo
        })

    summary_df = pd.DataFrame(summary_list)

    # Sort by country and survey year
    summary_df = summary_df.sort_values(['country', 'survey_year'])

    # Save to Excel with formatting
    try:
        with pd.ExcelWriter(output_dir / 'country_summaries.xlsx', engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Summary']

            # Add formats
            number_format = workbook.add_format({'num_format': '#,##0'})
            decimal_format = workbook.add_format({'num_format': '0.0'})

            # Apply formats to columns
            worksheet.set_column('C:C', 12, number_format)  # n_children
            worksheet.set_column('D:F', 12, number_format)  # n_events, n_censored, n_never_bf
            worksheet.set_column('G:H', 15, decimal_format)  # pct_weaned
            worksheet.set_column('I:M', 15, decimal_format)  # durations
            worksheet.set_column('N:P', 15, decimal_format)  # pct_weaned at timepoints

            # Add note about weighting
            worksheet.write(len(summary_df) + 2, 0,
                          "Note: Weighted statistics use DHS survey weights (v005/1000000)")
            worksheet.write(len(summary_df) + 3, 0,
                          "Note: n_never_bf counts children who were never breastfed (m4=94)")
    except:
        # Fallback to simple Excel save
        summary_df.to_excel(output_dir / 'country_summaries.xlsx', index=False)

    return summary_df

# -------------------- Main Function --------------------

def main(zip_dir: str, scan_file: str, output_dir: str):
    """Main extraction function."""

    # Setup paths
    zip_dir = Path(zip_dir)
    scan_file = Path(scan_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for individual country files
    country_dir = output_dir / "by_country"
    country_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("BREASTFEEDING DATA EXTRACTION (v2.2 - Fixed m4=94)")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)

    # Read scan results
    ready_datasets = read_scan_results(scan_file)
    logger.info(f"Found {len(ready_datasets)} ready datasets")

    # Process each dataset
    all_data = []
    failed_files = []

    for idx, row in ready_datasets.iterrows():
        zip_name = row['zip']
        dta_name = row['dta']
        zip_path = zip_dir / zip_name

        if not zip_path.exists():
            logger.warning(f"Zip file not found: {zip_name}")
            failed_files.append(zip_name)
            continue

        country_code = extract_country_code(zip_name)
        survey_year = extract_survey_year(zip_name)  # Will be refined from data if v007 available

        try:
            # Process the KR file
            df_country = process_kr_file(zip_path, dta_name, country_code,
                                        survey_year, logger)

            if not df_country.empty:
                # Save individual country file
                actual_year = df_country['survey_year'].iloc[0]
                country_file = country_dir / f"{country_code}_{actual_year}_breastfeeding.csv"
                df_country.to_csv(country_file, index=False)

                # Add to combined dataset
                all_data.append(df_country)

        except Exception as e:
            logger.error(f"Failed to process {zip_name}: {e}")
            failed_files.append(zip_name)

    # Combine all data
    if all_data:
        logger.info("\nCombining all country data...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save combined dataset
        logger.info(f"Saving combined dataset with {len(combined_df)} children...")
        combined_df.to_csv(output_dir / 'combined_breastfeeding_data.csv', index=False)

        # Also save as Stata file if possible
        try:
            combined_df.to_stata(output_dir / 'combined_breastfeeding_data.dta',
                               write_index=False, version=117)
            logger.info("Saved Stata format (.dta)")
        except:
            logger.warning("Could not save Stata format")

        # Create summary statistics
        logger.info("\nCreating summary statistics...")
        summary_df = create_summary_statistics(combined_df, output_dir)
        logger.info(f"Created summary for {len(summary_df)} country-survey combinations")

        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"Total children extracted: {len(combined_df):,}")
        logger.info(f"Total countries: {combined_df['country'].nunique()}")
        logger.info(f"Total country-survey combinations: {len(summary_df)}")
        logger.info(f"Total events (stopped BF): {combined_df['event'].sum():,}")
        logger.info(f"Total censored: {(1-combined_df['event']).sum():,}")
        logger.info(f"Failed files: {len(failed_files)}")
        if failed_files:
            logger.info(f"Failed: {', '.join(failed_files[:5])}" +
                       ("..." if len(failed_files) > 5 else ""))
        logger.info("=" * 60)
        logger.info("Note: All statistics use DHS survey weights (v005/1000000)")
        logger.info("Note: Never-breastfed children (m4=94) are included as events at duration 0")

    else:
        logger.error("No data extracted!")

    logger.info(f"Completed at: {datetime.now()}")
    logger.info(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract breastfeeding data from ready KR datasets"
    )
    parser.add_argument("zip_dir", help="Directory containing DHS zip files")
    parser.add_argument("--scan", default="Bfeed_scan.xlsx",
                       help="Excel file from 01_scan_to_excel.py")
    parser.add_argument("--out", default="02_extract_breastfeeding_data",
                       help="Output directory for extracted data")

    args = parser.parse_args()

    main(args.zip_dir, args.scan, args.out)