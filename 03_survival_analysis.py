# 03_survival_analysis.py
# python 03_survival_analysis.py --data ".\02_extract_breastfeeding_data\combined_breastfeeding_data.csv" --out ".\03_survival_analysis"

# Performs weighted survival analysis on breastfeeding duration data
# Uses DHS survey weights (v005/1e6) for all analyses

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Survival analysis libraries
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
plt.style.use('default')  # Use default style to avoid seaborn version issues
sns.set_palette("husl")

# -------------------- Configuration --------------------

# Regional groupings based on DHS country codes
REGIONS = {
    'West Africa': ['BF', 'BJ', 'CI', 'GH', 'GM', 'GN', 'LB', 'ML', 'NI', 'NG', 'SL', 'SN', 'TG',
                    'MR'],  # Added MR (Mauritania)

    'East Africa': ['BI', 'BU', 'ER', 'ET', 'KE', 'KM', 'MW', 'RW', 'TZ', 'UG', 'ZM', 'ZW', 'MZ', 'LS',
                    'MG'],  # Added MG (Madagascar)

    'Central Africa': ['AO', 'CD', 'CG', 'CM', 'GA', 'ST', 'TD', 'CF'],

    'North Africa': ['EG', 'MA', 'TN'],

    'Southern Africa': ['ZA', 'SZ', 'NM'],  # South Africa, Eswatini, Namibia

    'South Asia': ['AF', 'BD', 'IA', 'IN', 'LK', 'MV', 'NP', 'PK',
                   # Indian states (subnational DHS data):
                   'AP', 'AS', 'BH', 'DL', 'GJ', 'GO', 'HP', 'HR', 'KA',
                   'MB', 'MH', 'MN', 'MP', 'OR', 'PJ', 'RJ', 'SK', 'UP', 'WB'],

    'Southeast Asia': ['ID', 'KH', 'LA', 'MM', 'PH', 'TH', 'TL', 'VN',
                       'PG'],  # Added PG (Papua New Guinea)

    'Latin America & Caribbean': ['BO', 'BR', 'CO', 'DR', 'EC', 'GT', 'GU', 'HN', 'HT',
                                  'MX', 'NC', 'PE', 'PY',
                                  'AR', 'GY', 'JM'],  # Added Argentina, Guyana, Jamaica

    'Europe/Central Asia': ['AL', 'AM', 'AZ', 'KK', 'KY', 'MD', 'TJ', 'TR', 'UA', 'UZ'],

    'Middle East': ['JO', 'YE'],  # Jordan, Yemen
}

# -------------------- Helper Functions --------------------

def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / "survival_analysis_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def assign_region(country_code: str) -> str:
    """Assign region based on country code."""
    for region, countries in REGIONS.items():
        if country_code in countries:
            return region
    return 'Other'

def create_output_dirs(base_dir: Path):
    """Create output directory structure."""
    dirs = {
        'base': base_dir,
        'curves': base_dir / 'survival_curves',
        'cox': base_dir / 'cox_models',
        'tables': base_dir / 'tables'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

# -------------------- Analysis Functions --------------------

def kaplan_meier_by_group(data: pd.DataFrame, group_var: str, output_dir: Path,
                          title_suffix: str = "", top_n: int = 12):
    """Create weighted Kaplan-Meier curves by group with cleaner visualization."""
    
    # Use tab20 colormap for distinct colors
    cmap = cm.tab20  # Direct access to colormap
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(12, 8))

    groups = data[group_var].dropna().unique()
    if top_n and len(groups) > top_n:
        groups = data.groupby(group_var).size().nlargest(top_n).index

    medians = {}
    for i, group in enumerate(sorted(groups)):
        g = data.loc[data[group_var] == group].copy()
        if len(g) < 30 or g['event'].sum() == 0:
            continue

        # weights if available
        if 'weight' in g.columns:
            w = g['weight']
            valid = w.notna() & (w > 0)
            g = g.loc[valid]
            if len(g) < 30 or g['event'].sum() == 0 or g['weight'].sum() <= 0:
                continue
            kmf.fit(g['duration_months'], event_observed=g['event'], weights=g['weight'],
                    label=f"{group} (n={len(g):,})")
        else:
            kmf.fit(g['duration_months'], event_observed=g['event'],
                    label=f"{group} (n={len(g):,})")

        # Get color from colormap
        color = cmap(i % 20)  # tab20 has 20 colors
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, linewidth=2)
        medians[group] = kmf.median_survival_time_

    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Probability of Still Breastfeeding', fontsize=12)
    ax.set_title(f'Breastfeeding Duration by {title_suffix} (Weighted)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 36)
    ax.grid(True, alpha=0.25)

    # WHO 6-month reference
    ax.axvline(x=6, color='#CC3344', linestyle='--', alpha=0.6, label='WHO 6-month recommendation')

    # lean legend outside
    leg = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=9, borderaxespad=0.)
    
    # FIXED: Use 'legend_handles' instead of 'legendHandles'
    for lh in leg.legend_handles:
        lh.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(output_dir / f"km_by_{group_var.replace('/', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    return medians

def cox_regression_analysis(data: pd.DataFrame, output_dir: Path, logger):
    """Perform weighted Cox proportional hazards regression."""

    logger.info("Running Cox regression analysis...")

    # Prepare data for Cox regression
    cox_data = data.copy()

    # Ensure we have required columns
    if 'survey_year' not in cox_data.columns:
        cox_data['survey_year'] = cox_data.groupby('country')['survey_year'].transform('first')

    # Create dummy variables for categorical variables
    if 'urban' in cox_data.columns:
        cox_data['urban'] = cox_data['urban'].fillna(0)

    if 'wealth_quintile' in cox_data.columns:
        # Create wealth quintile dummies (using quintile 3 as reference)
        for q in [1, 2, 4, 5]:
            cox_data[f'wealth_q{q}'] = (cox_data['wealth_quintile'] == q).astype(int)

    if 'mother_educ' in cox_data.columns:
        # Group education into categories
        cox_data['educ_none'] = (cox_data['mother_educ'] == 0).astype(int)
        cox_data['educ_primary'] = (cox_data['mother_educ'] == 1).astype(int)
        cox_data['educ_secondary_plus'] = (cox_data['mother_educ'] >= 2).astype(int)

    # Select variables for model
    model_vars = ['duration_months', 'event', 'weight', 'country', 'survey_year']

    # Add available covariates
    potential_vars = ['urban', 'wealth_q1', 'wealth_q2', 'wealth_q4', 'wealth_q5',
                     'educ_none', 'educ_secondary_plus']

    for var in potential_vars:
        if var in cox_data.columns:
            model_vars.append(var)

    # Filter to complete cases with valid weights
    cox_final = cox_data[model_vars].dropna()
    cox_final = cox_final[cox_final['weight'] > 0]

    if len(cox_final) < 100:
        logger.warning("Insufficient data for Cox regression")
        return None

    # Fit Cox model with weights and stratification
    cph = CoxPHFitter()

    try:
        # Remove stratification variables from covariates
        covariate_cols = [c for c in cox_final.columns
                         if c not in ['duration_months', 'event', 'weight', 'country', 'survey_year']]

        cph.fit(cox_final,
               duration_col='duration_months',
               event_col='event',
               weights_col='weight',
               strata=['country', 'survey_year'],
               robust=True)

        # Save results
        results_df = cph.summary
        results_df.to_excel(output_dir / 'cox_regression_results.xlsx')

        # Create forest plot
        fig, ax = plt.subplots(figsize=(10, 6))
        cph.plot(ax=ax)
        ax.set_title('Hazard Ratios for Early Weaning (Weighted)', fontsize=14, fontweight='bold')
        ax.axvline(x=1, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()

        # Save only once to cox_models folder at 300 dpi
        plt.savefig(output_dir / 'hazard_ratios_forest_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Cox regression completed. C-index: {cph.concordance_index_:.3f}")

        return cph

    except Exception as e:
        logger.error(f"Cox regression failed: {e}")
        return None

def analyze_early_weaning(data: pd.DataFrame, output_dir: Path):
    """Analyze proportion of very early weaning (<6 months) using weights."""

    results = []

    for country in data['country'].unique():
        country_data = data[data['country'] == country].copy()

        if 'weight' in country_data.columns:
            # Weighted calculation
            w = country_data['weight']
            valid = w.notna() & (w > 0)
            cw = country_data.loc[valid]

            if len(cw) == 0:
                continue

            ew = cw[(cw['event'] == 1) & (cw['duration_months'] < 6)]

            pct_weaned_before_6m = 100.0 * ew['weight'].sum() / cw['weight'].sum() if cw['weight'].sum() > 0 else 0
            n_weaned = len(ew)
        else:
            # Unweighted fallback
            country_early = country_data[(country_data['event'] == 1) &
                                        (country_data['duration_months'] < 6)]
            pct_weaned_before_6m = (len(country_early) / len(country_data) * 100) if len(country_data) > 0 else 0
            n_weaned = len(country_early)

        results.append({
            'country': country,
            'n_children': len(country_data),
            'n_weaned_before_6m': n_weaned,
            'pct_weaned_before_6m': pct_weaned_before_6m,
            'median_early_weaning_age': country_data[(country_data['event'] == 1) &
                                                    (country_data['duration_months'] < 6)]['duration_months'].median()
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pct_weaned_before_6m', ascending=False)
    results_df.to_excel(output_dir / 'early_weaning_by_country.xlsx', index=False)

    # Create bar plot of top 20 countries with highest early weaning
    top20 = results_df.head(20)

    if len(top20) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gradient colors from dark red (highest) to light yellow (lowest)
        # Using a red-orange-yellow gradient for better visual impact
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(top20))[::-1])
        
        bars = ax.bar(range(len(top20)), top20['pct_weaned_before_6m'], color=colors)
        ax.set_xticks(range(len(top20)))
        ax.set_xticklabels(top20['country'], rotation=45, ha='right')
        ax.set_ylabel('% Weaned Before 6 Months (Weighted)', fontsize=12)
        ax.set_title('Countries with Highest Rates of Early Weaning (<6 months)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # NO VALUE LABELS - removed the for loop that added text labels

        plt.tight_layout()

        # Save only once to tables folder at 300 dpi
        plt.savefig(output_dir / 'early_weaning_top20_countries.png', dpi=300, bbox_inches='tight')
        plt.close()

    return results_df

def weighted_mean(x, w):
    """Calculate weighted mean."""
    valid = w.notna() & (w > 0) & x.notna()
    if valid.sum() == 0:
        return np.nan
    return np.average(x[valid], weights=w[valid])

def create_summary_report(data: pd.DataFrame, output_dir: Path, logger):
    """Create comprehensive summary report with weighted statistics."""

    logger.info("Creating summary report...")

    with pd.ExcelWriter(output_dir / 'survival_results.xlsx', engine='openpyxl') as writer:

        # Overall summary (weighted)
        if 'weight' in data.columns:
            w = data['weight']
            valid = w.notna() & (w > 0)
            dw = data.loc[valid]

            overall = pd.DataFrame([{
                'Total Children': len(data),
                'Total Countries': data['country'].nunique(),
                'Total Events (Weaned)': int(data['event'].sum()),
                'Total Censored': int((1 - data['event']).sum()),
                'Weaning Rate % (weighted)': 100.0 * (dw['event']*dw['weight']).sum() / dw['weight'].sum() if dw['weight'].sum() > 0 else np.nan,
                'Weaning Rate % (unweighted)': data['event'].mean() * 100,
                'Mean Duration (weighted)': np.average(dw['duration_months'], weights=dw['weight']) if len(dw) > 0 else np.nan,
                'Mean Duration (unweighted)': data['duration_months'].mean(),
                'Median Follow-up (months)': data['duration_months'].median()
            }])
        else:
            # Unweighted version
            overall = pd.DataFrame([{
                'Total Children': len(data),
                'Total Countries': data['country'].nunique(),
                'Total Events (Weaned)': int(data['event'].sum()),
                'Total Censored': int((1 - data['event']).sum()),
                'Overall Weaning Rate (%)': data['event'].mean() * 100,
                'Mean Duration (months)': data['duration_months'].mean(),
                'Median Follow-up (months)': data['duration_months'].median()
            }])

        overall.T.to_excel(writer, sheet_name='Overall_Summary')

        # By country (weighted)
        if 'weight' in data.columns:
            country_groups = []
            for country, group in data.groupby('country'):
                w = group['weight']
                valid = w.notna() & (w > 0)
                if valid.sum() == 0:
                    continue
                gw = group.loc[valid]

                country_groups.append({
                    'country': country,
                    'N': len(group),
                    'Mean_Duration_Weighted': np.average(gw['duration_months'], weights=gw['weight']),
                    'Mean_Duration_Unweighted': group['duration_months'].mean(),
                    'Median_Duration': group['duration_months'].median(),
                    'N_Events': group['event'].sum(),
                    'Event_Rate_Weighted': (gw['event']*gw['weight']).sum()/gw['weight'].sum()*100,
                    'Event_Rate_Unweighted': group['event'].mean() * 100
                })

            country_summary = pd.DataFrame(country_groups)
            country_summary = country_summary.sort_values('Median_Duration', ascending=False)
        else:
            # Unweighted version
            country_summary = data.groupby('country').agg({
                'duration_months': ['count', 'mean', 'median'],
                'event': ['sum', 'mean']
            }).round(2)
            country_summary.columns = ['N', 'Mean_Duration', 'Median_Duration', 'N_Events', 'Event_Rate']
            country_summary = country_summary.sort_values('Median_Duration', ascending=False)

        country_summary.to_excel(writer, sheet_name='By_Country', index=False)

        # By region
        data['region'] = data['country'].apply(assign_region)

        if 'weight' in data.columns:
            region_groups = []
            for region, group in data.groupby('region'):
                w = group['weight']
                valid = w.notna() & (w > 0)
                if valid.sum() == 0:
                    continue
                gw = group.loc[valid]

                region_groups.append({
                    'region': region,
                    'N': len(group),
                    'Mean_Duration_Weighted': np.average(gw['duration_months'], weights=gw['weight']),
                    'Mean_Duration_Unweighted': group['duration_months'].mean(),
                    'Median_Duration': group['duration_months'].median(),
                    'N_Events': group['event'].sum(),
                    'Event_Rate_Weighted': (gw['event']*gw['weight']).sum()/gw['weight'].sum()*100,
                    'Event_Rate_Unweighted': group['event'].mean() * 100
                })

            region_summary = pd.DataFrame(region_groups)
            region_summary = region_summary.sort_values('Median_Duration', ascending=False)
        else:
            region_summary = data.groupby('region').agg({
                'duration_months': ['count', 'mean', 'median'],
                'event': ['sum', 'mean']
            }).round(2)
            region_summary.columns = ['N', 'Mean_Duration', 'Median_Duration', 'N_Events', 'Event_Rate']
            region_summary = region_summary.sort_values('Median_Duration', ascending=False)

        region_summary.to_excel(writer, sheet_name='By_Region', index=False)

        # Urban vs Rural (if available)
        if 'urban' in data.columns:
            if 'weight' in data.columns:
                urban_groups = []
                for urban_status, group in data.groupby('urban'):
                    w = group['weight']
                    valid = w.notna() & (w > 0)
                    if valid.sum() == 0:
                        continue
                    gw = group.loc[valid]

                    urban_groups.append({
                        'Residence': 'Urban' if urban_status == 1 else 'Rural',
                        'N': len(group),
                        'Mean_Duration_Weighted': np.average(gw['duration_months'], weights=gw['weight']),
                        'Mean_Duration_Unweighted': group['duration_months'].mean(),
                        'Median_Duration': group['duration_months'].median(),
                        'N_Events': group['event'].sum(),
                        'Event_Rate_Weighted': (gw['event']*gw['weight']).sum()/gw['weight'].sum()*100,
                        'Event_Rate_Unweighted': group['event'].mean() * 100
                    })

                urban_summary = pd.DataFrame(urban_groups)
            else:
                urban_summary = data.groupby('urban').agg({
                    'duration_months': ['count', 'mean', 'median'],
                    'event': ['sum', 'mean']
                }).round(2)
                urban_summary.index = ['Rural', 'Urban']
                urban_summary.columns = ['N', 'Mean_Duration', 'Median_Duration', 'N_Events', 'Event_Rate']

            urban_summary.to_excel(writer, sheet_name='Urban_vs_Rural')

    logger.info("Summary report saved to survival_results.xlsx")

# -------------------- Main Function --------------------

def main(data_file: str, output_dir: str):
    """Main analysis function."""

    # Setup paths
    data_file = Path(data_file)
    output_dir = Path(output_dir)
    dirs = create_output_dirs(output_dir)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("WEIGHTED SURVIVAL ANALYSIS OF BREASTFEEDING DURATION")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {data_file}")
    data = pd.read_csv(data_file)
    logger.info(f"Loaded {len(data):,} observations from {data['country'].nunique()} countries")

    # Clean data - remove any rows with null countries
    data = data[data['country'].notna()]

    # Add survey weights
    if 'v005' in data.columns:
        data['weight'] = pd.to_numeric(data['v005'], errors='coerce') / 1e6
        logger.info("Survey weights (v005/1e6) added for weighted analysis")
    else:
        logger.warning("No v005 variable found - proceeding with unweighted analysis")

    # Add region variable
    data['region'] = data['country'].apply(assign_region)

    # 1. Kaplan-Meier curves by country (using default top_n=12)
    logger.info("\nCreating weighted Kaplan-Meier curves by country...")
    country_medians = kaplan_meier_by_group(
        data, 'country', dirs['curves'], 'Country'
    )

    # 2. Kaplan-Meier curves by region
    logger.info("Creating weighted Kaplan-Meier curves by region...")
    region_medians = kaplan_meier_by_group(
        data, 'region', dirs['curves'], 'Region'
    )

    # 3. Urban vs Rural (if available)
    if 'urban' in data.columns:
        logger.info("Creating weighted urban vs rural comparison...")
        data['residence'] = data['urban'].map({0: 'Rural', 1: 'Urban'})
        urban_medians = kaplan_meier_by_group(
            data, 'residence', dirs['curves'], 'Urban/Rural Residence'
        )

    # 4. Cox regression
    if 'weight' in data.columns and 'survey_year' in data.columns:
        cox_model = cox_regression_analysis(data, dirs['cox'], logger)
    else:
        logger.warning("Skipping Cox regression - requires weights and survey_year")
        cox_model = None

    # 5. Early weaning analysis
    logger.info("\nAnalyzing early weaning patterns...")
    early_weaning_results = analyze_early_weaning(data, dirs['tables'])

    # 6. Create summary report
    create_summary_report(data, dirs['tables'], logger)

    # 7. Log-rank tests for key comparisons (check for events first)
    logger.info("\nPerforming log-rank tests...")

    if 'urban' in data.columns:
        # Urban vs Rural
        urban_data = data[data['urban'] == 1]
        rural_data = data[data['urban'] == 0]

        # Check both groups have events
        if (len(urban_data) > 30 and len(rural_data) > 30 and
            urban_data['event'].sum() > 0 and rural_data['event'].sum() > 0):
            lr_result = logrank_test(
                urban_data['duration_months'], rural_data['duration_months'],
                urban_data['event'], rural_data['event']
            )
            logger.info(f"Urban vs Rural log-rank test: p-value = {lr_result.p_value:.4f}")
        else:
            logger.warning("Insufficient data or no events for urban/rural comparison")

    # Regional comparison
    if data['region'].nunique() > 1:
        groups = []
        for region in data['region'].unique():
            if pd.notna(region):
                region_data = data[data['region'] == region]
                # Check region has events and sufficient data
                if len(region_data) > 30 and region_data['event'].sum() > 0:
                    groups.append(region)

        if len(groups) > 1:
            mlr_result = multivariate_logrank_test(
                data[data['region'].isin(groups)]['duration_months'],
                data[data['region'].isin(groups)]['region'],
                data[data['region'].isin(groups)]['event']
            )
            logger.info(f"Regional comparison log-rank test: p-value = {mlr_result.p_value:.4f}")
        else:
            logger.warning("Insufficient regions with events for comparison")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Output saved to: {output_dir.absolute()}")
    logger.info(f"Key findings:")

    if country_medians:
        valid_medians = [v for v in country_medians.values() if not np.isnan(v) and not np.isinf(v)]
        if valid_medians:
            logger.info(f"  - Median BF duration range: {min(valid_medians):.1f} - {max(valid_medians):.1f} months")

    logger.info(f"  - Countries with data: {data['country'].nunique()}")

    if 'weight' in data.columns:
        w = data['weight']
        valid = w.notna() & (w > 0)
        dw = data.loc[valid]
        early_pct = (dw[(dw['event']==1) & (dw['duration_months']<6)]['weight'].sum() /
                    dw['weight'].sum() * 100) if dw['weight'].sum() > 0 else 0
        logger.info(f"  - Early weaning (<6mo): {early_pct:.1f}% (weighted)")
    else:
        early_pct = (data[(data['event']==1) & (data['duration_months']<6)].shape[0] / len(data) * 100)
        logger.info(f"  - Early weaning (<6mo): {early_pct:.1f}% (unweighted)")

    logger.info("=" * 60)
    logger.info(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform weighted survival analysis on breastfeeding duration data"
    )
    parser.add_argument("--data",
                       default="./02_extract_breastfeeding_data/combined_breastfeeding_data.csv",
                       help="Path to combined breastfeeding data CSV")
    parser.add_argument("--out",
                       default="./03_survival_analysis",
                       help="Output directory for results")

    args = parser.parse_args()
    main(args.data, args.out)