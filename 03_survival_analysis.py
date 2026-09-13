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

# Publication figure settings (R1.5): larger fonts, 300 dpi kept in every savefig
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 15,
})

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
    fig, ax = plt.subplots(figsize=(9, 5.5))

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
            if len(g) < 30 or g['weight'].sum() <= 0:
                continue
        label = f"{group} (n={len(g):,})"

        # Children coded m4 = 93 stopped breastfeeding at an unknown time before the
        # interview (left-censored). An ordinary Kaplan-Meier estimator cannot use
        # them, so where they occur we fit the Turnbull NPMLE from the interval
        # bounds [t_lower, t_upper] instead.
        n_left = int((g['censor_type'] == 'left').sum()) if 'censor_type' in g.columns else 0
        if n_left > 0 and {'t_lower', 't_upper'}.issubset(g.columns):
            lo = pd.to_numeric(g['t_lower'], errors='coerce').to_numpy(dtype=float)
            up = pd.to_numeric(g['t_upper'], errors='coerce').to_numpy(dtype=float)
            up = np.where(np.isnan(up), np.inf, up)
            if 'weight' in g.columns:
                kmf.fit_interval_censoring(lo, up, weights=g['weight'].to_numpy(dtype=float),
                                           label=label)
            else:
                kmf.fit_interval_censoring(lo, up, label=label)
        else:
            if g['event'].sum() == 0:
                continue
            if 'weight' in g.columns:
                kmf.fit(g['duration_months'], event_observed=g['event'], weights=g['weight'],
                        label=label)
            else:
                kmf.fit(g['duration_months'], event_observed=g['event'], label=label)

        # Get color from colormap
        color = cmap(i % 20)  # tab20 has 20 colors
        sf = kmf.survival_function_
        ax.step(sf.index.values, sf.iloc[:, 0].values, where='post',
                color=color, linewidth=2, label=label)
        med = kmf.median_survival_time_
        if isinstance(med, (pd.DataFrame, pd.Series)):   # interval-censored: bounds
            med = float(np.asarray(med, dtype=float).ravel().mean())
        medians[group] = float(med)

    ax.set_xlabel('Time (months)', fontsize=13)
    ax.set_ylabel('Probability of Still Breastfeeding', fontsize=13)
    ax.set_title(f'Probability of Continuing Any Breastfeeding by {title_suffix} (Weighted)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0, 36)
    ax.grid(True, alpha=0.25)

    # 6-month reference line (R2.13: this is the WHO exclusive-breastfeeding recommendation, not a minimum duration of any breastfeeding)
    ax.axvline(x=6, color='#CC3344', linestyle='--', alpha=0.6, label='6 months (WHO exclusive\nbreastfeeding recommendation)')

    # lean legend outside
    leg = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10, borderaxespad=0.)
    
    # FIXED: Use 'legend_handles' instead of 'legendHandles'
    for lh in leg.legend_handles:
        lh.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(output_dir / f"km_by_{group_var.replace('/', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    return medians

# Human-readable covariate labels for the forest plot (wording follows the manuscript).
# Rows with var=None are group headers naming the reference category.
FOREST_ROWS = [
    ('Residence (reference: rural)', None),
    ('    Urban residence', 'urban'),
    ('Household wealth (reference: middle quintile, Q3)', None),
    ('    Poorest quintile (Q1)', 'wealth_q1'),
    ('    Second quintile (Q2)', 'wealth_q2'),
    ('    Fourth quintile (Q4)', 'wealth_q4'),
    ('    Richest quintile (Q5)', 'wealth_q5'),
    ('Maternal education (reference: primary)', None),
    ('    No formal education', 'educ_none'),
    ('    Secondary or higher', 'educ_secondary_plus'),
]

def plot_forest(results_df: pd.DataFrame, output_dir: Path):
    """Forest plot of hazard ratios (log scale) with 95% CIs and numeric annotations."""
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter

    hr = results_df['exp(coef)']
    lo = results_df['exp(coef) lower 95%']
    hi = results_df['exp(coef) upper 95%']

    rows = [(lab, var) for lab, var in FOREST_ROWS if var is None or var in results_df.index]
    mapped = {var for _, var in FOREST_ROWS if var is not None}
    for var in results_df.index:            # any covariate not in the mapping still gets drawn
        if var not in mapped:
            rows.append((f'    {var}', var))

    n = len(rows)
    ys = np.arange(n)[::-1]                  # first row at the top
    fig, ax = plt.subplots(figsize=(9, 0.5 * n + 1.2))

    xmax_data = float(hi.max())
    x_text = xmax_data * 1.06
    for y, (lab, var) in zip(ys, rows):
        if var is None:
            continue
        ax.errorbar(hr[var], y, xerr=[[hr[var] - lo[var]], [hi[var] - hr[var]]],
                    fmt='s', color='black', markersize=6, capsize=3, linewidth=1.2)
        ax.text(x_text, y, f"{hr[var]:.2f} ({lo[var]:.2f}\u2013{hi[var]:.2f})",
                va='center', ha='left', fontsize=11)

    ax.axvline(1.0, color='grey', linestyle='--', linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in rows])
    for tick, (lab, var) in zip(ax.get_yticklabels(), rows):
        if var is None:
            tick.set_fontweight('bold')
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_xscale('log')
    xmin = float(lo.min()) * 0.95
    xmax = xmax_data * 1.45
    ax.set_xlim(xmin, xmax)
    ticks = [t for t in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5] if xmin <= t <= xmax_data * 1.08]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlabel('Hazard ratio for cessation of any breastfeeding (95% CI), log scale', fontsize=13)
    ax.set_title('Adjusted Hazard Ratios from Survey-Weighted, Stratified Cox Models',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'hazard_ratios_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

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

        # Create forest plot (manuscript wording, HR on log scale)
        plot_forest(results_df, output_dir)

        logger.info(f"Cox regression completed. C-index: {cph.concordance_index_:.3f}")

        return cph

    except Exception as e:
        logger.error(f"Cox regression failed: {e}")
        return None

def _survival_at(g: pd.DataFrame, t: float):
    """Weighted probability of still breastfeeding at t months.

    Uses the Turnbull NPMLE where the group contains children coded m4 = 93
    (stopped before the interview, duration unrecorded, i.e. left-censored) and
    the ordinary Kaplan-Meier estimator otherwise. Returns None if neither can
    be fitted.
    """
    kmf = KaplanMeierFitter()
    w = g['weight'].to_numpy(dtype=float) if 'weight' in g.columns else None
    n_left = int((g['censor_type'] == 'left').sum()) if 'censor_type' in g.columns else 0
    try:
        if n_left > 0 and {'t_lower', 't_upper'}.issubset(g.columns):
            lo = pd.to_numeric(g['t_lower'], errors='coerce').to_numpy(dtype=float)
            up = pd.to_numeric(g['t_upper'], errors='coerce').to_numpy(dtype=float)
            up = np.where(np.isnan(up), np.inf, up)
            kmf.fit_interval_censoring(lo, up, weights=w)
        else:
            if g['event'].sum() == 0:
                return None
            kmf.fit(g['duration_months'], event_observed=g['event'], weights=w)
    except Exception:
        return None
    sf = kmf.survival_function_
    idx = sf.index.get_indexer([t], method='ffill')[0]
    return float(sf.iloc[idx, 0]) if idx >= 0 else 1.0


def analyze_early_weaning(data: pd.DataFrame, output_dir: Path):
    """Prevalence of cessation of any breastfeeding before 6 months, by country.

    Estimated as 1 - S(6) from the weighted survival curve rather than by counting
    children whose recorded duration is under 6 months. Counting understates the
    prevalence: children still under 6 months at interview cannot yet have a
    recorded duration of 6 months, and children coded m4 = 93 have no recorded
    duration at all.
    """

    results = []

    for country in data['country'].unique():
        country_data = data[data['country'] == country].copy()

        if 'weight' in country_data.columns:
            w = country_data['weight']
            country_data = country_data.loc[w.notna() & (w > 0)]
        if len(country_data) == 0:
            continue

        s6 = _survival_at(country_data, 6)
        if s6 is None:
            continue
        pct_weaned_before_6m = 100.0 * (1.0 - s6)

        # crude count kept for comparison with the previously published figures
        cw = country_data
        ew = cw[(cw['event'] == 1) & (cw['duration_months'] < 6)]
        crude = (100.0 * ew['weight'].sum() / cw['weight'].sum()
                 if 'weight' in cw.columns and cw['weight'].sum() > 0 else np.nan)

        results.append({
            'country': country,
            'n_children': len(country_data),
            'n_left_censored': int((country_data['censor_type'] == 'left').sum())
                               if 'censor_type' in country_data.columns else 0,
            'n_weaned_before_6m': len(ew),
            'pct_weaned_before_6m': pct_weaned_before_6m,
            'pct_weaned_before_6m_crude_count': crude,
            'median_early_weaning_age': ew['duration_months'].median()
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pct_weaned_before_6m', ascending=False)
    results_df.to_excel(output_dir / 'early_weaning_by_country.xlsx', index=False)

    # Create bar plot of top 20 countries with highest early weaning
    top20 = results_df.head(20)

    if len(top20) > 0:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Create gradient colors from dark red (highest) to light yellow (lowest)
        # Using a red-orange-yellow gradient for better visual impact
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(top20))[::-1])
        
        bars = ax.bar(range(len(top20)), top20['pct_weaned_before_6m'], color=colors)
        ax.set_xticks(range(len(top20)))
        ax.set_xticklabels(top20['country'], rotation=45, ha='right')
        ax.set_ylabel('% Ceased Any Breastfeeding Before 6 Months (Weighted)', fontsize=13)
        ax.set_title('20 Countries with the Highest Prevalence of Cessation\nof Any Breastfeeding Before 6 Months',
                    fontsize=13, fontweight='bold')
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
                'Median of recorded duration column (NOT a survival median)': data['duration_months'].median()
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
                'Median of recorded duration column (NOT a survival median)': data['duration_months'].median()
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

def main(data_file: str, output_dir: str, skip_cox: bool = False, cox_results: str = None):
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
    data = pd.read_csv(data_file, low_memory=False)
    logger.info(f"Loaded {len(data):,} observations from {data['country'].nunique()} countries")

    # Clean data - remove any rows with null countries
    data = data[data['country'].notna()]

    if 'censor_type' in data.columns:
        vc = data['censor_type'].value_counts()
        logger.info(f"Censoring mix: exact {int(vc.get('exact', 0)):,}, "
                    f"right {int(vc.get('right', 0)):,}, left (m4=93) {int(vc.get('left', 0)):,}")
        logger.info("Curves and medians use the Turnbull estimator wherever left-censored "
                    "children are present, and Kaplan-Meier otherwise")
    else:
        logger.warning("No censor_type column: this looks like an extraction from before v2.3, "
                       "in which children coded m4 = 93 were dropped")

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
    saved_cox = Path(cox_results) if cox_results else dirs['cox'] / 'cox_regression_results.xlsx'
    if cox_results:
        logger.info(f"--cox-results: drawing the forest plot from {saved_cox} (no refit)")
        cox_tbl = pd.read_excel(saved_cox, index_col=0)
        plot_forest(cox_tbl, dirs['cox'])
        cox_tbl.to_excel(dirs['cox'] / 'cox_regression_results.xlsx')
        cox_model = None
    elif skip_cox and saved_cox.exists():
        logger.info("--skip-cox: re-drawing the forest plot from saved cox_regression_results.xlsx (no refit)")
        plot_forest(pd.read_excel(saved_cox, index_col=0), dirs['cox'])
        cox_model = None
    elif 'weight' in data.columns and 'survey_year' in data.columns:
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
        s6 = _survival_at(dw, 6)
        s12 = _survival_at(dw, 12)
        s24 = _survival_at(dw, 24)
        if s6 is not None:
            logger.info(f"  - Ceased any breastfeeding by 6 months:  {100*(1-s6):.1f}% (weighted, 1 - S(6))")
        if s12 is not None:
            logger.info(f"  - Ceased any breastfeeding by 12 months: {100*(1-s12):.1f}%")
        if s24 is not None:
            logger.info(f"  - Ceased any breastfeeding by 24 months: {100*(1-s24):.1f}%")
        crude = (dw[(dw['event']==1) & (dw['duration_months']<6)]['weight'].sum() /
                 dw['weight'].sum() * 100) if dw['weight'].sum() > 0 else 0
        logger.info(f"  - (crude count of recorded durations < 6 months: {crude:.1f}%, "
                    f"understates the prevalence)")
    else:
        early_pct = (data[(data['event']==1) & (data['duration_months']<6)].shape[0] / len(data) * 100)
        logger.info(f"  - Early weaning (<6mo): {early_pct:.1f}% (unweighted crude count)")

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
    parser.add_argument("--cox-results", default=None,
                        help="path to cox results from r26_cox_exact.py; draws the forest plot "
                             "from that fit instead of refitting here")
    parser.add_argument("--skip-cox", action="store_true",
                       help="Do not refit the Cox model; redraw the forest plot from the saved results file")

    args = parser.parse_args()
    main(args.data, args.out, args.skip_cox, args.cox_results)