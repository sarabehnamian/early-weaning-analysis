# 04_comparative_analysis.py
# python 04_comparative_analysis.py --data ".\02_extract_breastfeeding_data\combined_breastfeeding_data.csv" --out ".\04_comparative_analysis"

# Performs comparative geographic analysis and creates publication-ready outputs
#
# Outputs (in .\04_comparative_analysis\):
#   - rankings\: Country and regional rankings
#   - inequalities\: Within-country inequality analyses  
#   - trends\: Temporal trend analysis if multiple surveys available
#   - maps\: Geographic visualizations
#   - publication_tables\: Tables formatted for papers
#   - final_report.pdf: Complete analysis report
#
# Requirements: pandas, numpy, matplotlib, geopandas (optional for maps)

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Analysis libraries
from scipy import stats
from lifelines import KaplanMeierFitter

# Plotting
import matplotlib.pyplot as plt
plt.style.use('default')  # Use default style

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

REGIONS = {
    # Identical to 03_survival_analysis.py so that regional Ns agree across all figures (R3.9)
    'West Africa': ['BF', 'BJ', 'CI', 'GH', 'GM', 'GN', 'LB', 'ML', 'NI', 'NG', 'SL', 'SN', 'TG', 'MR'],
    'East Africa': ['BI', 'BU', 'ER', 'ET', 'KE', 'KM', 'MW', 'RW', 'TZ', 'UG', 'ZM', 'ZW', 'MZ', 'LS', 'MG'],
    'Central Africa': ['AO', 'CD', 'CG', 'CM', 'GA', 'ST', 'TD', 'CF'],
    'North Africa': ['EG', 'MA', 'TN'],
    'Southern Africa': ['ZA', 'SZ', 'NM'],
    'South Asia': ['AF', 'BD', 'IA', 'IN', 'LK', 'MV', 'NP', 'PK',
                   'AP', 'AS', 'BH', 'DL', 'GJ', 'GO', 'HP', 'HR', 'KA',
                   'MB', 'MH', 'MN', 'MP', 'OR', 'PJ', 'RJ', 'SK', 'UP', 'WB'],
    'Southeast Asia': ['ID', 'KH', 'LA', 'MM', 'PH', 'TH', 'TL', 'VN', 'PG'],
    'Latin America & Caribbean': ['BO', 'BR', 'CO', 'DR', 'EC', 'GT', 'GU', 'HN', 'HT',
                                  'MX', 'NC', 'PE', 'PY', 'AR', 'GY', 'JM'],
    'Europe/Central Asia': ['AL', 'AM', 'AZ', 'KK', 'KY', 'MD', 'TJ', 'TR', 'UA', 'UZ'],
    'Middle East': ['JO', 'YE'],
}

# Country name mapping for better labels
COUNTRY_NAMES = {
    'ET': 'Ethiopia', 'KE': 'Kenya', 'TZ': 'Tanzania', 'UG': 'Uganda', 'RW': 'Rwanda',
    'MW': 'Malawi', 'ZM': 'Zambia', 'ZW': 'Zimbabwe', 'MZ': 'Mozambique', 'LS': 'Lesotho',
    'NG': 'Nigeria', 'GH': 'Ghana', 'SN': 'Senegal', 'ML': 'Mali', 'BF': 'Burkina Faso',
    'IN': 'India', 'PK': 'Pakistan', 'BD': 'Bangladesh', 'NP': 'Nepal', 'AF': 'Afghanistan',
    'ID': 'Indonesia', 'PH': 'Philippines', 'KH': 'Cambodia', 'MM': 'Myanmar', 'VN': 'Vietnam',
    'BO': 'Bolivia', 'PE': 'Peru', 'GT': 'Guatemala', 'HT': 'Haiti', 'HN': 'Honduras',
    'EG': 'Egypt', 'MA': 'Morocco', 'JO': 'Jordan', 'AM': 'Armenia', 'AZ': 'Azerbaijan',
    'BJ': 'Benin', 'CI': "Côte d'Ivoire", 'GM': 'Gambia', 'GN': 'Guinea', 'LB': 'Liberia',
    'NE': 'Niger', 'SL': 'Sierra Leone', 'TG': 'Togo', 'MR': 'Mauritania',
    'BI': 'Burundi', 'BU': 'Burundi', 'ER': 'Eritrea', 'KM': 'Comoros', 'MG': 'Madagascar',
    'AO': 'Angola', 'CD': 'DR Congo', 'CG': 'Congo', 'CM': 'Cameroon', 'GA': 'Gabon',
    'ST': 'São Tomé', 'TD': 'Chad', 'CF': 'Central African Rep',
    'ZA': 'South Africa', 'SZ': 'Eswatini', 'NM': 'Namibia',
    'LK': 'Sri Lanka', 'MV': 'Maldives', 'IA': 'India', 
    'LA': 'Laos', 'TH': 'Thailand', 'TL': 'Timor-Leste', 'PG': 'Papua New Guinea',
    'BR': 'Brazil', 'CO': 'Colombia', 'DR': 'Dominican Rep', 'EC': 'Ecuador',
    'MX': 'Mexico', 'NC': 'Nicaragua', 'NI': 'Niger', 'GU': 'Guatemala', 'PY': 'Paraguay',
    'AR': 'Argentina', 'GY': 'Guyana', 'JM': 'Jamaica',
    'AL': 'Albania', 'KK': 'Kazakhstan', 'KY': 'Kyrgyzstan', 'MD': 'Moldova',
    'TJ': 'Tajikistan', 'TR': 'Turkey', 'UA': 'Ukraine', 'UZ': 'Uzbekistan',
    'YE': 'Yemen', 'TN': 'Tunisia',
    # Indian state-level DHS files (NFHS)
    'AP': 'Andhra Pradesh, India', 'AS': 'Assam, India', 'BH': 'Bihar, India', 'DL': 'Delhi, India',
    'GJ': 'Gujarat, India', 'GO': 'Goa, India', 'HP': 'Himachal Pradesh, India', 'HR': 'Haryana, India',
    'KA': 'Karnataka, India', 'MB': 'Meghalaya, India', 'MH': 'Maharashtra, India', 'MN': 'Manipur, India',
    'MP': 'Madhya Pradesh, India', 'OR': 'Odisha, India', 'PJ': 'Punjab, India', 'RJ': 'Rajasthan, India',
    'SK': 'Sikkim, India', 'UP': 'Uttar Pradesh, India', 'WB': 'West Bengal, India'
}

# -------------------- Helper Functions --------------------

def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / "comparative_analysis_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
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

def get_country_name(code: str) -> str:
    """Get full country name from code."""
    return COUNTRY_NAMES.get(code, code)

def _fit_survival(data: pd.DataFrame):
    """Weighted survival fit.

    Children coded m4 = 93 stopped breastfeeding at an unknown time before the
    interview (left-censored); an ordinary Kaplan-Meier estimator cannot use them,
    so where they are present the Turnbull NPMLE is fitted from the interval
    bounds [t_lower, t_upper]. Returns a fitted KaplanMeierFitter, or None.
    """
    if len(data) < 10:
        return None
    d = data
    w = None
    if 'weight' in d.columns:
        valid = d['weight'].notna() & (d['weight'] > 0)
        d = d.loc[valid]
        if len(d) < 10:
            return None
        w = d['weight'].to_numpy(dtype=float)
    kmf = KaplanMeierFitter()
    n_left = int((d['censor_type'] == 'left').sum()) if 'censor_type' in d.columns else 0
    try:
        if n_left > 0 and {'t_lower', 't_upper'}.issubset(d.columns):
            lo = pd.to_numeric(d['t_lower'], errors='coerce').to_numpy(dtype=float)
            up = pd.to_numeric(d['t_upper'], errors='coerce').to_numpy(dtype=float)
            up = np.where(np.isnan(up), np.inf, up)
            kmf.fit_interval_censoring(lo, up, weights=w)
        else:
            if d['event'].sum() == 0:
                return None
            kmf.fit(d['duration_months'], d['event'], weights=w)
    except Exception:
        return None
    return kmf


def survival_at(data: pd.DataFrame, t: float) -> float:
    """Weighted probability of still breastfeeding at t months (np.nan if not estimable)."""
    kmf = _fit_survival(data)
    if kmf is None:
        return np.nan
    sf = kmf.survival_function_
    idx = sf.index.get_indexer([t], method='ffill')[0]
    return float(sf.iloc[idx, 0]) if idx >= 0 else 1.0


def calculate_median_duration(data: pd.DataFrame) -> float:
    """Median duration of any breastfeeding (Turnbull where left-censored, else Kaplan-Meier)."""
    kmf = _fit_survival(data)
    if kmf is None:
        return np.nan
    median = kmf.median_survival_time_
    if isinstance(median, (pd.DataFrame, pd.Series)):   # interval-censored: bounds
        median = float(np.asarray(median, dtype=float).ravel().mean())
    median = float(median)
    # Cap infinite values at 36 months for better visualization
    if np.isinf(median):
        median = 36.0
    return median

# -------------------- Analysis Functions --------------------

def create_country_rankings(data: pd.DataFrame, output_dir: Path, logger):
    """Create comprehensive country rankings."""
    
    logger.info("Creating country rankings...")
    
    rankings = []
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        
        # Skip if no data for this country
        if len(country_data) == 0:
            continue
        
        # Calculate key metrics
        median_duration = calculate_median_duration(country_data)
        
        # Use weights if available
        if 'weight' in country_data.columns:
            weights = country_data['weight']
            valid_weights = weights.notna() & (weights > 0)
            
            if valid_weights.sum() > 0:
                weighted_data = country_data[valid_weights]
                wd = weighted_data  # alias for clarity
                total_weight = wd['weight'].sum()
                
                # FIXED: Calculate weighted percentages directly on weighted_data
                # never breastfed is m4 = 94 where the flag is available; duration 0 also
                # counts children who breastfed for less than a month
                if 'never_breastfed' in wd.columns:
                    w_never = wd.loc[wd['never_breastfed'] == 1, 'weight'].sum()
                else:
                    w_never = wd.loc[(wd['duration_months'] == 0) & (wd['event'] == 1), 'weight'].sum()
                w_early = wd.loc[(wd['event'] == 1) & (wd['duration_months'] < 6) & (wd['duration_months'] > 0), 'weight'].sum()
                w_very = wd.loc[(wd['event'] == 1) & (wd['duration_months'] < 3) & (wd['duration_months'] > 0), 'weight'].sum()
                w_still_12 = wd.loc[wd['duration_months'] >= 12, 'weight'].sum()
                w_still_24 = wd.loc[wd['duration_months'] >= 24, 'weight'].sum()
                
                pct_never_bf = 100 * w_never / total_weight if total_weight > 0 else 0
                # Cessation and continuation come from the survival curve, not from counting
                # recorded durations: a child interviewed at 3 months cannot yet have a
                # recorded duration of 6 months, and children coded m4 = 93 have no
                # recorded duration at all.
                s3, s6, s12, s24 = (survival_at(wd, 3), survival_at(wd, 6),
                                    survival_at(wd, 12), survival_at(wd, 24))
                pct_weaned_before_6mo = 100 * (1 - s6) if not np.isnan(s6) else np.nan
                pct_weaned_before_3mo = 100 * (1 - s3) if not np.isnan(s3) else np.nan
                pct_weaned_0_6mo_excl_never = (pct_weaned_before_6mo - pct_never_bf
                                               if not np.isnan(pct_weaned_before_6mo) else np.nan)
                pct_still_bf_at_12mo = 100 * s12 if not np.isnan(s12) else np.nan
                pct_still_bf_at_24mo = 100 * s24 if not np.isnan(s24) else np.nan
                # crude counts kept for comparison with the previously published figures
                crude_before_6mo = 100 * (w_never + w_early) / total_weight if total_weight > 0 else 0
                crude_still_24 = 100 * w_still_24 / total_weight if total_weight > 0 else 0
            else:
                # Unweighted fallback
                never_bf = (country_data['duration_months'] == 0) & (country_data['event'] == 1)
                pct_never_bf = 100 * never_bf.sum() / len(country_data)
                
                early_weaned = (country_data['event'] == 1) & (country_data['duration_months'] < 6)
                very_early_weaned = (country_data['event'] == 1) & (country_data['duration_months'] < 3)
                early_excl_never = (country_data['event'] == 1) & (country_data['duration_months'] < 6) & (country_data['duration_months'] > 0)
                
                pct_weaned_0_6mo_excl_never = 100 * early_excl_never.sum() / len(country_data)
                pct_weaned_before_6mo = 100 * early_weaned.sum() / len(country_data)
                pct_weaned_before_3mo = 100 * very_early_weaned.sum() / len(country_data)
                pct_still_bf_at_12mo = 100 * (country_data['duration_months'] >= 12).sum() / len(country_data)
                pct_still_bf_at_24mo = 100 * (country_data['duration_months'] >= 24).sum() / len(country_data)
        else:
            # Unweighted calculations
            never_bf = (country_data['duration_months'] == 0) & (country_data['event'] == 1)
            pct_never_bf = 100 * never_bf.sum() / len(country_data)
            
            early_weaned = (country_data['event'] == 1) & (country_data['duration_months'] < 6)
            very_early_weaned = (country_data['event'] == 1) & (country_data['duration_months'] < 3)
            early_excl_never = (country_data['event'] == 1) & (country_data['duration_months'] < 6) & (country_data['duration_months'] > 0)
            
            pct_weaned_0_6mo_excl_never = 100 * early_excl_never.sum() / len(country_data)
            pct_weaned_before_6mo = 100 * early_weaned.sum() / len(country_data)
            pct_weaned_before_3mo = 100 * very_early_weaned.sum() / len(country_data)
            pct_still_bf_at_12mo = 100 * (country_data['duration_months'] >= 12).sum() / len(country_data)
            pct_still_bf_at_24mo = 100 * (country_data['duration_months'] >= 24).sum() / len(country_data)
        
        rankings.append({
            'country_code': country,
            'country_name': get_country_name(country),
            'region': assign_region(country),
            'n_children': len(country_data),
            'median_duration_months': median_duration,
            'mean_duration_months': country_data['duration_months'].mean(),
            'pct_never_breastfed': pct_never_bf,
            'pct_weaned_0_6mo_excl_never': pct_weaned_0_6mo_excl_never,
            'pct_weaned_before_6mo': pct_weaned_before_6mo,
            'pct_weaned_before_3mo': pct_weaned_before_3mo,
            'pct_still_bf_at_12mo': pct_still_bf_at_12mo,
            'pct_still_bf_at_24mo': pct_still_bf_at_24mo,
            'n_left_censored': int((country_data['censor_type'] == 'left').sum())
                               if 'censor_type' in country_data.columns else 0,
            'pct_weaned_before_6mo_crude_count': locals().get('crude_before_6mo', np.nan),
            'pct_still_bf_at_24mo_crude_count': locals().get('crude_still_24', np.nan)
        })
    
    rankings_df = pd.DataFrame(rankings)
    
    # Remove rows with NaN median duration for ranking
    rankings_df = rankings_df.dropna(subset=['median_duration_months'])
    
    # Add rankings
    rankings_df['rank_median'] = rankings_df['median_duration_months'].rank(ascending=False, method='min')
    rankings_df['rank_early_weaning'] = rankings_df['pct_weaned_before_6mo'].rank(ascending=True, method='min')
    rankings_df['rank_never_bf'] = rankings_df['pct_never_breastfed'].rank(ascending=True, method='min')
    
    # Sort by median duration
    rankings_df = rankings_df.sort_values('median_duration_months', ascending=False)
    
    # Save full rankings
    rankings_df.to_excel(output_dir / 'country_rankings_complete.xlsx', index=False)
    
    # Create visualization of top and bottom performers
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5))
    
    # Top 15 performers with gradient colors
    top15 = rankings_df.head(15)
    if len(top15) > 0:
        # Create gradient from dark green (best) to light green
        colors_top = plt.cm.Greens(np.linspace(0.95, 0.4, len(top15)))
        
        bars = ax1.barh(range(len(top15)), top15['median_duration_months'], color=colors_top)
        ax1.set_yticks(range(len(top15)))
        ax1.set_yticklabels([f"{row['country_name']} ({row['country_code']})" 
                             for _, row in top15.iterrows()])
        ax1.set_xlabel('Median duration of any breastfeeding (months)', fontsize=13)
        ax1.set_title('(A) 15 countries with the longest median duration', loc='left', fontsize=13, fontweight='bold')
        ax1.axvline(x=6, color='red', linestyle='--', alpha=0.6, label='6 months (WHO exclusive breastfeeding recommendation)')
        ax1.axvline(x=24, color='blue', linestyle='--', alpha=0.4, label='24 months (WHO continued breastfeeding recommendation)')
        ax1.set_xlim(0, 40)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
    
    # Bottom 15 performers with gradient colors
    bottom15 = rankings_df.tail(15) if len(rankings_df) >= 15 else rankings_df.tail(len(rankings_df))
    if len(bottom15) > 0:
        # Create gradient from dark red (worst) to light yellow
        colors_bottom = plt.cm.YlOrRd(np.linspace(0.95, 0.3, len(bottom15)))
        
        bars = ax2.barh(range(len(bottom15)), bottom15['median_duration_months'], color=colors_bottom)
        ax2.set_yticks(range(len(bottom15)))
        ax2.set_yticklabels([f"{row['country_name']} ({row['country_code']})" 
                             for _, row in bottom15.iterrows()])
        ax2.set_xlabel('Median duration of any breastfeeding (months)', fontsize=13)
        ax2.set_title('(B) 15 countries with the shortest median duration', loc='left', fontsize=13, fontweight='bold')
        ax2.axvline(x=6, color='red', linestyle='--', alpha=0.6)
        ax2.axvline(x=24, color='blue', linestyle='--', alpha=0.4)
        ax2.set_xlim(0, 40)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
    
    # One shared legend below both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('Country Rankings: Median Duration of Any Breastfeeding', fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(output_dir / 'country_rankings_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rankings_df

def analyze_never_breastfed(data: pd.DataFrame, output_dir: Path, logger):
    """Analyze never breastfed patterns as a distinct category."""
    
    logger.info("Analyzing never breastfed patterns...")
    
    # Identify never breastfed: m4 = 94 where the flag is available, otherwise fall
    # back to duration 0, which also counts children who breastfed under a month
    if 'never_breastfed' in data.columns:
        data['never_bf'] = data['never_breastfed'].fillna(0).astype(int)
        logger.info("  never-breastfed defined as m4 = 94")
    else:
        data['never_bf'] = ((data['duration_months'] == 0) & (data['event'] == 1)).astype(int)
        logger.warning("  no never_breastfed flag in this extraction: falling back to "
                       "duration 0, which also counts children who breastfed under a month")
    
    # Calculate by country
    country_never_bf = []
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        
        if 'weight' in country_data.columns:
            weights = country_data['weight']
            valid = weights.notna() & (weights > 0)
            if valid.sum() > 0:
                wd = country_data[valid]
                pct_never = 100 * (wd['never_bf'] * wd['weight']).sum() / wd['weight'].sum()
            else:
                pct_never = 100 * country_data['never_bf'].mean()
        else:
            pct_never = 100 * country_data['never_bf'].mean()
        
        country_never_bf.append({
            'country_code': country,
            'country_name': get_country_name(country),
            'region': assign_region(country),
            'n_children': len(country_data),
            'pct_never_breastfed': pct_never,
            'n_never_bf': country_data['never_bf'].sum()
        })
    
    df_never = pd.DataFrame(country_never_bf)
    df_never = df_never.sort_values('pct_never_breastfed', ascending=False)
    df_never.to_excel(output_dir / 'never_breastfed_analysis.xlsx', index=False)
    
    # Create visualization - Top 20 countries
    top20 = df_never.head(20)
    if len(top20) > 0:
        fig, ax = plt.subplots(figsize=(9, 7))
        # Darker colors for taller bars
        colors = plt.cm.Reds(np.linspace(0.9, 0.3, len(top20)))
        bars = ax.barh(range(len(top20)), top20['pct_never_breastfed'], color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"{row['country_name']} ({row['country_code']})" 
                            for _, row in top20.iterrows()])
        ax.set_xlabel('% of children never breastfed (weighted)', fontsize=13)
        ax.set_title('20 Countries with the Highest Prevalence of Never-Breastfed Children', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / 'never_breastfed_by_country.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Regional summary
    data['region'] = data['country'].apply(assign_region)
    regional_never = []
    for region in data['region'].unique():
        if region == 'Other':
            continue
        region_data = data[data['region'] == region]
        
        if 'weight' in region_data.columns:
            weights = region_data['weight']
            valid = weights.notna() & (weights > 0)
            if valid.sum() > 0:
                wd = region_data[valid]
                pct_never = 100 * (wd['never_bf'] * wd['weight']).sum() / wd['weight'].sum()
            else:
                pct_never = 100 * region_data['never_bf'].mean()
        else:
            pct_never = 100 * region_data['never_bf'].mean()
        
        regional_never.append({
            'region': region,
            'pct_never_bf': pct_never,
            'n_children': len(region_data),
            'n_countries': region_data['country'].nunique()
        })
    
    regional_never_df = pd.DataFrame(regional_never)
    regional_never_df = regional_never_df.sort_values('pct_never_bf', ascending=False)
    regional_never_df.to_excel(output_dir / 'never_breastfed_by_region.xlsx', index=False)
    
    # Create regional never-BF visualization
    if len(regional_never_df) > 0:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Darker colors for taller bars
        colors = plt.cm.YlOrRd(np.linspace(0.9, 0.3, len(regional_never_df)))
        bars = ax.bar(range(len(regional_never_df)), regional_never_df['pct_never_bf'], color=colors)
        ax.set_xticks(range(len(regional_never_df)))
        ax.set_xticklabels(regional_never_df['region'], rotation=45, ha='right')
        ax.set_ylabel('% of children never breastfed (weighted)', fontsize=13)
        ax.set_title('Never-Breastfed Prevalence by Region', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'never_breastfed_by_region.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return df_never, regional_never_df

def analyze_inequalities(data: pd.DataFrame, output_dir: Path, logger):
    """Analyze within-country inequalities."""
    
    logger.info("Analyzing within-country inequalities...")
    
    inequalities = []
    
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        
        # Skip if insufficient data
        if len(country_data) < 100:
            continue
        
        inequality_metrics = {
            'country_code': country,
            'country_name': get_country_name(country),
            'n_total': len(country_data)
        }
        
        # Urban-rural inequality
        if 'urban' in country_data.columns:
            urban_data = country_data[country_data['urban'] == 1]
            rural_data = country_data[country_data['urban'] == 0]
            
            if len(urban_data) > 30 and len(rural_data) > 30:
                urban_median = calculate_median_duration(urban_data)
                rural_median = calculate_median_duration(rural_data)
                
                inequality_metrics['urban_median'] = urban_median
                inequality_metrics['rural_median'] = rural_median
                inequality_metrics['urban_rural_gap'] = rural_median - urban_median if not pd.isna(rural_median) and not pd.isna(urban_median) else np.nan
                inequality_metrics['n_urban'] = len(urban_data)
                inequality_metrics['n_rural'] = len(rural_data)
        
        # Wealth inequality
        if 'wealth_quintile' in country_data.columns:
            poorest = country_data[country_data['wealth_quintile'] == 1]
            richest = country_data[country_data['wealth_quintile'] == 5]
            
            if len(poorest) > 30 and len(richest) > 30:
                poorest_median = calculate_median_duration(poorest)
                richest_median = calculate_median_duration(richest)
                
                inequality_metrics['poorest_median'] = poorest_median
                inequality_metrics['richest_median'] = richest_median
                inequality_metrics['wealth_gap'] = poorest_median - richest_median if not pd.isna(poorest_median) and not pd.isna(richest_median) else np.nan
                inequality_metrics['n_poorest'] = len(poorest)
                inequality_metrics['n_richest'] = len(richest)
        
        # Education inequality
        if 'mother_educ' in country_data.columns:
            no_educ = country_data[country_data['mother_educ'] == 0]
            secondary_plus = country_data[country_data['mother_educ'] >= 2]
            
            if len(no_educ) > 30 and len(secondary_plus) > 30:
                no_educ_median = calculate_median_duration(no_educ)
                educated_median = calculate_median_duration(secondary_plus)
                
                inequality_metrics['no_education_median'] = no_educ_median
                inequality_metrics['secondary_plus_median'] = educated_median
                inequality_metrics['education_gap'] = no_educ_median - educated_median if not pd.isna(no_educ_median) and not pd.isna(educated_median) else np.nan
        
        inequalities.append(inequality_metrics)
    
    inequality_df = pd.DataFrame(inequalities)
    
    # Save results
    if len(inequality_df) > 0:
        inequality_df.to_excel(output_dir / 'within_country_inequalities.xlsx', index=False)
    
    # Visualize inequalities
    if len(inequality_df) > 0 and 'urban_rural_gap' in inequality_df.columns:
        # Filter for only countries with valid urban AND rural data AND meaningful gaps
        plot_data = inequality_df[
            (inequality_df['urban_rural_gap'].notna()) & 
            (inequality_df['urban_median'].notna()) & 
            (inequality_df['rural_median'].notna()) &
            (~np.isinf(inequality_df['urban_median'])) &
            (~np.isinf(inequality_df['rural_median'])) &
            (inequality_df['n_urban'] >= 30) &  # Ensure adequate sample sizes
            (inequality_df['n_rural'] >= 30) &
            (np.abs(inequality_df['urban_rural_gap']) > 0.1)  # Remove near-zero gaps
        ].copy()
        
        # Cap extreme values for better visualization
        plot_data['urban_rural_gap'] = plot_data['urban_rural_gap'].clip(-15, 20)
        plot_data = plot_data.sort_values('urban_rural_gap')
        
        if len(plot_data) > 0:
            # Make figure height reasonable based on actual data
            n_countries = len(plot_data)
            fig_height = min(14, max(8, n_countries * 0.18))  # Adjust height based on actual countries
            fig, ax = plt.subplots(figsize=(12, fig_height))
            
            # Create gradient colors from red (negative) through yellow to green (positive)
            gaps = plot_data['urban_rural_gap'].values
            norm = plt.Normalize(vmin=gaps.min(), vmax=gaps.max())
            colors = plt.cm.RdYlGn(norm(gaps))
            
            bars = ax.barh(range(len(plot_data)), plot_data['urban_rural_gap'], color=colors, alpha=0.8)
            
            ax.set_yticks(range(len(plot_data)))
            ax.set_yticklabels(plot_data['country_name'], fontsize=10)
            ax.set_xlabel('Urban-Rural Gap in Median BF Duration (months)', fontsize=12)
            
            # Create title with subtitle on separate lines
            ax.set_title(f'Urban-Rural Inequalities in Breastfeeding Duration (n={len(plot_data)} countries)\nShowing only countries with valid data and gap > 0.1 months\n(Positive = Rural > Urban)', 
                        fontsize=13, fontweight='bold', pad=15)
            
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'urban_rural_inequality.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    return inequality_df

def analyze_regional_patterns(data: pd.DataFrame, output_dir: Path, logger):
    """Analyze regional patterns and create comparisons."""
    
    logger.info("Analyzing regional patterns...")
    
    # Add region
    data['region'] = data['country'].apply(assign_region)
    
    # Regional summary
    regional_stats = []
    
    for region in data['region'].unique():
        if region == 'Other':
            continue
            
        region_data = data[data['region'] == region]
        
        if len(region_data) == 0:
            continue
        
        # Calculate statistics with weights if available
        if 'weight' in region_data.columns:
            weights = region_data['weight']
            valid_weights = weights.notna() & (weights > 0)
            
            if valid_weights.sum() > 0:
                wd = region_data[valid_weights]
                total_weight = wd['weight'].sum()
                
                # Calculate percentages
                if 'never_breastfed' in wd.columns:
                    w_never = wd.loc[wd['never_breastfed'] == 1, 'weight'].sum()
                else:
                    w_never = wd.loc[(wd['event']==1) & (wd['duration_months']==0), 'weight'].sum()
                pct_never_bf = 100 * w_never / total_weight if total_weight > 0 else 0
                # Cessation before 6 months and continuation at 24 months come from the
                # survival curve, for the reasons given in create_country_rankings.
                s6, s24 = survival_at(wd, 6), survival_at(wd, 24)
                pct_early_weaning = 100 * (1 - s6) if not np.isnan(s6) else np.nan
                pct_extended_bf = 100 * s24 if not np.isnan(s24) else np.nan
            else:
                pct_never_bf = 100 * ((region_data['event']==1) & (region_data['duration_months']==0)).mean()
                pct_early_weaning = 100 * ((region_data['event']==1) & (region_data['duration_months']<6)).mean()
                pct_extended_bf = 100 * (region_data['duration_months'] >= 24).mean()
        else:
            pct_never_bf = 100 * ((region_data['event']==1) & (region_data['duration_months']==0)).mean()
            pct_early_weaning = 100 * ((region_data['event']==1) & (region_data['duration_months']<6)).mean()
            pct_extended_bf = 100 * (region_data['duration_months'] >= 24).mean()
        
        regional_stats.append({
            'region': region,
            'n_countries': region_data['country'].nunique(),
            'n_children': len(region_data),
            'median_duration': calculate_median_duration(region_data),
            'mean_duration': region_data['duration_months'].mean(),
            'pct_never_bf': pct_never_bf,  # Added
            'pct_early_weaning': pct_early_weaning,
            'pct_extended_bf': pct_extended_bf
        })
    
    regional_df = pd.DataFrame(regional_stats)
    
    if len(regional_df) > 0:
        regional_df = regional_df.sort_values('median_duration', ascending=False)
        regional_df.to_excel(output_dir / 'regional_comparison.xlsx', index=False)
        
        # Create regional comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sort regional_df by median duration for consistent ordering across all plots
        regional_df = regional_df.sort_values('median_duration', ascending=False)
        n_regions = len(regional_df)
        
        # (A) Median duration by region (GREENS - dark for tall bars)
        ax = axes[0, 0]
        colors_median = plt.cm.Greens(np.linspace(0.9, 0.3, n_regions))
        bars = ax.bar(range(n_regions), regional_df['median_duration'], color=colors_median)
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(regional_df['region'], rotation=45, ha='right')
        ax.set_ylabel('Median duration (months)', fontsize=13)
        ax.set_title('(A) Median duration of any breastfeeding', loc='left', fontsize=13, fontweight='bold')
        ax.axhline(y=6, color='red', linestyle='--', alpha=0.6, label='6 months (WHO exclusive breastfeeding rec.)')
        ax.axhline(y=24, color='blue', linestyle='--', alpha=0.4, label='24 months (WHO continued breastfeeding rec.)')
        ax.set_ylim(0, 30)
        ax.legend(fontsize=9, loc='upper right', frameon=False)
        ax.grid(axis='y', alpha=0.3)
        
        # (B) Cessation before 6 months by region (ORANGES - dark for tall bars)
        ax = axes[0, 1]
        regional_sorted_ew = regional_df.sort_values('pct_early_weaning', ascending=False)
        colors_ew = plt.cm.Oranges(np.linspace(0.9, 0.3, n_regions))
        bars = ax.bar(range(n_regions), regional_sorted_ew['pct_early_weaning'], color=colors_ew)
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(regional_sorted_ew['region'], rotation=45, ha='right')
        ax.set_ylabel('% ceased before 6 months', fontsize=13)
        ax.set_title('(B) Cessation of any breastfeeding before 6 months', loc='left', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # (C) Sample size by region (BLUES - dark for tall bars)
        ax = axes[1, 0]
        regional_sorted_n = regional_df.sort_values('n_children', ascending=False)
        colors_n = plt.cm.Blues(np.linspace(0.9, 0.3, n_regions))
        bars = ax.bar(range(n_regions), regional_sorted_n['n_children']/1000, color=colors_n)
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(regional_sorted_n['region'], rotation=45, ha='right')
        ax.set_ylabel('Number of children (thousands)', fontsize=13)
        ax.set_title('(C) Sample size', loc='left', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # (D) Continued breastfeeding at 24 months by region (PURPLES - dark for tall bars)
        ax = axes[1, 1]
        regional_sorted_ext = regional_df.sort_values('pct_extended_bf', ascending=False)
        colors_ext = plt.cm.Purples(np.linspace(0.9, 0.3, n_regions))
        bars = ax.bar(range(n_regions), regional_sorted_ext['pct_extended_bf'], color=colors_ext)
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(regional_sorted_ext['region'], rotation=45, ha='right')
        ax.set_ylabel('% still breastfeeding at 24 months', fontsize=13)
        ax.set_title('(D) Continued breastfeeding at 24 months', loc='left', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'regional_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return regional_df

def create_publication_tables(rankings_df: pd.DataFrame, inequality_df: pd.DataFrame, 
                            regional_df: pd.DataFrame, never_bf_country: pd.DataFrame,
                            never_bf_region: pd.DataFrame, output_dir: Path, logger):
    """Create publication-ready tables."""
    
    logger.info("Creating publication-ready tables...")
    
    # Create Excel writer with better filename
    with pd.ExcelWriter(output_dir / 'Summary_Tables.xlsx', engine='openpyxl') as writer:
        
        # Table 1: Top and bottom 10 countries
        if len(rankings_df) > 0:
            # Filter out countries with very small sample sizes for publication
            pub_rankings = rankings_df[rankings_df['n_children'] >= 100].copy()
            
            top10 = pub_rankings.head(10)[['country_name', 'region', 'n_children', 
                                          'median_duration_months', 'pct_never_breastfed', 'pct_weaned_before_6mo']]
            bottom10 = pub_rankings.tail(10)[['country_name', 'region', 'n_children',
                                             'median_duration_months', 'pct_never_breastfed', 'pct_weaned_before_6mo']]
            
            # Format for publication
            top10.columns = ['Country', 'Region', 'N', 'Median Duration (mo)', '% Never BF', '% Weaned <6mo']
            bottom10.columns = ['Country', 'Region', 'N', 'Median Duration (mo)', '% Never BF', '% Weaned <6mo']
            
            # Round numeric columns
            for col in ['Median Duration (mo)', '% Never BF', '% Weaned <6mo']:
                if col in top10.columns:
                    top10[col] = top10[col].round(1)
                if col in bottom10.columns:
                    bottom10[col] = bottom10[col].round(1)
            
            # Save tables
            top10.to_excel(writer, sheet_name='Table1_Top10_Countries', index=False)
            bottom10.to_excel(writer, sheet_name='Table1_Bottom10_Countries', index=False)
        
        # Table 2: Regional comparison (UPDATED to include never BF)
        if len(regional_df) > 0:
            regional_table = regional_df[['region', 'n_countries', 'n_children', 
                                         'median_duration', 'pct_never_bf',
                                         'pct_early_weaning', 'pct_extended_bf']]
            regional_table.columns = ['Region', 'Countries', 'N Children', 
                                     'Median Duration (mo)', '% Never BF',
                                     '% Weaned <6mo', '% Still BF at 24mo']
            regional_table = regional_table.round(1)
            regional_table.to_excel(writer, sheet_name='Table2_Regional_Analysis', index=False)
        
        # Table 3: Inequalities (if available)
        if len(inequality_df) > 0 and 'urban_rural_gap' in inequality_df.columns:
            # Filter for countries with substantial data
            ineq_table = inequality_df[(inequality_df['urban_rural_gap'].notna()) & 
                                       (inequality_df['n_urban'] >= 50) & 
                                       (inequality_df['n_rural'] >= 50)].head(20)
            if len(ineq_table) > 0:
                ineq_table = ineq_table[['country_name', 'urban_median', 'rural_median', 
                                        'urban_rural_gap', 'n_urban', 'n_rural']].round(1)
                ineq_table.columns = ['Country', 'Urban Median (mo)', 'Rural Median (mo)', 
                                     'Gap (mo)', 'N Urban', 'N Rural']
                ineq_table.to_excel(writer, sheet_name='Table3_Urban_Rural_Inequality', index=False)
        
        # Table 4: Wealth inequalities (if available)
        if len(inequality_df) > 0 and 'wealth_gap' in inequality_df.columns:
            wealth_table = inequality_df[(inequality_df['wealth_gap'].notna()) & 
                                        (inequality_df['n_poorest'] >= 50) & 
                                        (inequality_df['n_richest'] >= 50)].head(20)
            if len(wealth_table) > 0:
                wealth_table = wealth_table[['country_name', 'poorest_median', 'richest_median',
                                           'wealth_gap', 'n_poorest', 'n_richest']].round(1)
                wealth_table.columns = ['Country', 'Poorest Quintile (mo)', 'Richest Quintile (mo)',
                                       'Gap (mo)', 'N Poorest', 'N Richest']
                wealth_table.to_excel(writer, sheet_name='Table4_Wealth_Inequality', index=False)
        
        # Table 5: Never Breastfed Rankings (NEW)
        if len(never_bf_country) > 0:
            never_bf_table = never_bf_country.head(20)[
                ['country_name', 'region', 'n_children', 'pct_never_breastfed']
            ]
            never_bf_table.columns = ['Country', 'Region', 'N Children', '% Never Breastfed']
            never_bf_table['% Never Breastfed'] = never_bf_table['% Never Breastfed'].round(1)
            never_bf_table.to_excel(writer, sheet_name='Table5_Never_Breastfed', index=False)
    
    logger.info("Publication tables saved")

# -------------------- Main Function --------------------

def main(data_file: str, output_dir: str):
    """Main comparative analysis function."""
    
    # Setup paths
    data_file = Path(data_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['rankings', 'inequalities', 'regional', 'final_tables']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("COMPARATIVE ANALYSIS OF BREASTFEEDING DURATION")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {data_file}")
    data = pd.read_csv(data_file, low_memory=False)
    if 'censor_type' in data.columns:
        vc = data['censor_type'].value_counts()
        logger.info(f"Censoring mix: exact {int(vc.get('exact', 0)):,}, "
                    f"right {int(vc.get('right', 0)):,}, left (m4=93) {int(vc.get('left', 0)):,}")
        logger.info("Medians and prevalences use the Turnbull estimator wherever left-censored "
                    "children are present, and Kaplan-Meier otherwise")
    else:
        logger.warning("No censor_type column: this looks like an extraction from before v2.3, "
                       "in which children coded m4 = 93 were dropped")
    logger.info(f"Loaded {len(data):,} observations from {data['country'].nunique()} countries")
    
    # Clean data - remove any rows with null countries
    data = data[data['country'].notna()]
    
    # Add survey weights if available
    if 'v005' in data.columns:
        data['weight'] = pd.to_numeric(data['v005'], errors='coerce') / 1e6
        logger.info("Survey weights (v005/1e6) added for weighted analysis")
    
    # 1. Country rankings (UPDATED to include never-BF)
    rankings_df = create_country_rankings(data, output_dir / 'rankings', logger)
    
    # 2. Inequality analysis
    inequality_df = analyze_inequalities(data, output_dir / 'inequalities', logger)
    
    # 3. Regional patterns
    regional_df = analyze_regional_patterns(data, output_dir / 'regional', logger)
    
    # 3.5 Never breastfed analysis (NEW)
    never_bf_country, never_bf_region = analyze_never_breastfed(
        data, output_dir / 'inequalities', logger
    )
    
    # 4. Publication tables (UPDATED to include never-BF)
    create_publication_tables(rankings_df, inequality_df, regional_df, 
                            never_bf_country, never_bf_region,
                            output_dir / 'final_tables', logger)
    
    # 5. Create summary report
    logger.info("\nCreating final summary report...")
    
    summary_text = f"""
COMPARATIVE ANALYSIS SUMMARY
============================

Dataset Overview:
- Total children analyzed: {len(data):,}
- Countries included: {data['country'].nunique()}
- Regions covered: {len(regional_df) if len(regional_df) > 0 else 0}

Key Findings:
"""
    
    if len(rankings_df) > 0:
        summary_text += f"""
1. GEOGRAPHIC VARIATION
- Median BF duration range: {rankings_df['median_duration_months'].min():.1f} - {rankings_df['median_duration_months'].max():.1f} months
- Top performer: {rankings_df.iloc[0]['country_name']} ({rankings_df.iloc[0]['median_duration_months']:.1f} months)
- Lowest performer: {rankings_df.iloc[-1]['country_name']} ({rankings_df.iloc[-1]['median_duration_months']:.1f} months)
"""
    
    if len(regional_df) > 0:
        summary_text += f"""
2. REGIONAL PATTERNS
- Best performing region: {regional_df.iloc[0]['region']} ({regional_df.iloc[0]['median_duration']:.1f} months)
- Worst performing region: {regional_df.iloc[-1]['region']} ({regional_df.iloc[-1]['median_duration']:.1f} months)
"""
    
    # Calculate weighted early weaning and never-BF if weights available
    if 'weight' in data.columns:
        weights = data['weight']
        valid_weights = weights.notna() & (weights > 0)
        if valid_weights.sum() > 0:
            weighted_data = data[valid_weights]
            s6_global = survival_at(weighted_data, 6)
            early_weaning_pct = (100 * (1 - s6_global) if not np.isnan(s6_global)
                                 else np.nan)
            if 'never_breastfed' in weighted_data.columns:
                never_bf_pct = (weighted_data.loc[weighted_data['never_breastfed'] == 1, 'weight'].sum() /
                                weighted_data['weight'].sum() * 100)
            else:
                never_bf_pct = (weighted_data[(weighted_data['duration_months']==0) & 
                                            (weighted_data['event']==1)]['weight'].sum() / 
                               weighted_data['weight'].sum() * 100)
        else:
            early_weaning_pct = (data[(data['event']==1) & (data['duration_months']<6)].shape[0] / len(data) * 100)
            never_bf_pct = (data[(data['duration_months']==0) & (data['event']==1)].shape[0] / len(data) * 100)
    else:
        early_weaning_pct = (data[(data['event']==1) & (data['duration_months']<6)].shape[0] / len(data) * 100)
        never_bf_pct = (data[(data['duration_months']==0) & (data['event']==1)].shape[0] / len(data) * 100)
    
    summary_text += f"""
3. EARLY WEANING AND NEVER BREASTFED BURDEN
- Global never breastfed: {never_bf_pct:.1f}%
- Global cessation of any breastfeeding before 6 months: {early_weaning_pct:.1f}%
- Countries with >30% early weaning: {len(rankings_df[rankings_df['pct_weaned_before_6mo'] > 30]) if len(rankings_df) > 0 else 0}
- Countries with >5% never breastfed: {len(rankings_df[rankings_df['pct_never_breastfed'] > 5]) if len(rankings_df) > 0 else 0}

4. INEQUALITIES (where data available)"""
    
    # Add inequality statistics if available
    if len(inequality_df) > 0:
        if 'urban_rural_gap' in inequality_df.columns:
            urban_rural_data = inequality_df['urban_rural_gap'].dropna()
            if len(urban_rural_data) > 0:
                summary_text += f"""
- Countries with urban-rural data: {len(urban_rural_data)}
- Average urban-rural gap: {urban_rural_data.mean():.1f} months"""
        
        if 'wealth_gap' in inequality_df.columns:
            wealth_data = inequality_df['wealth_gap'].dropna()
            if len(wealth_data) > 0:
                summary_text += f"""
- Countries with wealth data: {len(wealth_data)}
- Average wealth gap: {wealth_data.mean():.1f} months"""
    
    summary_text += f"""

Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    # Save summary report
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write(summary_text)
    
    # Final log
    logger.info("=" * 60)
    logger.info("COMPARATIVE ANALYSIS COMPLETE")
    logger.info(f"All outputs saved to: {output_dir.absolute()}")
    logger.info("Key outputs:")
    logger.info("  - Country rankings: rankings/country_rankings_complete.xlsx")
    logger.info("  - Inequality analysis: inequalities/within_country_inequalities.xlsx")
    logger.info("  - Never breastfed analysis: inequalities/never_breastfed_analysis.xlsx")
    logger.info("  - Regional patterns: regional/regional_comparison.xlsx")
    logger.info("  - Summary tables: final_tables/Summary_Tables.xlsx")
    logger.info("=" * 60)
    logger.info(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform comparative geographic analysis of breastfeeding duration"
    )
    parser.add_argument("--data", 
                       default="./02_extract_breastfeeding_data/combined_breastfeeding_data.csv",
                       help="Path to combined breastfeeding data CSV")
    parser.add_argument("--out", 
                       default="./04_comparative_analysis",
                       help="Output directory for results")
    
    args = parser.parse_args()
    main(args.data, args.out)