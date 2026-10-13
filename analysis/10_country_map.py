# 10_country_map.py
# World map of the countries in the analysis, each country coloured by how many
# children it contributes (sample density). Countries with no DHS KR data are grey.
#
# Usage (from D:\early-weaning-analysis):
#   python analysis\10_country_map.py --data ".\02_extract_v24\combined_breastfeeding_data.csv" --out ".\r30_country_map"
#   python analysis\10_country_map.py ... --metric n_surveys        # colour by number of surveys instead
#   python analysis\10_country_map.py ... --shapes ".\ne_50m_admin_0_countries.zip"   # offline copy of the map
#
# Country codes are DHS codes, NOT ISO codes (MD = Madagascar, MB = Moldova, NI = Niger,
# BU = Burundi, GU = Guatemala, ...). Indian NFHS state files (AP, AR, JM, MG, NA, ...)
# are subsets of India and are not drawn as countries.
# Until 02 is re-run with the v000 country fix, the KE, MZ, TN and TR totals still
# contain the Kerala, Mizoram, Tamil Nadu and Tripura state files.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

NE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"

# DHS country code -> ISO3 (Natural Earth ADM0_A3)
DHS_TO_ISO3 = {
    'AF': 'AFG', 'AL': 'ALB', 'AM': 'ARM', 'AO': 'AGO', 'AZ': 'AZE', 'BD': 'BGD', 'BF': 'BFA',
    'BJ': 'BEN', 'BO': 'BOL', 'BR': 'BRA', 'BU': 'BDI', 'CD': 'COD', 'CF': 'CAF', 'CG': 'COG',
    'CI': 'CIV', 'CM': 'CMR', 'CO': 'COL', 'DR': 'DOM', 'EC': 'ECU', 'EG': 'EGY', 'ET': 'ETH',
    'GA': 'GAB', 'GH': 'GHA', 'GM': 'GMB', 'GN': 'GIN', 'GU': 'GTM', 'GY': 'GUY', 'HN': 'HND',
    'HT': 'HTI', 'IA': 'IND', 'ID': 'IDN', 'JO': 'JOR', 'KE': 'KEN', 'KH': 'KHM', 'KK': 'KAZ',
    'KM': 'COM', 'KY': 'KGZ', 'LB': 'LBR', 'LK': 'LKA', 'LS': 'LSO', 'MA': 'MAR', 'MB': 'MDA',
    'MD': 'MDG', 'ML': 'MLI', 'MM': 'MMR', 'MR': 'MRT', 'MV': 'MDV', 'MW': 'MWI', 'MX': 'MEX',
    'MZ': 'MOZ', 'NC': 'NIC', 'NG': 'NGA', 'NI': 'NER', 'NM': 'NAM', 'NP': 'NPL', 'OS': 'NGA',
    'PE': 'PER', 'PG': 'PNG', 'PH': 'PHL', 'PK': 'PAK', 'PY': 'PRY', 'RW': 'RWA', 'SD': 'SDN',
    'SL': 'SLE', 'SN': 'SEN', 'ST': 'STP', 'SZ': 'SWZ', 'TD': 'TCD', 'TG': 'TGO', 'TH': 'THA',
    'TJ': 'TJK', 'TL': 'TLS', 'TN': 'TUN', 'TR': 'TUR', 'TT': 'TTO', 'TZ': 'TZA', 'UA': 'UKR',
    'UG': 'UGA', 'UZ': 'UZB', 'VN': 'VNM', 'YE': 'YEM', 'ZA': 'ZAF', 'ZM': 'ZMB', 'ZW': 'ZWE',
}
# Indian NFHS-2 state files (DHS country code IA); not drawn as separate countries
INDIA_STATE_CODES = {'AP', 'AR', 'AS', 'BH', 'DL', 'GJ', 'GO', 'HP', 'HR', 'JM', 'KA', 'MG',
                     'MH', 'MN', 'MP', 'NA', 'OR', 'PJ', 'RJ', 'SK', 'UP', 'WB'}

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'legend.fontsize': 11})


def main(data_file, out_dir, metric, shapes):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(data_file, usecols=lambda c: c in {'country', 'survey_year', 'v005'},
                    low_memory=False)
    d = d[d['country'].notna()]
    if 'v005' in d.columns:                       # same sample as the weighted analyses
        d = d[pd.to_numeric(d['v005'], errors='coerce') > 0]

    states = d[d['country'].isin(INDIA_STATE_CODES)]
    nat = d[~d['country'].isin(INDIA_STATE_CODES)].copy()
    unknown = sorted(set(nat['country']) - set(DHS_TO_ISO3))
    if unknown:
        print(f"WARNING: no ISO3 mapping for {unknown}; these are left off the map")
    nat['iso3'] = nat['country'].map(DHS_TO_ISO3)
    nat = nat[nat['iso3'].notna()]

    agg = (nat.groupby('iso3')
              .agg(n_children=('country', 'size'),
                   n_surveys=('survey_year', 'nunique'),
                   dhs_codes=('country', lambda s: ','.join(sorted(set(s)))),
                   first_year=('survey_year', 'min'),
                   last_year=('survey_year', 'max'))
              .reset_index())
    agg.to_excel(out / 'r30_country_map_data.xlsx', index=False)
    print(f"{len(agg)} countries on the map; {len(nat):,} children "
          f"(+ {len(states):,} in Indian state files, not drawn separately)")

    world = gpd.read_file(shapes or NE_URL)
    world = world[world['ADM0_A3'] != 'ATA']                     # drop Antarctica
    missing = sorted(set(agg['iso3']) - set(world['ADM0_A3']))
    if missing:
        print(f"WARNING: not in the map file: {missing}")
    world = world.merge(agg, left_on='ADM0_A3', right_on='iso3', how='left')
    world = world.to_crs('+proj=robin')                          # Robinson projection

    # Classes: sample sizes span ~1,000 to ~500,000, so bins are roughly logarithmic
    if metric == 'n_children':
        bounds = [0, 5_000, 10_000, 25_000, 50_000, 100_000, 1_000_000]
        labels = ['<5k', '5–10k', '10–25k', '25–50k', '50–100k', '≥100k']
        title = 'Children contributing to the analysis, by country'
    else:
        bounds = [0.5, 1.5, 2.5, 3.5, 5.5, 20]
        labels = ['1', '2', '3', '4–5', '≥6']
        title = 'Number of DHS surveys contributing, by country'
    cmap = plt.get_cmap('Blues', len(labels) + 1)                # one hue, light -> dark
    colors = [cmap(i + 1) for i in range(len(labels))]
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    world.plot(ax=ax, color='#E4E4E4', edgecolor='white', linewidth=0.3)          # no data
    has = world[world[metric].notna()]
    has.plot(ax=ax, column=metric, cmap=cmap, norm=BoundaryNorm(bounds, cmap.N), edgecolor='white', linewidth=0.3)
    ax.set_axis_off()
    ax.set_title(title, fontweight='bold', loc='left')

    # Discrete legend bar: equal-width class boxes, white gaps, labels underneath
    lax = fig.add_axes([0.08, 0.10, 0.36, 0.025])
    for i, c in enumerate(colors):
        lax.add_patch(plt.Rectangle((i + 0.03, 0), 0.94, 1, color=c))
    lax.set_xlim(0, len(colors)); lax.set_ylim(0, 1)
    lax.set_yticks([]); lax.set_xticks(np.arange(len(labels)) + 0.5)
    lax.set_xticklabels(labels, fontsize=10); lax.tick_params(length=0)
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_title('Children (n)' if metric == 'n_children' else 'Surveys (n)',
                  fontsize=10, loc='left')
    lax.text(0, -2.3, 'Grey: no DHS Children\u2019s Recode data in this study',
             fontsize=9, color='#555555', transform=lax.transData)

    fname = out / f"r30_country_map_{metric}"
    plt.savefig(f"{fname}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{fname}.tif", dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close()
    print(f"Wrote {fname}.png / .tif and r30_country_map_data.xlsx")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./02_extract_v24/combined_breastfeeding_data.csv')
    ap.add_argument('--out', default='./r30_country_map')
    ap.add_argument('--metric', choices=['n_children', 'n_surveys'], default='n_children')
    ap.add_argument('--shapes', default=None, help='local Natural Earth zip (else downloaded)')
    a = ap.parse_args()
    main(a.data, a.out, a.metric, a.shapes)
