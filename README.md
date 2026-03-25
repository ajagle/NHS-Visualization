# NHS A&E Performance & Waiting Times Analysis

A data analysis project examining NHS England A&E Attendances and Emergency Admissions data (October 2025 - January 2026) to identify trends in waiting times, seasonal demand patterns, and trust-level performance variation.

## Key Findings

- **4-Hour Target Performance** declined from 73.6% (Oct 2025) to 71.8% (Jan 2026) — all months significantly below the 95% national target, with the sharpest decline in January consistent with winter pressure
- **12+ Hour DTA Waits** increased by 31.7% from October to January — a critical patient safety metric showing the system under increasing strain during winter months
- **Regional Disparities** show a 5.8 percentage point gap between the best (London, 74.8%) and worst performing (South West, 69.0%) regions in January 2026
- **Volume-Performance Relationship**: OLS regression analysis reveals a weak negative correlation between trust size and 4-hour performance, suggesting larger trusts face systemic challenges beyond simple demand scaling
- **Busiest Trusts** include Barts Health, Royal Free London, and Manchester University NHS FT — each handling 40,000+ attendances monthly

## Visualisations (12 Figures)

All charts use an NHS-inspired colour palette with professional styling, numbered figure captions, and data source footnotes.

| Figure | Chart Type | Description |
|--------|-----------|-------------|
| 01 | Dual-axis bar + line | Total attendances with month-on-month % change overlay |
| 02 | Gap chart | 4-hour performance vs 95% target with shaded performance gap |
| 03 | Stacked bar + proportion line | DTA waiting times (4-12hr / 12+ hr) with 12hr proportion trend |
| 04 | Lollipop chart | Regional 4-hour performance with colour-coded bands and national average reference |
| 05 | Diverging bar chart | Top/bottom 10 trusts showing deviation from national average |
| 06 | Stacked area chart | Emergency admissions broken down by Type 1 / Type 2 / Other |
| 07 | Annotated heatmap | Regional attendances with column normalisation and MoM % change annotations |
| 08 | Waterfall chart | Cumulative monthly change in 12+ hour waits from October baseline |
| 09 | Violin + box plot | Trust-level 4-hour performance distribution showing spread and skew by month |
| 10 | Scatter + OLS regression | Attendances volume vs performance with R², p-value, and colour-coded regions |
| 11 | Small multiples (faceted) | Regional 4-hour performance trends — one panel per region with period change |
| 12 | Diverging heatmap | Month-on-month percentage point change in 4-hour performance by region |

## Project Structure

```
nhs-ae-analysis/
├── data/                          # Raw monthly CSV datasets
│   ├── October-2025.csv
│   ├── November-2025.csv
│   ├── December-2025.csv
│   └── January-2026.csv
├── output/                        # Generated charts (12 PNGs) and summary CSVs
│   ├── 01_monthly_attendances.png
│   ├── 02_4hr_performance_gap.png
│   ├── 03_dta_waiting_times.png
│   ├── 04_regional_lollipop.png
│   ├── 05_trust_diverging_bar.png
│   ├── 06_emergency_admissions_stacked.png
│   ├── 07_regional_heatmap.png
│   ├── 08_12hr_waits_waterfall.png
│   ├── 09_trust_distribution.png
│   ├── 10_volume_vs_performance.png
│   ├── 11_regional_small_multiples.png
│   ├── 12_regional_change_heatmap.png
│   ├── national_monthly_summary.csv
│   ├── regional_summary_latest.csv
│   └── trust_detail_latest.csv
├── nhs_ae_analysis.py             # Main analysis script
├── requirements.txt
└── README.md
```

## Data Source

Publicly available NHS England **A&E Attendances and Emergency Admissions** monthly statistics, published by NHS England.

- **Source**: [NHS England Statistical Work Areas](https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/)
- **Period**: October 2025 – January 2026
- **Metrics covered**: A&E attendances (Type 1, Type 2, Other), 4-hour breaches, emergency admissions, DTA waiting times (4–12hrs and 12+ hrs)

## Methodology

1. **Data Cleaning & Preprocessing**: Loaded and preprocessed four monthly CSV datasets using Pandas — handled missing values, stripped whitespace from trust codes/names, converted numeric columns, and removed aggregate summary rows
2. **Feature Engineering**: Derived key performance metrics including total attendances, 4-hour compliance rate (%), emergency admissions by type, and DTA waiting time brackets per trust and region
3. **Exploratory Data Analysis**: Aggregated data at national, regional, and trust level across 197 providers to surface trends, seasonal patterns, and outliers
4. **Statistical Analysis**: Applied OLS linear regression (SciPy) to test the relationship between trust volume and 4-hour performance, reporting R², p-value, and slope
5. **Visualisation**: Created 12 publication-quality charts using Matplotlib with an NHS-inspired colour palette, dual-axis designs, faceted layouts, diverging charts, and statistical annotations

## How to Run

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
git clone https://github.com/ajagle/nhs-ae-analysis.git
cd nhs-ae-analysis
pip install -r requirements.txt
```

### Run the Analysis

```bash
python nhs_ae_analysis.py
```

All outputs (12 charts and 3 summary CSVs) are saved to the `output/` directory.

## Skills Demonstrated

- **Data Wrangling**: Cleaning and preprocessing real-world NHS datasets with inconsistent formatting using Pandas
- **Exploratory Data Analysis**: Identifying trends, seasonal patterns, and outliers across 197 trusts and 7 regions over 4 months
- **Statistical Analysis**: OLS regression to quantify volume-performance relationships with significance testing (SciPy)
- **Advanced Data Visualisation**: Publication-quality charts using Matplotlib — including dual-axis plots, waterfall charts, diverging bars, violin plots, lollipop charts, small multiples, and annotated heatmaps
- **Domain Knowledge**: Understanding of NHS A&E performance metrics, the 4-hour operational standard, Decision to Admit (DTA) pathways, and winter pressure dynamics
- **Communication**: Translating complex data into clear, non-technical insights suitable for operational stakeholders and service improvement planning

## License

This project uses publicly available NHS England data published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
