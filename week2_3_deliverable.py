
from pathlib import Path
import pandas as pd

# ============================================================
# IDX Exchange Data Analyst Internship — Weeks 2–3 Deliverable
# ============================================================
# Inputs:
#   data/   : Original CRMLS monthly CSV files
#   output/ : Week 1 residential-filtered Listing and Sold CSVs
#
# Filtering logic:
#   1. Week 1 files contain records where PropertyType is
#      "Residential" (after trimming surrounding whitespace).
#   2. For the new Sold analysis dataset, retain residential
#      records, remove exact duplicate rows, and retain only
#      records with a positive ClosePrice and a valid CloseDate.
#   3. Do NOT remove repeated ListingKeys automatically:
#      different records with the same key may contain
#      meaningful differences.
#
# Raw and processed real estate data should not be uploaded
# to the public GitHub repository.
# ============================================================

PROJECT_DIR = Path("/content/drive/MyDrive/IDX-Exchange")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

LISTING_PATH = (
    OUTPUT_DIR / "CRMLSListing_202401_202604_Residential.csv"
)
SOLD_PATH = (
    OUTPUT_DIR / "CRMLSSold_202401_202604_Residential.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    """Read a CSV while avoiding mixed-type inference warnings."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path, low_memory=False)


# ------------------------------------------------------------
# 1. Identify property types in the original source files
# ------------------------------------------------------------

type_series = []

source_files = sorted(DATA_DIR.glob("CRMLS*.csv"))

if not source_files:
    raise FileNotFoundError(
        f"No original CRMLS CSV files found in {DATA_DIR}"
    )

for file_path in source_files:
    try:
        types_df = pd.read_csv(
            file_path,
            usecols=["PropertyType"],
            dtype="string",
            low_memory=False
        )

        values = (
            types_df["PropertyType"]
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )

        type_series.append(values)

    except ValueError:
        print(f"WARNING: PropertyType column not found in {file_path.name}")

if type_series:
    all_types = pd.concat(type_series, ignore_index=True)

    property_type_report = (
        all_types.value_counts()
        .rename_axis("PropertyType")
        .reset_index(name="row_count")
    )

    property_type_report.to_csv(
        OUTPUT_DIR / "week2_3_unique_property_types.csv",
        index=False
    )

    print("\nUnique PropertyType values found in source data:")
    print(property_type_report.to_string(index=False))
else:
    raise ValueError("Could not find PropertyType in the source files.")


# ------------------------------------------------------------
# 2. Load the Week 1 residential-filtered datasets
# ------------------------------------------------------------

listing_df = load_csv(LISTING_PATH)
sold_df = load_csv(SOLD_PATH)

for name, df in [("Listing", listing_df), ("Sold", sold_df)]:
    if "PropertyType" not in df.columns:
        raise KeyError(f"{name} dataset has no PropertyType column.")

    unexpected = (
        df["PropertyType"].astype("string").str.strip().ne("Residential")
    )

    if unexpected.any():
        raise ValueError(
            f"{name} dataset contains {int(unexpected.sum())} "
            "rows that are not PropertyType == 'Residential'."
        )


# ------------------------------------------------------------
# 3. Create a NEW filtered Sold dataset for sale-price analysis
# ------------------------------------------------------------

# Keep the Week 1 source file unchanged.
sold_filtered = sold_df.copy()

# Remove exact duplicate rows only; do not deduplicate by ListingKey.
rows_before = len(sold_filtered)
sold_filtered = sold_filtered.drop_duplicates().copy()
exact_duplicates_removed = rows_before - len(sold_filtered)

# Parse dates and prices for reliable filtering.
sold_filtered["CloseDate"] = pd.to_datetime(
    sold_filtered["CloseDate"], errors="coerce"
)

sold_filtered["ClosePrice"] = pd.to_numeric(
    sold_filtered["ClosePrice"], errors="coerce"
)

# Exclude records without a valid sale price or closing date.
valid_sale = (
    sold_filtered["ClosePrice"].gt(0)
    & sold_filtered["CloseDate"].notna()
)

invalid_sale_rows_removed = int((~valid_sale).sum())
sold_filtered = sold_filtered.loc[valid_sale].copy()

FILTERED_SOLD_PATH = (
    OUTPUT_DIR
    / "CRMLSSold_202401_202604_Residential_AnalysisFiltered.csv"
)

sold_filtered.to_csv(FILTERED_SOLD_PATH, index=False)

print("\nSold dataset filtering:")
print(f"Rows before filtering: {rows_before:,}")
print(f"Exact duplicate rows removed: {exact_duplicates_removed:,}")
print(f"Rows removed for invalid price/date: {invalid_sale_rows_removed:,}")
print(f"Rows in new filtered dataset: {len(sold_filtered):,}")
print(f"Saved to: {FILTERED_SOLD_PATH}")


# ------------------------------------------------------------
# 4. Generate null-count summary tables
# ------------------------------------------------------------

datasets = {
    "Listing": listing_df,
    "Sold": sold_df,
    "Filtered Sold": sold_filtered
}

null_summary_rows = []

for dataset_name, df in datasets.items():
    null_counts = df.isna().sum()

    for column in df.columns:
        null_count = int(null_counts[column])

        null_summary_rows.append({
            "dataset": dataset_name,
            "column": column,
            "row_count": len(df),
            "null_count": null_count,
            "non_null_count": int(len(df) - null_count),
            "null_percentage": round(
                null_count / len(df) * 100, 4
            ) if len(df) else 0
        })

null_summary = pd.DataFrame(null_summary_rows)

null_summary_path = OUTPUT_DIR / "week2_3_null_count_summary.csv"
null_summary.to_csv(null_summary_path, index=False)


# ------------------------------------------------------------
# 5. Flag every column with MORE THAN 90% missing values
# ------------------------------------------------------------

missing_over_90 = null_summary.loc[
    null_summary["null_percentage"] > 90
].copy()

missing_report_path = (
    OUTPUT_DIR / "week2_3_missing_over_90_percent.csv"
)
missing_over_90.to_csv(missing_report_path, index=False)

print("\nColumns with more than 90% missing values:")
if missing_over_90.empty:
    print("None.")
else:
    print(
        missing_over_90[
            ["dataset", "column", "null_count", "null_percentage"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 6. Numeric distribution summary for the required variables
# ------------------------------------------------------------

# Sale-price analysis uses the new filtered Sold dataset.
# For each metric, null/non-numeric values are excluded from
# its statistics. Other numeric values are not silently removed.

numeric_columns = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket"
]

distribution_rows = []

for column in numeric_columns:
    if column not in sold_filtered.columns:
        print(f"WARNING: {column} not found; skipping.")
        continue

    values = pd.to_numeric(
        sold_filtered[column], errors="coerce"
    ).dropna()

    if values.empty:
        distribution_rows.append({
            "variable": column,
            "valid_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p1": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None
        })
        continue

    distribution_rows.append({
        "variable": column,
        "valid_count": int(values.count()),
        "min": values.min(),
        "max": values.max(),
        "mean": values.mean(),
        "median": values.median(),
        "p1": values.quantile(0.01),
        "p5": values.quantile(0.05),
        "p25": values.quantile(0.25),
        "p50": values.quantile(0.50),
        "p75": values.quantile(0.75),
        "p95": values.quantile(0.95),
        "p99": values.quantile(0.99)
    })

distribution_summary = pd.DataFrame(distribution_rows)

distribution_path = (
    OUTPUT_DIR / "week2_3_numeric_distribution_summary.csv"
)
distribution_summary.to_csv(distribution_path, index=False)

print("\nNumeric distribution summary:")
print(distribution_summary.to_string(index=False))

print("\nAll deliverable files saved:")
for path in [
    OUTPUT_DIR / "week2_3_unique_property_types.csv",
    null_summary_path,
    missing_report_path,
    distribution_path,
    FILTERED_SOLD_PATH
]:
    print(f"- {path}")

# ============================================================
# 7. Numeric Distribution Review: Listing and Sold
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERIC_FIELDS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

# Create a separate residential analysis copy of Listing.
# Do not remove listings merely because ClosePrice is missing:
# active and pending listings may not have a closing price.
listing_analysis = listing_df.copy()

listing_analysis = listing_analysis.loc[
    listing_analysis["PropertyType"]
    .astype("string").str.strip().eq("Residential")
].drop_duplicates().copy()

listing_analysis_path = (
    OUTPUT_DIR
    / "CRMLSListing_202401_202604_Residential_AnalysisFiltered.csv"
)
listing_analysis.to_csv(listing_analysis_path, index=False)

# Rebuild the Sold analysis copy using the documented rules.
sold_analysis = sold_df.copy()

sold_analysis = sold_analysis.loc[
    sold_analysis["PropertyType"]
    .astype("string").str.strip().eq("Residential")
].drop_duplicates().copy()

sold_analysis["CloseDate"] = pd.to_datetime(
    sold_analysis["CloseDate"], errors="coerce"
)
sold_analysis["ClosePrice"] = pd.to_numeric(
    sold_analysis["ClosePrice"], errors="coerce"
)

sold_analysis = sold_analysis.loc[
    sold_analysis["ClosePrice"].gt(0)
    & sold_analysis["CloseDate"].notna()
].copy()

sold_analysis_path = (
    OUTPUT_DIR
    / "CRMLSSold_202401_202604_Residential_AnalysisFiltered.csv"
)
sold_analysis.to_csv(sold_analysis_path, index=False)

analysis_datasets = {
    "Listing": listing_analysis,
    "Sold": sold_analysis
}

# Save a fresh null-count summary for both original residential
# datasets and both final analysis datasets.
null_rows = []

for dataset_name, frame in {
    "Listing": listing_df,
    "Sold": sold_df,
    "Filtered Listing": listing_analysis,
    "Filtered Sold": sold_analysis
}.items():
    for column in frame.columns:
        null_count = int(frame[column].isna().sum())
        null_rows.append({
            "dataset": dataset_name,
            "column": column,
            "row_count": len(frame),
            "null_count": null_count,
            "non_null_count": len(frame) - null_count,
            "null_percentage": (
                round(100 * null_count / len(frame), 4)
                if len(frame) else 0
            )
        })

null_summary = pd.DataFrame(null_rows)
null_summary.to_csv(
    OUTPUT_DIR / "week2_3_null_count_summary.csv", index=False
)

missing_over_90 = null_summary.loc[
    null_summary["null_percentage"] > 90
].copy()
missing_over_90.to_csv(
    OUTPUT_DIR / "week2_3_missing_over_90_percent.csv",
    index=False
)

# Distribution summaries, IQR outlier counts, and plots.
plot_dir = OUTPUT_DIR / "week2_3_numeric_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

distribution_rows = []
outlier_rows = []

for dataset_name, frame in analysis_datasets.items():
    for column in NUMERIC_FIELDS:
        if column not in frame.columns:
            print(f"WARNING: {dataset_name} is missing {column}")
            continue

        values = pd.to_numeric(
            frame[column], errors="coerce"
        ).dropna()

        if values.empty:
            print(f"WARNING: No numeric values for {dataset_name}: {column}")
            continue

        q1 = values.quantile(0.25)
        median = values.quantile(0.50)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        distribution_rows.append({
            "dataset": dataset_name,
            "variable": column,
            "valid_count": int(values.count()),
            "null_count": int(frame[column].isna().sum()),
            "min": values.min(),
            "max": values.max(),
            "mean": values.mean(),
            "median": median,
            "p1": values.quantile(0.01),
            "p5": values.quantile(0.05),
            "p25": q1,
            "p50": median,
            "p75": q3,
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99)
        })

        below = values < lower_fence
        above = values > upper_fence

        outlier_rows.append({
            "dataset": dataset_name,
            "variable": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_iqr_fence": lower_fence,
            "upper_iqr_fence": upper_fence,
            "outliers_below": int(below.sum()),
            "outliers_above": int(above.sum()),
            "total_iqr_outliers": int(below.sum() + above.sum()),
            "smallest_value": values.min(),
            "largest_value": values.max()
        })

        # For readability, plots display the central 98% of values.
        # Full-data min/max and IQR outlier counts are retained above.
        p1 = values.quantile(0.01)
        p99 = values.quantile(0.99)
        plot_values = values.loc[values.between(p1, p99)]

        safe_name = (
            f"{dataset_name.lower().replace(' ', '_')}_{column}"
        )

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(plot_values, bins=40)
        ax.set_title(
            f"{dataset_name}: {column} Histogram "
            "(central 98% shown)"
        )
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}_histogram.png", dpi=120)
        plt.close(fig)

        # Boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(plot_values, vert=True, showfliers=False)
        ax.set_title(
            f"{dataset_name}: {column} Boxplot "
            "(central 98% shown)"
        )
        ax.set_ylabel(column)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}_boxplot.png", dpi=120)
        plt.close(fig)

distribution_summary = pd.DataFrame(distribution_rows)
distribution_summary.to_csv(
    OUTPUT_DIR / "week2_3_numeric_distribution_summary.csv",
    index=False
)

outlier_report = pd.DataFrame(outlier_rows)
outlier_report.to_csv(
    OUTPUT_DIR / "week2_3_numeric_outlier_flags.csv",
    index=False
)

print("\nAnalysis dataset row counts:")
print(f"Filtered Listing: {len(listing_analysis):,}")
print(f"Filtered Sold: {len(sold_analysis):,}")

print("\nNumeric distribution summary:")
print(distribution_summary.to_string(index=False))

print("\nIQR outlier review:")
print(outlier_report.to_string(index=False))

print("\nUpdated deliverables:")
print(f"- {listing_analysis_path}")
print(f"- {sold_analysis_path}")
print(f"- {OUTPUT_DIR / 'week2_3_null_count_summary.csv'}")
print(f"- {OUTPUT_DIR / 'week2_3_missing_over_90_percent.csv'}")
print(f"- {OUTPUT_DIR / 'week2_3_numeric_distribution_summary.csv'}")
print(f"- {OUTPUT_DIR / 'week2_3_numeric_outlier_flags.csv'}")
print(f"- {plot_dir} (histograms and boxplots)")
