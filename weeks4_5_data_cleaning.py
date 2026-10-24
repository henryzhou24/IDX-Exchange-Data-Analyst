"""
IDX Exchange — Weeks 4–5 Data Cleaning
--------------------------------------
Reads mortgage-enriched residential MLS datasets, standardizes key data types,
removes verified redundant columns, adds numeric/date/geographic quality flags,
and exports cleaned datasets plus a consolidated quality report.

Raw input CSV files are never overwritten. No rows are automatically removed.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/IDX-Exchange")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "Sold": OUTPUT_DIR / "CRMLSSold_202401_202604_Residential_WithMortgageRate.csv",
    "Listings": OUTPUT_DIR / "CRMLSListing_202401_202604_Residential_WithMortgageRate.csv",
}

OUTPUTS = {
    "Sold": OUTPUT_DIR / "CRMLSSold_202401_202604_Residential_Cleaned.csv",
    "Listings": OUTPUT_DIR / "CRMLSListing_202401_202604_Residential_Cleaned.csv",
}
REPORT_PATH = OUTPUT_DIR / "weeks4_5_data_quality_report.csv"

DATE_FIELDS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

NUMERIC_FIELDS = [
    "ClosePrice",
    "ListPrice",
    "LivingArea",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "Latitude",
    "Longitude",
    "rate_30yr_fixed",
]

NUMERIC_RULES = {
    "ClosePrice": ("closeprice_nonpositive_flag", lambda s: s.notna() & (s <= 0)),
    "LivingArea": ("livingarea_nonpositive_flag", lambda s: s.notna() & (s <= 0)),
    "DaysOnMarket": ("daysonmarket_negative_flag", lambda s: s.notna() & (s < 0)),
    "BedroomsTotal": ("bedrooms_negative_flag", lambda s: s.notna() & (s < 0)),
    "BathroomsTotalInteger": ("bathrooms_negative_flag", lambda s: s.notna() & (s < 0)),
}

NUMERIC_FLAGS = [item[0] for item in NUMERIC_RULES.values()]
DATE_FLAGS = [
    "listing_after_close_flag",
    "purchase_after_close_flag",
    "negative_timeline_flag",
]
GEO_FLAGS = [
    "coordinates_missing_flag",
    "latitude_zero_flag",
    "longitude_zero_flag",
    "longitude_positive_flag",
    "coordinates_out_of_range_flag",
    "coordinates_outside_ca_flag",
]


def load_inputs():
    data = {}
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

        print(f"Loading {name}: {path.name}")
        data[name] = pd.read_csv(path, low_memory=False)

        if "rate_30yr_fixed" not in data[name].columns:
            raise ValueError(
                f"{name} input is missing the required rate_30yr_fixed column."
            )

    return data


def remove_verified_listing_duplicates(df):
    """Remove only suffix columns proven equal to their base columns."""
    df = df.copy()

    verified_candidates = [
        "PropertyType.1",
        "ListAgentFirstName.1",
        "DaysOnMarket.1",
        "LivingArea.1",
        "Longitude.1",
        "Latitude.1",
        "ListPrice.1",
        "ListAgentLastName.1",
        "CloseDate.1",
    ]

    for suffix_col in verified_candidates:
        base_col = suffix_col[:-2]

        if suffix_col not in df.columns:
            continue

        if base_col not in df.columns:
            raise ValueError(
                f"Cannot verify {suffix_col}: base column {base_col} is missing."
            )

        if not df[base_col].equals(df[suffix_col]):
            raise ValueError(
                f"{base_col} and {suffix_col} no longer match exactly. "
                "Review the columns before removing either one."
            )

        df = df.drop(columns=[suffix_col])

    # Preserve the canonical address and recover a value from the suffix
    # only if the canonical address is missing.
    if "UnparsedAddress.1" in df.columns:
        if "UnparsedAddress" not in df.columns:
            df = df.rename(columns={"UnparsedAddress.1": "UnparsedAddress"})
        else:
            recover = (
                df["UnparsedAddress"].isna()
                & df["UnparsedAddress.1"].notna()
            )
            df.loc[recover, "UnparsedAddress"] = df.loc[
                recover, "UnparsedAddress.1"
            ]
            df = df.drop(columns=["UnparsedAddress.1"])

    # Keep the unresolved buyer-office discrepancy for review.
    if "BuyerOfficeName.1" in df.columns:
        review_name = "BuyerOfficeName_suffix_review"
        if review_name in df.columns:
            raise ValueError(f"Review column already exists: {review_name}")
        df = df.rename(
            columns={"BuyerOfficeName.1": review_name}
        )

    return df


def standardize_types(df, dataset_name):
    """Convert specified fields and report values that fail conversion."""
    df = df.copy()
    conversion_rows = []

    for col in DATE_FIELDS:
        if col not in df.columns:
            continue

        original = df[col]
        converted = pd.to_datetime(original, errors="coerce")
        failures = int((original.notna() & converted.isna()).sum())
        conversion_rows.append({
            "dataset": dataset_name,
            "column": col,
            "target_type": "datetime",
            "conversion_failures": failures,
            "final_dtype": str(converted.dtype),
        })
        df[col] = converted

    for col in NUMERIC_FIELDS:
        if col not in df.columns:
            continue

        original = df[col]
        converted = pd.to_numeric(original, errors="coerce")
        failures = int((original.notna() & converted.isna()).sum())
        conversion_rows.append({
            "dataset": dataset_name,
            "column": col,
            "target_type": "numeric",
            "conversion_failures": failures,
            "final_dtype": str(converted.dtype),
        })
        df[col] = converted

    return df, conversion_rows


def add_quality_flags(df):
    """Add numeric, date-consistency, and geographic flags."""
    df = df.copy()

    for col, (flag_col, condition) in NUMERIC_RULES.items():
        if col in df.columns:
            df[flag_col] = condition(df[col]).astype(bool)

    listing = df["ListingContractDate"]
    purchase = df["PurchaseContractDate"]
    close = df["CloseDate"]

    df["listing_after_close_flag"] = (
        listing.notna() & close.notna() & (listing > close)
    )
    df["purchase_after_close_flag"] = (
        purchase.notna() & close.notna() & (purchase > close)
    )

    listing_after_purchase = (
        listing.notna() & purchase.notna() & (listing > purchase)
    )
    purchase_after_close = (
        purchase.notna() & close.notna() & (purchase > close)
    )
    df["negative_timeline_flag"] = (
        listing_after_purchase | purchase_after_close
    )

    lat = df["Latitude"]
    lon = df["Longitude"]

    df["coordinates_missing_flag"] = lat.isna() | lon.isna()
    df["latitude_zero_flag"] = lat.notna() & lat.eq(0)
    df["longitude_zero_flag"] = lon.notna() & lon.eq(0)
    df["longitude_positive_flag"] = lon.notna() & lon.gt(0)

    df["coordinates_out_of_range_flag"] = (
        (lat.notna() & ~lat.between(-90, 90))
        | (lon.notna() & ~lon.between(-180, 180))
    )

    # Broad California bounding box; outside does not necessarily mean invalid.
    df["coordinates_outside_ca_flag"] = (
        lat.notna()
        & lon.notna()
        & (
            ~lat.between(32.5, 42.1)
            | ~lon.between(-124.5, -114.0)
        )
    )

    return df


def build_quality_report(originals, cleaned):
    """Build one report covering validation, missingness, and quality flags."""
    rows = []

    def add(dataset, category, check, value, notes=""):
        rows.append({
            "dataset": dataset,
            "category": category,
            "check": check,
            "value": value,
            "notes": notes,
        })

    for name, df in cleaned.items():
        original = originals[name]

        add(name, "Validation", "Original row count", len(original))
        add(
            name, "Validation", "Cleaned row count", len(df),
            "Rows retained; no row-level filtering performed."
        )
        add(
            name, "Validation", "Row count unchanged",
            len(original) == len(df)
        )
        add(name, "Validation", "Cleaned column count", len(df.columns))
        add(
            name, "Validation", "Duplicate column names",
            int(df.columns.duplicated().sum())
        )
        add(name, "Validation", "Exact duplicate rows", int(df.duplicated().sum()))
        add(
            name, "Validation", "Mortgage rate field present",
            "rate_30yr_fixed" in df.columns
        )
        add(
            name, "Validation", "Missing mortgage rate values",
            int(df["rate_30yr_fixed"].isna().sum())
        )

        fields = [
            "CloseDate", "PurchaseContractDate", "ListingContractDate",
            "ContractStatusChangeDate", "ClosePrice", "ListPrice",
            "LivingArea", "DaysOnMarket", "BedroomsTotal",
            "BathroomsTotalInteger", "Latitude", "Longitude",
            "rate_30yr_fixed",
        ]
        for col in fields:
            if col in df.columns:
                missing = int(df[col].isna().sum())
                add(
                    name, "Missingness", col, missing,
                    f"{missing / len(df) * 100:.2f}% of rows missing."
                )

        for flag in NUMERIC_FLAGS:
            if flag in df.columns:
                add(
                    name, "Numeric quality", flag, int(df[flag].sum()),
                    "Flagged rows; records retained."
                )

        for flag in DATE_FLAGS:
            add(
                name, "Date consistency", flag, int(df[flag].sum()),
                "Flagged rows; counts may overlap."
            )

        for flag in GEO_FLAGS:
            add(
                name, "Geographic quality", flag, int(df[flag].sum()),
                "Flagged rows; counts may overlap."
            )

        for col in DATE_FIELDS:
            if col in df.columns:
                add(
                    name, "Data types", f"{col} dtype", str(df[col].dtype),
                    "Expected datetime dtype."
                )

        for category, flags in [
            ("Numeric quality", NUMERIC_FLAGS),
            ("Date consistency", DATE_FLAGS),
            ("Geographic quality", GEO_FLAGS),
        ]:
            available = [flag for flag in flags if flag in df.columns]
            add(
                name, "Record-level summary",
                f"Rows with any {category.lower()} flag",
                int(df[available].any(axis=1).sum())
            )

    return pd.DataFrame(rows)


def main():
    originals = load_inputs()
    cleaned = {}
    conversion_rows = []

    for name, original in originals.items():
        print(f"\nProcessing {name}: {len(original):,} input rows")

        df = original.copy()

        if name == "Listings":
            df = remove_verified_listing_duplicates(df)

        df, conversions = standardize_types(df, name)
        conversion_rows.extend(conversions)
        df = add_quality_flags(df)

        if len(df) != len(original):
            raise AssertionError(f"{name}: row count changed unexpectedly.")
        if df.columns.duplicated().any():
            raise AssertionError(f"{name}: duplicate column names remain.")
        if "rate_30yr_fixed" not in df.columns:
            raise AssertionError(f"{name}: mortgage-rate field missing.")

        cleaned[name] = df

    conversion_report = pd.DataFrame(conversion_rows)
    print("\n=== TYPE CONVERSION CHECK ===")
    print(conversion_report.to_string(index=False))

    total_failures = int(conversion_report["conversion_failures"].sum())
    print(f"\nTotal conversion failures: {total_failures:,}")

    report = build_quality_report(originals, cleaned)

    for name, df in cleaned.items():
        df.to_csv(OUTPUTS[name], index=False)
        print(
            f"Exported {OUTPUTS[name].name}: "
            f"{len(df):,} rows, {len(df.columns):,} columns"
        )

    report.to_csv(REPORT_PATH, index=False)
    print(f"Exported {REPORT_PATH.name}: {len(report):,} report rows")

    print("\n=== FINAL SUMMARY ===")
    for name, df in cleaned.items():
        print(
            f"{name}: rows={len(df):,}, columns={len(df.columns):,}, "
            f"exact_duplicate_rows={df.duplicated().sum():,}, "
            f"missing_mortgage_rates={df['rate_30yr_fixed'].isna().sum():,}"
        )

    if total_failures:
        print(
            "\nWARNING: Some values failed type conversion. "
            "Review the conversion report before treating the outputs as final."
        )


if __name__ == "__main__":
    main()
