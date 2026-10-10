
import os
import pandas as pd

BASE_DIR = "/content/drive/MyDrive/IDX-Exchange"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

SOLD_INPUT = os.path.join(
    OUTPUT_DIR,
    "CRMLSSold_202401_202604_Residential_AnalysisFiltered.csv"
)
LISTINGS_INPUT = os.path.join(
    OUTPUT_DIR,
    "CRMLSListing_202401_202604_Residential_AnalysisFiltered.csv"
)

SOLD_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "CRMLSSold_202401_202604_Residential_WithMortgageRate.csv"
)
LISTINGS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "CRMLSListing_202401_202604_Residential_WithMortgageRate.csv"
)

# Step 1: Fetch weekly mortgage rates directly from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=["observation_date"])
mortgage.columns = ["date", "rate_30yr_fixed"]

# FRED may use "." for missing observations
mortgage["rate_30yr_fixed"] = pd.to_numeric(
    mortgage["rate_30yr_fixed"], errors="coerce"
)
mortgage = mortgage.dropna(subset=["date", "rate_30yr_fixed"])

# Step 2: Resample weekly observations to calendar-month averages
mortgage["year_month"] = mortgage["date"].dt.to_period("M")

mortgage_monthly = (
    mortgage.groupby("year_month", as_index=False)["rate_30yr_fixed"]
    .mean()
)

print("Monthly mortgage rates fetched:", len(mortgage_monthly))
print(mortgage_monthly.tail())

# Step 3: Load filtered datasets
sold = pd.read_csv(SOLD_INPUT, low_memory=False)
listings = pd.read_csv(LISTINGS_INPUT, low_memory=False)

# Step 4: Create year-month keys
sold["year_month"] = pd.to_datetime(
    sold["CloseDate"], errors="coerce"
).dt.to_period("M")

listings["year_month"] = pd.to_datetime(
    listings["ListingContractDate"], errors="coerce"
).dt.to_period("M")

# Step 5: Left-merge monthly rates onto each dataset
sold_with_rates = sold.merge(
    mortgage_monthly, on="year_month", how="left", validate="many_to_one"
)

listings_with_rates = listings.merge(
    mortgage_monthly, on="year_month", how="left", validate="many_to_one"
)

# Step 6: Validate merge completeness
sold_nulls = sold_with_rates["rate_30yr_fixed"].isna().sum()
listings_nulls = listings_with_rates["rate_30yr_fixed"].isna().sum()

print("\n--- Merge validation ---")
print(f"Sold rows: {len(sold_with_rates):,}")
print(f"Sold rows with missing mortgage rate: {sold_nulls:,}")
print(f"Listings rows: {len(listings_with_rates):,}")
print(f"Listings rows with missing mortgage rate: {listings_nulls:,}")

# Save outputs even if validation fails, so unmatched rows can be inspected
sold_with_rates.to_csv(SOLD_OUTPUT, index=False)
listings_with_rates.to_csv(LISTINGS_OUTPUT, index=False)

print("\nSaved:")
print(SOLD_OUTPUT)
print(LISTINGS_OUTPUT)

if sold_nulls == 0 and listings_nulls == 0:
    print("\nSUCCESS: All rows have a matched mortgage rate.")
else:
    print(
        "\nWARNING: Some rows have no matched mortgage rate. "
        "Check their year_month values and FRED date coverage."
    )

# Preview
print("\nSold preview:")
print(
    sold_with_rates[
        ["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]
    ].head()
)

print("\nListings preview:")
print(
    listings_with_rates[
        [
            "ListingContractDate",
            "year_month",
            "ListPrice",
            "rate_30yr_fixed",
        ]
    ].head()
)
