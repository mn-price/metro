import numpy as np
import pandas as pd

from scripts import config


def add_reference_tables(df: pd.DataFrame) -> pd.DataFrame:

    # Import reference tables
    ref = pd.read_csv(config.Paths.raw_data / "reference_tables.csv")

    # Keep relevant columns
    cols = [
        "country_cpi",
        "region_cpi",
        "development_status_2",
        "iso2_code",
        "iso3_code",
    ]
    ref = ref.filter(items=cols, axis=1)

    # Merge in reference tables
    df = pd.merge(left=df, right=ref, how="left", on="iso2_code")

    return df


def map_cpi_region_onto_tcp_region(df: pd.DataFrame) -> pd.DataFrame:

    # Define mapping
    region_mapping = {
        "Central Asia and Eastern Europe": "Eurasia",
        "East Asia and Pacific": "Asia-Pacific",
        "Latin America & Caribbean": "Latin America",
        "Middle East and North Africa": "MENA-Africa",
        "Other Oceania": "Asia-Pacific",
        "South Asia": "Asia-Pacific",
        "US & Canada": "North America",
        "Western Europe": "Europe",
    }

    # Map region_cpi to region_tcp
    df["region_tcp"] = df["region_cpi"].map(region_mapping)

    # Return the updated DataFrame
    return df


def divide_across_years(df: pd.DataFrame, var_to_pro_rate: str) -> pd.DataFrame:

    # Check value before distribution
    print(f"Total before {var_to_pro_rate} distribution: {df[var_to_pro_rate].sum()}")

    # Store column names
    cols = df.columns

    # Find min and max values
    min_year = df.start_year.min().astype(int)
    max_year = df.end_year.max().astype(int)

    # Create list of years
    years = list(range(min_year, max_year + 1))

    # Add new columns for each year, filling with the pro-rated cost of the project
    for year in years:
        df[year] = np.where(
            (df["start_year"] <= year) & (df["end_year"] >= year),
            df[var_to_pro_rate] / (df["end_year"] - df["start_year"] + 1),
            0,
        )

    # define output variable
    output_var = "distributed_" + var_to_pro_rate

    # Melt into long format
    melted_df = df.melt(
        id_vars=cols,
        value_vars=years,
        var_name="distributed_year",
        value_name=output_var,
    )

    # Remove rows with no distributed value (i.e. remove all of rows for a project created but outside of the timeline)
    melted_df = melted_df.loc[lambda d: d[output_var] > 0]

    # Check value after distribution
    print(f"Total after {var_to_pro_rate} distribution: {melted_df[output_var].sum()}")

    return melted_df


def remove_data_without_start_end_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows that are missing either a start year or an end year.
    """

    df = df.loc[lambda d: ~(d.start_year.isna())]
    df = df.loc[lambda d: ~(d.end_year.isna())]

    return df
