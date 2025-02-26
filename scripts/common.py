"""
Script for functions used across rolling costs and track cost scripts.
"""

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


def map_country_onto_uitp_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps countries by iso3_code onto the regions reported in the UITP report. There is no official mapping from UITP
    for their regions, so a best-guess has been made. Mapping available in the "UIPT country-region mapping.csv" file.
    """

    # import country-region mapping file
    mapping = pd.read_csv(config.Paths.raw_data / "UITP country-region mapping.csv")

    # Drop irrelevant columns
    mapping = mapping.filter(items=["iso3_code", "uitp_region"], axis=1)

    # merge with dataframe on iso_code
    df_merged = pd.merge(left=df, right=mapping, how="left", on="iso3_code")

    return df_merged


# Storing function incase we revert to region-to-region mapping. Has been replaced by `map_cpi_country_onto_tcp_region`.
# def map_cpi_region_onto_tcp_region(df: pd.DataFrame) -> pd.DataFrame:
#
#     # Define mapping
#     region_mapping = {
#         "Central Asia and Eastern Europe": "Eurasia",
#         "East Asia and Pacific": "Asia-Pacific",
#         "Latin America & Caribbean": "Latin America",
#         "Middle East and North Africa": "MENA-Africa",
#         "Other Oceania": "Asia-Pacific",
#         "South Asia": "Asia-Pacific",
#         "US & Canada": "North America",
#         "Western Europe": "Europe",
#     }
#
#     # Map region_cpi to region_tcp
#     df["region_tcp"] = df["region_cpi"].map(region_mapping)
#
#     # Return the updated DataFrame
#     return df


def divide_across_years(df: pd.DataFrame, var_to_pro_rate: str) -> pd.DataFrame:
    """
    Splits data evenly across the years between the start year and end year (inclusive). For example, if $100m across
    2018 to 2022, new lines will be added for years 2018, 2019, ..., 2022, with `var_to_pro_rate` equal to $20m (100/5).

    Args:
        df: pd.DataFrame with data to pro-rate.
        var_to_pro_rate: column name of variable that needs to be pro-rated

    return: pd.DataFrame with additional column "distributed_{var_to_pro_rate}" and additional lines for years between
    start years and end years.
    """

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


def remove_non_metro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function removes projects which are not specific to metro. Some projects in both the track cost and rolling cost
    dataset are for light rail or commuter/regional rail. We manually go through the track cost and rolling stock cost
    datasets to split into metro and other types of rail transport.
    """

    return df.loc[lambda d: d.metro == "Metro"]


def create_dev_status_3_column(df: pd.DataFrame) -> pd.DataFrame:

    emde = ["EMDE", "LDC", "China"]

    # Create dev_status_3_column where all EMDEs (EMDE, LDCs and China) are stored as EMDE and AEs as AEs.
    df["development_status_3"] = np.where(
        df["development_status_2"] == "Advanced",
        "AE",
        np.where(df["development_status_2"].isin(emde), "EMDE", "Error"),
    )

    return df


def merge_in_uitp_new_cars_data(df: pd.DataFrame) -> pd.DataFrame:

    # import in uitp cars per km data
    cars_per_km = pd.read_csv(config.Paths.raw_data / "uitp_cars_per_km.csv")

    # merge in data on region
    merged_df = df.merge(cars_per_km, on=["uitp_region"], how="left")

    return merged_df


def convert_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses IMF exchange rates (averaged annually) to convert LCU to USD. Dataset currently uses international USD, stored
    in the real_cost column, which we want to replace with standard USD.
    """

    # Change name of current real costs column (this function
    df = df.rename(columns={"real_cost":"real_cost_usd_ppp"})

    # read in exchange rates. Change year to start_year for merge.
    xr = _read_exchange_rates()
    xr = xr.rename(columns={"year":"start_year"})

    # Merge exchange rates with costs data by year and currency.
    merged_df = df.merge(xr, on=["start_year", "currency"], how="left")

    # Add in new real_cost column
    merged_df['real_cost'] = merged_df['cost']*merged_df['lcu_to_usd_xr']

    return merged_df


def _read_exchange_rates() -> pd.DataFrame:
    """
    Function reads in exchange rates data and melts it into long format ready to merge with costs data.
    """
    # read in exchange rates and standardise column names
    xr = pd.read_csv(config.Paths.raw_data / "Exchange Rates.csv", header=2)

    # rename currency code column
    xr = xr.rename(columns={"Currency Code": "year"})

    # Melt into long format ready to merge with costs data by year and currency.
    melted_xr = xr.melt(
        id_vars="year",
        var_name="currency",
        value_name="lcu_to_usd_xr",
    )

    return melted_xr
