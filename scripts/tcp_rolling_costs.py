from scripts import config

import pandas as pd
import numpy as np
from bblocks.cleaning_tools.clean import clean_numeric_series

from scripts.common import (
    add_reference_tables,
    map_cpi_region_onto_tcp_region,
    divide_across_years,
    remove_data_without_start_end_year,
)


"""

"""


def _keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Country",
        "City",
        "Trainset",
        "Trains",
        "Cars",
        "Train length",
        "Length",
        "Contract year",
        "Start year",
        "End year",
        "Currency",
        "Cost",
        "PPP rate",
        "Reference"
    ]

    return df.filter(items=cols, axis=1)


def import_tcp_track_data() -> pd.DataFrame:
    """
    Rolling cost data sourced from: https://transitcosts.com/data/
    Fixes a few minor issues (casing of column names, renaming country to iso2_code, and fixing incorrect iso2_codes.
    :return: pd.DataFrame of raw transit cost data
    """

    # Read in data
    df = pd.read_csv(
        config.Paths.raw_data / "Rolling stock costs - Sheet1.csv",
        dtype={
            "trains": float,
            "cars": float,
            "train_length": float,
            "length": float,
            "contract_year": int,
            "start_year": int,
            "end_year": int,
            "ppp_rate": float,
        },
    )

    # filter for relevant columns
    df = _keep_relevant_columns(df)

    # Convert to lower case
    df.columns = df.columns.str.lower()
    df = df.rename(
        columns={
            "train length": "train_length",
            "contract year": "contract_year",
            "start year": "start_year",
            "end year": "end_year",
            "ppp rate": "ppp_rate",
        }
    )

    # Convert annoying columns into relevant types
    df["cars"] = clean_numeric_series(df["cars"], to=float)
    df["cost"] = clean_numeric_series(df["cost"], to=float)
    df["start_year"] = clean_numeric_series(df["start_year"], to=int)
    df["end_year"] = clean_numeric_series(df["end_year"], to=int)
    df["contract_year"] = clean_numeric_series(df["contract_year"], to=int)

    # Rename columns
    df = df.rename(columns={"country": "iso2_code"})

    # Fix incorrect iso2_codes
    df["iso2_code"] = np.where(df["city"] == "London", "GB", df["iso2_code"])

    return df

def fix_calsta_project_data(df: pd.DataFrame) -> pd.DataFrame:
    """US project CalSTA is has an incorrect start/end year combo, where start year (2027) is after end year (2026).
    Function manually changes end year so both are 2027. """

    df['end_year'] = np.where(
        (df['iso2_code'] == "US") &
        (df["city"] == "CalSTA") &
        (df['reference'] == "https://dot.ca.gov/news-releases/news-release-2024-007"),
        2027,
        df['end_year']
    )

    return df


def add_real_cost_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function adds column for cost value in USD (PPP). We do this by multiplying the cost value by the USD (PPP) converter.
    The provided value in the original data source also deflates, which we do not want to do given our data is in nominal prices.
    """
    # Remove rows without relevant data
    df = df.loc[lambda d: ~(d.ppp_rate.isna())]
    df = df.loc[lambda d: ~(d.cost.isna())]

    # Calculate real cost (in millions)
    df["real_cost"] = df["cost"] * df["ppp_rate"] / 1000000

    return df

def distribute_all_columns_over_years(df: pd.DataFrame) -> pd.DataFrame:

    # Define cols to merge on
    cols = ['iso2_code', 'city', 'trainset', 'trains', 'cars', 'train_length',
    'length', 'contract_year', 'start_year', 'end_year', 'currency', 'cost',
    'ppp_rate', 'real_cost', 'country_cpi', 'region_cpi',
    'development_status_2', 'iso3_code', 'region_tcp', 'distributed_year']

    # Distribute over years
    df_cars = df.copy()
    df_cars = divide_across_years(df_cars, var_to_pro_rate="cars")

    df_length = df.copy()
    df_length = divide_across_years(df_length, var_to_pro_rate="length")

    df_real_cost = df.copy()
    df_real_cost = divide_across_years(df_real_cost, var_to_pro_rate="real_cost")

    # Merge cost_len into cost_km
    merged_cost_df = df_length.merge(
        df_cars,
        how="left",
        on=cols,
    )

    # Merge cost_len into cost_km
    merged_cost_df = merged_cost_df.merge(
        df_real_cost,
        how="left",
        on=cols,
    )

    return merged_cost_df

def tcp_rolling_stock_pipeline() -> pd.DataFrame:

    # Import data
    df = import_tcp_track_data().pipe(fix_calsta_project_data)

    # Remove rows without start year, end year or carriage data
    df = remove_data_without_start_end_year(df)
    df = df.loc[lambda d: d.end_year >= 2010]
    df = df.loc[lambda d: ~(d.cars.isna())]

    # create column for USD value (not deflated), removing rows that cannot be converted
    df = add_real_cost_column(df)

    # convert length from m to km
    df['length'] = df['length'] / 1000

    # Add reference tables and map CPI region onto UITP region
    df = add_reference_tables(df).pipe(map_cpi_region_onto_tcp_region)

    # Distribute cars, length and cost over years
    df = distribute_all_columns_over_years(df)

    # map CPI regions onto TCP regions
    merged_df = map_cpi_region_onto_tcp_region(df)

    # Aggregate by region
    regional_data = (
        merged_df.groupby(by=["region_tcp", "distributed_year"])[
            ['distributed_length', 'distributed_cars', 'distributed_real_cost']
        ]
        .sum()
        .reset_index(drop=False)
    )

    # Calculate cost per km
    regional_data["cost_per_km_distributed"] = (
            regional_data["distributed_real_cost"] / regional_data["distributed_length"]
    )

    # Calculate cost per km
    regional_data["cost_per_cars"] = (
            regional_data["distributed_real_cost"] / regional_data["distributed_cars"]
    )

    # export as csv
    regional_data.to_csv(config.Paths.output / "regional_cars_cost_per_km.csv", index=False)

    return regional_data


if __name__ == "__main__":
    rolling_stock = tcp_rolling_stock_pipeline()
