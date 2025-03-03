from scripts import config

import pandas as pd
import numpy as np
from bblocks.cleaning_tools.clean import clean_numeric_series

from scripts.common import (
    add_reference_tables,
    create_dev_status_3_column,
    divide_across_years,
    remove_data_without_start_end_year,
    remove_non_metro,
    convert_to_usd,
    map_country_onto_uitp_region
)


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
        "Reference",
        "Metro",
    ]

    return df.filter(items=cols, axis=1)


def import_tcp_track_data() -> pd.DataFrame:
    """
    Rolling cost data sourced from: https://transitcosts.com/data/
    Fixes a few minor issues (casing of column names, renaming country to iso2_code, fixing incorrect iso2_codes, and
    cleaning numeric series (i.e. removing "," from numbers).

    return: pd.DataFrame of raw transit cost data
    """

    # Read in data
    df = pd.read_csv(config.Paths.raw_data / "rolling_stock_cost_tcp_raw.csv")

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
            "country": "iso2_code",
        }
    )

    # Convert columns into relevant types
    df["cars"] = clean_numeric_series(df["cars"], to=float)
    df["cost"] = clean_numeric_series(df["cost"], to=float)
    df["start_year"] = clean_numeric_series(df["start_year"], to=int)
    df["end_year"] = clean_numeric_series(df["end_year"], to=int)
    df["contract_year"] = clean_numeric_series(df["contract_year"], to=int)

    # Fix incorrect iso2_codes
    df["iso2_code"] = np.where(df["city"] == "London", "GB", df["iso2_code"])

    return df


def fix_calsta_project_data(df: pd.DataFrame) -> pd.DataFrame:
    """US project CalSTA is has an incorrect start/end year combo, where start year (2027) is after end year (2026).
    Function manually changes end year so both are 2027."""

    df["end_year"] = np.where(
        (df["iso2_code"] == "US")
        & (df["city"] == "CalSTA")
        & (df["reference"] == "https://dot.ca.gov/news-releases/news-release-2024-007"),
        2027,
        df["end_year"],
    )

    return df

def distribute_all_columns_over_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function runs through the different columns that need to be distributed over years using `divide_across_years`
    function.

    return: pd.DataFrame with additional column "distributed_{var_to_pro_rate}" and additional lines for years between
    start years and end years, for cars, length, and real_cost.
    """

    # Define cols to merge on
    cols = [
        "iso2_code",
        "city",
        "trainset",
        "trains",
        "cars",
        "train_length",
        "length",
        "contract_year",
        "start_year",
        "end_year",
        "currency",
        "cost",
        "ppp_rate",
        "real_cost",
        "metro",
        "country_cpi",
        "region_cpi",
        "development_status_2",
        "development_status_3",
        "iso3_code",
        "uitp_region",
        "distributed_year",
    ]

    # Distribute number of cars over years
    df_cars = df.copy()
    df_cars = divide_across_years(df_cars, var_to_pro_rate="cars")

    # Distribute length of track over years
    df_length = df.copy()
    df_length = divide_across_years(df_length, var_to_pro_rate="length")

    # Distribute real cost of project over years
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

def add_global_average(df: pd.DataFrame, type: str) -> pd.DataFrame:
    """
    Calculates global average/minimum values for rolling stock costs per km of new track and adds them as new rows to the
    rolling stock cost dataframe.

    Args:
        type: string to specify the type of average to calculate. Mean or Min.
    """
    # Store original columns
    cols = df.columns.tolist()

    # Calculate global average by year, using aggregation for each relevant column
    global_avg = (
        df.groupby("distributed_year")
        .agg(
            distributed_length=("distributed_length", type),
            distributed_cars=("distributed_cars", type),
            distributed_real_cost=("distributed_real_cost", type),
            car_cost_per_km=("car_cost_per_km", type),
            cost_per_cars=("cost_per_cars", type),
        )
        .reset_index()
    )

    # Add a new column for the UITP region
    global_avg["uitp_region"] = f"global {type}"

    # Reorder columns to match original df
    global_avg = global_avg[cols]

    # Add the global average data to the bottom of the original df
    df_new = pd.concat([df, global_avg], axis=0, ignore_index=True)

    return df_new


def tcp_rolling_stock_pipeline() -> pd.DataFrame:
    """
    Pipeline function to create dataframe of average rolling stock costs per km of new track, by year and region.
    """

    # Import data
    df = import_tcp_track_data().pipe(fix_calsta_project_data)

    # Remove all non-metro projects
    df = remove_non_metro(df)

    # Add reference tables and map countries onto UITP region
    df = add_reference_tables(df)
    df = map_country_onto_uitp_region(df)

    # Add development_status_3 column to group EMDEs (i.e. all EMDEs, China and LDC)
    df = create_dev_status_3_column(df)

    # create column for USD value (not deflated), removing rows that cannot be converted
    df = convert_to_usd(df)
    df = df.loc[lambda d: ~(d.lcu_to_usd_xr.isna())]

    # Convert to millions
    df['real_cost'] = df['real_cost']/1000000

    # Remove rows without start year, end year, cost or carriage data
    df = remove_data_without_start_end_year(df)
    df = df.loc[lambda d: d.end_year >= 2010]
    df = df.loc[lambda d: ~(d.cars.isna())]

    # Distribute cars, length and cost over years
    df = distribute_all_columns_over_years(df)

    # Now to calculate the average costs by km by region and development status.
    # First, calculate the average costs per car by development status.

    # Aggregate by development status
    dev_status_data = (
        df.groupby(by=["development_status_3", "distributed_year"])[
            ["distributed_length", "distributed_cars", "distributed_real_cost"]
        ]
        .sum()
        .reset_index(drop=False)
    )

    # Calculate cost per car
    dev_status_data["cost_per_cars"] = (
            dev_status_data["distributed_real_cost"] / dev_status_data["distributed_cars"]
    )

    # Calculate cost per km
    dev_status_data["car_cost_per_km"] = (
            dev_status_data["distributed_real_cost"] / dev_status_data["distributed_length"]
    )

    # export as csv
    dev_status_data.to_csv(
        config.Paths.output / "dev_status_cost_per_car.csv", index=False
    )

    # Now calculate the average costs per car by region

    # Aggregate by uitp region
    regional_data = (
        df.groupby(by=["uitp_region", "distributed_year"])[
            ["distributed_length", "distributed_cars", "distributed_real_cost"]
        ]
        .sum()
        .reset_index(drop=False)
    )

    # Calculate cost per car
    regional_data["cost_per_cars"] = (
            regional_data["distributed_real_cost"] / regional_data["distributed_cars"]
    )

    # Calculate cost per car
    regional_data["car_cost_per_km"] = (
            regional_data["distributed_real_cost"] / regional_data["distributed_length"]
    )

    # Create a global minimum variable
    regional_data = add_global_average(regional_data, "min")

    # export as csv
    regional_data.to_csv(
        config.Paths.output / "regional_cost_per_car.csv", index=False
    )

    return regional_data


if __name__ == "__main__":
    rolling_stock = tcp_rolling_stock_pipeline()


