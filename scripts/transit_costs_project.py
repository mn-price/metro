from scripts import config

import pandas as pd
import numpy as np

from scripts.common import (
    add_reference_tables,
    map_cpi_region_onto_tcp_region,
    divide_across_years,
    remove_data_without_start_end_year,
)

"""
Script processes data from Transit Costs Project (tcp), available: https://transitcosts.com/data/
Most recently published in January 15, 2025. Two primary datasets:
- Transit Cost data
- Rolling Costs data

NEED TO:
- Filter the raw dataset so we only look at metro projects
- Improve currency conversion. currently we use what they are calling real_cost, but this converts a currency into 
  PPP converted USD. It doesn't look like they have been deflated, which is fine, but I do not trust their use of xr and
  deflators. 
- Why is UK and US so high? 
"""


def import_tcp_track_data() -> pd.DataFrame:
    """
    Opens transit line cost data sourced from: https://ultraviolet.library.nyu.edu/records/9wnjp-kez15
    Fixes a few minor issues (casing of column names, renaming country to iso2_code, and fixing incorrect iso2_codes.
    :return: pd.DataFrame of raw transit cost data
    """

    # Read in data
    df = pd.read_csv(config.Paths.raw_data / "Merged-Costs-1-4.csv")

    # Convert to lower case
    df.columns = df.columns.str.lower()

    # Rename columns
    df = df.rename(columns={"country": "iso2_code"})

    # Fix incorrect iso2_codes in transit costs data
    df["iso2_code"] = np.where(df["city"] == "London", "GB", df["iso2_code"])
    df["iso2_code"] = np.where(df["city"] == "Santo Domingo", "DO", df["iso2_code"])

    return df


def keep_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Function to remove all calculation columns and only keep final data"""
    cols = [
        "country_cpi",
        "region_cpi",
        "development_status_2",
        "iso2_code",
        "iso3_code",
        "city",
        "line",
        "phase",
        "start_year",
        "end_year",
        "distributed_year",
        # "rr",
        "length",
        # "tunnelper",
        # "tunnel",
        # "elevated",
        # "atgrade",
        # "stations",
        # "platform_length_meters",
        "source1",
        "cost",
        "distributed_real_cost",
        "distributed_length",
        "currency",
        # "year",
        # "ppp_rate",
        "real_cost",
        # "cost_km_millions",
        # "anglo",
        # "inflation_index",
        # "real_cost_2023_dollars",
        # "cost_km_2023_dollars",
        "source2",
        "reference1",
        "reference2",
        "reference3",
        "comments",
    ]

    return df.filter(items=cols, axis=1)


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    group = [
        "country_cpi",
        "region_cpi",
        "development_status_2",
        "iso2_code",
        "iso3_code",
        "distributed_year",
    ]

    return df.groupby(by=group)["distributed_real_cost"].sum().reset_index(drop=False)


def distribute_all_columns_over_years(cost: pd.DataFrame) -> pd.DataFrame:
    # Add distributed cost data
    cost_km = cost.copy()
    cost_km = divide_across_years(cost_km, var_to_pro_rate="real_cost")

    # Add distributed length data
    cost_len = cost.copy()
    cost_len = divide_across_years(cost_len, var_to_pro_rate="length")

    # Merge cost_len into cost_km
    merged_cost_df = cost_km.merge(
        cost_len,
        how="left",
        on=[
            "iso2_code",
            "city",
            "line",
            "phase",
            "start_year",
            "end_year",
            "rr",
            "length",
            "tunnelper",
            "tunnel",
            "elevated",
            "atgrade",
            "stations",
            "platform_length_meters",
            "source1",
            "cost",
            "currency",
            "year",
            "ppp_rate",
            "real_cost",
            "cost_km_millions",
            "anglo",
            "inflation_index",
            "real_cost_2023_dollars",
            "cost_km_2023_dollars",
            "source2",
            "reference1",
            "reference2",
            "reference3",
            "comments",
            "country_cpi",
            "region_cpi",
            "development_status_2",
            "iso3_code",
            "distributed_year",
        ],
    ).sort_values(
        by=["country_cpi", "city", "distributed_year", "line", "phase"]
    ).reset_index(drop=True)

    return merged_cost_df

def tcp_track_pipeline() -> pd.DataFrame:
    """
    Full pipeline to import, clean, and processes raw transit infrastructure cost data before calculating cost per
    region.
    """

    # Read in metro infrastructure cost data and join reference files
    cost = import_tcp_track_data().pipe(add_reference_tables)

    # Remove projects without year data and projects ending before 2010
    cost = remove_data_without_start_end_year(cost)
    cost = cost.loc[lambda d: d.end_year >= 2010]

    # distribute costs across years
    cost = divide_across_years(cost, var_to_pro_rate="real_cost").pipe(
        keep_relevant_columns
    ).sort_values(
        by=["country_cpi", "city", "distributed_year", "line", "phase"]
    ).reset_index(drop=True)

    # Aggregate data
    agg_cost = aggregate_data(cost)

    return agg_cost


def tcp_cost_per_km() -> pd.DataFrame:
    # Read in metro infrastructure cost data and join reference files
    cost = import_tcp_track_data().pipe(add_reference_tables)

    # Remove projects without year data and projects ending before 2010
    cost = remove_data_without_start_end_year(cost)
    cost = cost.loc[lambda d: d.end_year >= 2010]

    # Distribute data over years
    cost = distribute_all_columns_over_years(cost)

    # map CPI regions onto TCP regions
    merged_cost_df = map_cpi_region_onto_tcp_region(cost)

    # Aggregate by region
    regional_data = (
        merged_cost_df.groupby(by=["region_tcp", "distributed_year"])[
            ["distributed_real_cost", "distributed_length"]
        ]
        .sum()
        .reset_index(drop=False)
    )

    # Calculate cost per km
    regional_data["cost_per_km_distributed"] = (
        regional_data["distributed_real_cost"] / regional_data["distributed_length"]
    )

    return regional_data


if __name__ == "__main__":
    # cost_by_country = tcp_track_pipeline()
    average_cost_by_region = tcp_cost_per_km()
