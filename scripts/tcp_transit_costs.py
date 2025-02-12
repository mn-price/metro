"""
Script processes data from Transit Costs Project (tcp), available: https://transitcosts.com/data/, to create a dataset
of average costs per km of track by UITP region and year. We aggregate by uitp region to match the values provided in
the UITP metro statistics brief
(available here: https://cms.uitp.org/wp/wp-content/uploads/2022/05/Statistics-Brief-Metro-Figures-2021-web.pdf)

TCP most recently published two primary datasets on January 15, 2025:
- Transit Cost data --> processed in tcp_transit_costs
- Rolling Costs data --> processed in tcp_rolling_costs

See `tcp_cost_per_km_pipeline` function at the end of the script for analysis steps.
"""

from scripts import config

import pandas as pd
import numpy as np

from scripts.common import (
    add_reference_tables,
    map_country_onto_uitp_region,
    divide_across_years,
    remove_data_without_start_end_year, remove_non_metro,
)


def import_tcp_track_data() -> pd.DataFrame:
    """
    Opens transit line cost data sourced from: https://ultraviolet.library.nyu.edu/records/9wnjp-kez15
    Fixes a few minor issues (casing of column names, renaming country to iso2_code, and fixing incorrect iso2_codes.

    return: pd.DataFrame of raw transit cost data
    """

    # Read in data
    df = pd.read_csv(config.Paths.raw_data / "track_cost_tcp_raw.csv")

    # Convert to lower case
    df.columns = df.columns.str.lower()

    # Rename columns
    df = df.rename(columns={"country": "iso2_code"})

    # Fix incorrect iso2_codes in transit costs data
    df["iso2_code"] = np.where(df["city"] == "London", "GB", df["iso2_code"])
    df["iso2_code"] = np.where(df["city"] == "Santo Domingo", "DO", df["iso2_code"])

    return df


def distribute_all_columns_over_years(cost: pd.DataFrame) -> pd.DataFrame:
    """
    Applies `divide_across_years` function from common.py to the `real_cost` and `length` columns to distibute their
    values equally across the years between start_year and end_year (inclusive). Merges datasets to output one
    merged_cost_df with additional columns for `distributed_year` and `distributed_cost`.
    """

    # Add distributed cost data
    cost_km = cost.copy()
    cost_km = divide_across_years(cost_km, var_to_pro_rate="real_cost")

    # Add distributed length data
    cost_len = cost.copy()
    cost_len = divide_across_years(cost_len, var_to_pro_rate="length")

    # Merge cost_len into cost_km
    merged_cost_df = (
        cost_km.merge(
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
        )
        .sort_values(by=["country_cpi", "city", "distributed_year", "line", "phase"])
        .reset_index(drop=True)
    )

    return merged_cost_df


def tcp_cost_per_km_pipeline() -> pd.DataFrame:
    """
    Pipeline function. Creates a pd.DataFrame of average cost per km by UITP region for all years with data beyond 2010.
    Exports this pd.DataFrame as a csv to the output folder.

    return: pd.DataFrame of average regional costs of track per km by uitp region.
    """
    # Read in metro infrastructure cost data and join reference files
    cost = import_tcp_track_data().pipe(add_reference_tables)

    # Remove projects without the necessary year data for calcs
    cost = remove_data_without_start_end_year(cost)

    # Remove non-metro
    cost = remove_non_metro(cost)

    # Distribute data over years and remove rows for flows before 2010
    cost = distribute_all_columns_over_years(cost)
    cost = cost.loc[lambda d: d.distributed_year >= 2010]

    # map countries onto UITP regions
    merged_cost_df = map_country_onto_uitp_region(cost)

    # Aggregate by region
    regional_data = (
        merged_cost_df.groupby(by=["uitp_region", "distributed_year"])[
            ["distributed_real_cost", "distributed_length"]
        ]
        .sum()
        .reset_index(drop=False)
    )

    # Calculate cost per km
    regional_data["cost_per_km_distributed"] = (
        regional_data["distributed_real_cost"] / regional_data["distributed_length"]
    )

    # export as csv
    regional_data.to_csv(
        config.Paths.output / "regional_cost_track_per_km.csv", index=False
    )

    return regional_data


if __name__ == "__main__":

    average_cost_by_region = tcp_cost_per_km_pipeline()
