from scripts import config

import pandas as pd
import numpy as np
from pydeflate import imf_exchange, set_pydeflate_path

set_pydeflate_path(config.Paths.raw_data)

"""
Script processes data from Transit Costs Project (tcp), available: https://transitcosts.com/data/
Most recently published in January 15, 2025. Two primary datasets:
- Transit Cost data
- Rolling Costs data
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


def remove_data_without_start_end_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows that are missing either a start year or an end year.
    """

    df = df.loc[lambda d: ~(d.start_year.isna())]
    df = df.loc[lambda d: ~(d.end_year.isna())]

    return df


# def _read_clean_xr_data() -> pd.DataFrame:
#     # Read in IRENA exchange rate and deflators
#     xr = pd.read_csv(config.Paths.raw_data / "exchange_and_deflators_by_2022.csv")
#
#     # Keep relevant columns (just exchange rate in this case)
#     cols = [
#         "ISO",
#         "Year",
#         "Currency",
#         "USD/LCU Exchange Rate",
#         "IMF Exchange Rate",
#         "OECD Exchange Rate",
#     ]
#     xr = xr.filter(items=cols, axis=1).rename(
#         columns={
#             "ISO": "iso3_code",
#             "Year": "year",
#             "Currency": "currency",
#             "IMF Exchange Rate": "imf_xr",
#             "OECD Exchange Rate": "oecd_xr",
#             "USD/LCU Exchange Rate": "exchange_rate",
#         }
#     )
#
#     return xr


# def _exchange_by_currency(df: pd.DataFrame, source_currency: str) -> pd.DataFrame:
#     """
#     Converts costs in a specific currency to USD using IMF exchange rates.
#     """
#
#     # Filter DataFrame by source currency
#     df_currency = df.loc[lambda d: d.currency == source_currency]
#
#     # Apply IMF exchange rate conversion
#     df_constant = imf_exchange(
#         data=df_currency,
#         source_currency=source_currency,
#         target_currency="USA",
#         id_column="iso3_code",
#         year_column="year",
#         value_column="cost",
#         target_value_column="cost_usd",
#     )
#
#     return df_constant


# def convert_to_usd(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Converts cost column to cost_usd based on the start year, currency, and country, using IRENA exchange rates.
#     """
#     # Define the source currencies
#     # source_currencies = [#"EUR",
#     #     "USD",
#     #     "GBP"]
#     #
#     # # Initialize an empty list to collect converted DataFrames
#     # df_constant = []
#     #
#     # for source_currency in source_currencies:
#     #     # Convert each currency to USD
#     #     df_currency = _exchange_by_currency(df, source_currency)
#     #
#     #     # Append to List
#     #     df_constant.append(df_currency)
#     #
#     # # Concatenate all converted DataFrames
#     # df_constant_final = pd.concat(df_constant, ignore_index=True)
#
#     df_constant = imf_exchange(
#         data=df,
#         source_currency="currency",
#         target_currency="USA",
#         id_column="iso3_code",
#         year_column="year",
#         value_column="cost",
#         target_value_column="cost_usd",
#     )
#
#     return df_constant


def divide_across_years(df: pd.DataFrame, var_to_pro_rate: str) -> pd.DataFrame:

    # Check total
    print(f"Total after distribution: {df[var_to_pro_rate].sum()}")

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

    # Check value
    print(f"Total after distribution: {melted_df[output_var].sum()}")

    return melted_df.sort_values(
        by=["country_cpi", "city", "distributed_year", "line", "phase"]
    ).reset_index(drop=True)


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
    )

    # Aggregate data
    agg_cost = aggregate_data(cost)

    return agg_cost


def tcp_cost_per_km() -> pd.DataFrame:
    # Read in metro infrastructure cost data and join reference files
    cost = import_tcp_track_data().pipe(add_reference_tables)

    # Remove projects without year data and projects ending before 2010
    cost = remove_data_without_start_end_year(cost)
    cost = cost.loc[lambda d: d.end_year >= 2010]

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
    )

    # Aggregate by region
    regional_data = (
        merged_cost_df.groupby(by=["region_cpi", "distributed_year"])[
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
