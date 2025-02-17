"""
Script pulls UITP regional data on evolution of length of track and applies the 'regional_cost_track_per_km.csv' and
'regional_cars_cost_per_km.csv' to estimate total costs.

Regional UITP evolution of track comes from World Metro Statistics report (2021). This will be updated in the end of Q1
2025 with the World Metro Statistics report (2025). 2021 report available here:
https://www.uitp.org/publications/metro-world-figures-2021/

In the next interation, we will be able to apply cost per car instead of cost of cars per km, as the UITP is releasing
their evolution of number of cars data in Q1 2025.
"""

import pandas as pd
import numpy as np
from scripts import config


def read_uitp_track_length_data() -> pd.DataFrame:
    return pd.read_csv(
        config.Paths.raw_data / "uitp_track_length_data.csv", encoding="UTF-8"
    )


def clean_uitp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function cleans raw UITP data, scraped from https://www.uitp.org/publications/metro-world-figures-2021/.
    Changes MENA-Africa region to MENA, melts regional values into long format (adding new columns for uitp_region and
    length_of_trac, and removes the total value.
    """

    # Change MENA-Africa to MENA (it does not include Africa in the UITP report)
    df = df.rename(columns={"Year": "year", "MENA-Africa": "MENA"})

    # Melt into long format
    df_melted = pd.melt(
        df,
        id_vars="year",
        value_vars=[
            "Asia-Pacific",
            "Eurasia",
            "Europe",
            "Latin America",
            "MENA",
            "North America",
            "Total",
        ],
        var_name="uitp_region",
        value_name="length_of_track",
    )

    # Drop total values
    df_melted = df_melted.loc[lambda d: ~(d.uitp_region == "Total")]

    return df_melted


def calculate_new_track_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column for 'new_track_length' to the dataframe by taking the difference between length_of_track in year
    x and length_of_track in year x-1.
    """

    # Very long-winded way of calculating the yearly track length increase ...
    df_year_plus_1 = df.copy().rename(
        columns={"length_of_track": "length_of_track_year_prior"}
    )
    df_year_plus_1["year"] = df_year_plus_1["year"] + 1
    df = pd.merge(df, df_year_plus_1, on=["year", "uitp_region"], how="left")
    df["new_track_length"] = df["length_of_track"] - df["length_of_track_year_prior"]

    # Drop calc columns
    df = df.filter(items=["year", "uitp_region", "new_track_length"], axis=1)

    # remove 2013 as NaN values
    df = df.loc[lambda d: ~(d.year == 2013)]

    return df


def estimate_future_years(df: pd.DataFrame, years=list[int]) -> pd.DataFrame:
    """
    Function assumes the same km of track growth from average of 2017-2020 continues into 2021, 2022 and 2023.
    TO BE EXCLUDED FROM CODE WHEN NEW DATA IS RELEASED!

    return: pd.DataFrame with additional rows for each region for 2021, 2022 and 2023.
    """
    # Filter and group df to calculate average new_track_length per region
    df_avg = df.loc[lambda d: d.year >= 2017]
    df_avg = (
        df_avg.groupby(by="uitp_region")["new_track_length"]
        .mean()
        .reset_index(drop=False)
    )

    # Create new rows based on df_avg
    new_rows = []
    for region in df_avg["uitp_region"]:
        avg_value = df_avg[df_avg["uitp_region"] == region]["new_track_length"].values[
            0
        ]

        for year in years:
            new_row = {
                "year": year,
                "uitp_region": region,
                "new_track_length": avg_value,
            }
            new_rows.append(new_row)

    # Convert new rows into a DataFrame
    new_rows_df = pd.DataFrame(new_rows)

    # Append the new rows to the original df
    df_updated = pd.concat([df, new_rows_df], ignore_index=True).sort_values(
        by=["uitp_region", "year"]
    )

    return df_updated


def estimate_track_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function adds a new column which multiplies the average cost per km of track (from tcp_transit_costs.py) and the
    new_track_length to get the total cost of track per year by region (stored in "new_track_track_costs").
    """

    # Store columns
    cols = list(df.columns)

    # Read in track costs data
    track_cost = pd.read_csv(config.Paths.output / "regional_cost_track_per_km.csv")

    # rename year column
    track_cost = track_cost.rename(columns={"distributed_year": "year"})

    # merge on year and region
    df_merged = pd.merge(
        left=df, right=track_cost, on=["year", "uitp_region"], how="left"
    )

    # Multiply new track by average cost
    df_merged["new_track_track_costs"] = (
        df_merged["new_track_length"] * df_merged["cost_per_km_distributed"]
    )

    # remove calc columns
    cols = cols + ["new_track_track_costs"]
    df_merged = df_merged.filter(items=cols, axis=1)

    return df_merged


def _fill_gaps_with_global_average(
    df: pd.DataFrame, track_cost: pd.DataFrame
) -> pd.DataFrame:
    """
    EDIT: This function now fills missing rolling stock costs with the lowest cost value from other regions. Global min
    added to the code in the rolling stock costs file (Quick fix. Will do a more efficient fix to this later...)

    ORIGINAL: Function fills gaps in average rolling costs per km for regions with a global average. Global average is
    unweighted, giving all regions equal weight towards the average regardless of the number of projects within that
    region. Adds two columns, one to identify which rows use the global average ('rolling_stock_cost_desc'), and one for
    the average cost of rolling stock per km of new track ('cost_per_km_distributed').
    """

    # create global average table with just year and value
    global_avg = track_cost.loc[lambda d: d.uitp_region == "global min"]
    global_avg = global_avg.filter(
        items=("year", "cost_per_km_distributed"), axis=1
    ).rename(columns={"cost_per_km_distributed": "global_min"})

    # merge into merged dataset
    df_with_avg = df.merge(global_avg, on=["year"], how="left")

    # Fill blanks with global average (adding column to mark which ones are averages)
    df_with_avg["rolling_stock_cost_desc"] = np.where(
        df_with_avg["cost_per_km_distributed"].isna(), "global min", "regional"
    )
    df_with_avg["cost_per_km_distributed"] = np.where(
        df_with_avg["cost_per_km_distributed"].isna(),
        df_with_avg["global_min"],
        df_with_avg["cost_per_km_distributed"],
    )

    return df_with_avg


def estimate_rolling_stock_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function adds a new column which multiplies the average rolling stock cost per km of track
    (from tcp_rolling_stock_costs.py) and the new_track_length to get the total cost of rolling stock per year by region
    (stored in "new_track_track_costs").
    """

    # Read in track costs data
    track_cost = pd.read_csv(config.Paths.output / "regional_cars_cost_per_km.csv")

    # rename year column
    track_cost = track_cost.rename(columns={"distributed_year": "year"})

    # merge on year and region
    df_merged = pd.merge(
        left=df, right=track_cost, on=["year", "uitp_region"], how="left"
    )

    # Fill blanks with global average
    df_merged = _fill_gaps_with_global_average(df_merged, track_cost)

    # Multiply new track by average cost
    df_merged["new_track_rolling_stock_costs"] = (
        df_merged["new_track_length"] * df_merged["cost_per_km_distributed"]
    )

    # remove calc columns
    cols = [
        "year",
        "uitp_region",
        "rolling_stock_cost_desc",
        "new_track_length",
        "new_track_track_costs",
        "new_track_rolling_stock_costs",
    ]
    df_merged = df_merged.filter(items=cols, axis=1)

    return df_merged


def metro_costs_pipeline() -> pd.DataFrame:
    """
    Pipeline function to create dataframe with total cost of metro by year and region.
    """

    # import carriage data
    df = read_uitp_track_length_data().pipe(clean_uitp)

    # Calculate additional track by year and region
    df = calculate_new_track_length(df)

    # extrapolate annual growth to 2021-2023
    years_to_extrapolate = [2021, 2022, 2023]
    df = estimate_future_years(df, years=years_to_extrapolate)

    # calculate cost of new track (both track costs and rolling stock costs)
    df = estimate_track_costs(df)
    df = estimate_rolling_stock_costs(df)

    # Calculate total cost of metro by year and region
    df["value_USDm"] = df["new_track_track_costs"] + df["new_track_rolling_stock_costs"]

    # export as csv
    df.to_csv(config.Paths.output / "metro_costs.csv", index=False)

    return df


if __name__ == "__main__":
    df = metro_costs_pipeline()
