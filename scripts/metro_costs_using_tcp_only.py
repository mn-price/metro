import pandas as pd

from scripts import config



def tcp_cost_pipeline() -> pd.DataFrame:
    """
    Pipeline function. Creates a pd.DataFrame of average cost per km by UITP region for all years with data beyond 2010.
    Exports this pd.DataFrame as a csv to the output folder.

    return: pd.DataFrame of average regional costs of track per km by uitp region.
    """
    # Read in metro track and rolling stock costs
    track_costs = pd.read_csv(config.Paths.output / "regional_cost_track_per_km.csv")
    rolling_stock_costs = pd.read_csv(config.Paths.output / "regional_cars_cost_per_km.csv")



    return rolling_stock_costs


if __name__ == "__main__":
    data = tcp_cost_pipeline()