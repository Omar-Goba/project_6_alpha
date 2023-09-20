from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
        sets the data
        args:
            df: pandas dataframe
        returns:
            pandas dataframe
    """
    ### drop the useless columns ###
    df.drop(columns="CustomerID", inplace=True)

    ### rename the columns ###
    df.rename(columns={
        "DateOfPurchase": "date", "ProductCategory": "category", 
        "TransactionAmount": "value", "Next30DaysPurchase": "y"}, inplace=True)

    ### format the dtype for each ###
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = df["category"].astype("category")
    df["value"] = df["value"].astype(np.float32)
    df["y"] = df["y"].astype(np.int8)

    ### return the dataframe ###
    return df


def get_data(path: str) -> pd.DataFrame:
    """
        laods the data into a pandas dataframe
        args:
            path: path to the data
        returns:
            pandas dataframe
    """
    ### load data ###
    df = pd.read_csv(path)
    breakpoint()

    ### return the dataframe ###
    return df

def main(show_pb: bool = True) -> int:
    """
        loads the data and renames the columns and sets the dtypes
        and saves it as a parquet file (to save space)
        args:
            show_pb: show the progress bar
        returns:
            0 if success
    """
    ### init usefull var ###
    inp_path: str = "./dbs/raw/db.csv"
    out_path: str = "./dbs/intermittent/db.parquet"

    ### init the progress bar ###
    if (show_pb):   pbar = tqdm(total=3)

    ### load the data into a pandas dataframe ###
    if (show_pb):   pbar.set_description("Loading data")
    df = get_data(inp_path)
    if (show_pb):   pbar.update()

    ### set the dtypes ###
    if (show_pb):   pbar.set_description("Setting dtypes")
    df = set_dtypes(df)
    if (show_pb):   pbar.update()

    ### save the dataframe ###
    if (show_pb):   pbar.set_description("Saving dataframe")
    df.to_parquet(out_path)
    if (show_pb):   pbar.update()

    ### close the progress bar ###
    if (show_pb):   pbar.set_description("Done")
    if (show_pb):   pbar.close()

    ### return 0 ###
    return 0

if (__name__ == "__main__"):    main()
#   __   _,_ /_ __, 
# _(_/__(_/_/_)(_/(_ 
#  _/_              
# (/                 
