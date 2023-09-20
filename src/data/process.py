from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os, warnings


def get_data(path: str) -> pd.DataFrame:
    """
        laods the data into a pandas dataframe
        args:
            path: path to the data
        returns:
            pandas dataframe
    """
    ### load data ###
    df = pd.read_parquet(path)

    ### return the dataframe ###
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        adds the date features to the dataframe
        args:
            df: pandas dataframe
        returns:
            pandas dataframe
    """
    ### add the date features ###
    df["dt_year"] = df["date"].dt.year
    df["dt_month"] = df["date"].dt.month
    df["dt_day"] = df["date"].dt.day
    df["dt_dayofweek"] = df["date"].dt.dayofweek
    df["dt_dayofyear"] = df["date"].dt.dayofyear
    df["dt_quarter"] = df["date"].dt.quarter
    df["dt_is_month_start"] = df["date"].dt.is_month_start
    df["dt_is_month_end"] = df["date"].dt.is_month_end
    df["dt_is_weekend"] = df["date"].dt.weekday.isin([5, 6])

    ### return the dataframe ###
    return df


def apply_kmeans(df: pd.DataFrame, x: str, y: str, n_clusters: int) -> pd.DataFrame:
    """
        applies a kmeans clustering on the dataframe
        on the y column after aggregating the x column
        args:
            df: pandas dataframe
            x: column to aggregate
            y: column to cluster
            n_clusters: number of clusters
        returns:
            pandas dataframe
    """
    ### init column name dynamicly ###
    cluster_name = f"cluster_{x}_{y}"

    ### aggregate the data ###
    db = df.groupby(x).agg({y: "sum"}).reset_index()
    
    ### apply the kmeans clustering ###
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(db[[x, y]])
    db[cluster_name] = kmeans.labels_

    ### merge the data ###
    df = df.merge(db[[x, cluster_name]], on=x, how="left")

    ### return the dataframe ###
    return df


def apply_kmeans_on_all(df: pd.DataFrame) -> pd.DataFrame:
    """
        applies a kmeans clustering on the dataframe
        on the y column after aggregating the x column
        args:
            df: pandas dataframe
        returns:
            pandas dataframe
    """
    ### Apply a k mean clustering on the temporal access for y ###
    df = apply_kmeans(df, "dt_day", "y", df.dt_day.nunique() // 4)
    df = apply_kmeans(df, "dt_dayofyear", "y", df.dt_dayofyear.nunique() // 4)
    df = apply_kmeans(df, "dt_month", "y", df.dt_month.nunique() // 4)

    ### Apply a k mean clustering on the category access for y ###
    df = apply_kmeans(df, "sparse_category", "y", df.category.nunique() // 2)

    ### apply a k mean clustering on the quantized value access for y ###
    df = apply_kmeans(df, "quantized_value", "y", df.quantized_value.nunique() // 2)

    return df


def show_kmeans_clustering(tr: pd.DataFrame) -> None:
    for col in list(filter(lambda x: x.startswith("cluster_"), tr.columns)):
        print(f"plotting {col}")
        plt.bar(tr[col.replace("cluster_", "")], tr[col])
        plt.show()


def apply_deviation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
        applies a deviation analysis on the dataframe
        The algorithm is as follows:
            1. Apply a k mean clustering on the category column for the value
                - This done to cluster all categories based on the value spent on it
            2. replace the labels with the avreage value spent
            3. take the diffrenece between the expected value spent on each catigory
                with the actual value spent
            4. apply once more kmeans clustering to transform the domain from 
                continuous to discrete
        args:
            df: pandas dataframe
        returns:
            pandas dataframe
    """
    ### Apply a k mean clustering on the category column for the value ###
    df = apply_kmeans(df, "sparse_category", "value", df.category.nunique() // 2)

    ### Replace cluster labels by the mean of the value ###
    df["cluster_sparse_category_value"] = df.groupby("cluster_sparse_category_value").value.transform("mean")

    ### Calculate the deviation ###
    df["deviation"] = df["value"] - df["cluster_sparse_category_value"]

    ### Apply a k mean clustering on the deviation for the y ###
    df = apply_kmeans(df, "deviation", "y", df.deviation.nunique() // 2)

    return df


def apply_spectral_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
        get the avreage customer retention on a daily bases aggregated by monthly
        ie. the avreage `y` for every day of the month accross the whole database
        then apply a fast fourier transform to it and calculate the magnitude and 
        phase of the resultant wave.
        args:
            df: pd.DataFrame
        returns:
            df: pd.DataFrame
    """
    ### copy database ###
    db = df.copy()

    ### aggregate the data on daily while filling missing values with 0 ###
    db = db.set_index("date").resample("d").y.sum()
    db = db.reset_index()
    db.date = db.date.dt.day
    db = db.groupby("date").sum().reset_index()

    ### compute the fast fourier transform ###
    day_of_month_num_y_freq = np.fft.fft(db.y)
    day_of_month_num_y_mag = np.vectorize(lambda x: abs(x))(day_of_month_num_y_freq)
    day_of_month_num_y_phi = np.vectorize(lambda x: np.angle(x))(day_of_month_num_y_freq)

    db = pd.DataFrame({
        "dt_day": np.arange(1, 32),
        "mag": day_of_month_num_y_mag,
        "phi": day_of_month_num_y_phi
    })

    ### merge the data ###
    df = df.merge(db, on="dt_day", how="left")

    return df


def select_features(df_x: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
    """
        select the features to be used in the model
        by using the KBest feature selection algorithm
        it returns the indices of the selected columns
        to avoid data leakage as the algorithm is fitted
        on the training data and then applied on the test
        args:
            df_x: pd.DataFrame
            df_y: pd.DataFrame
        returns:
            cols: list
    """
    ### init the selector ###
    n_features = int(df_x.shape[1] // 1.7)
    selector = SelectKBest(f_regression, k=n_features)

    ### fit the selector ###
    selector.fit(df_x.values, df_y.values)

    ### get the selected columns ###
    cols = selector.get_support(indices=True)

    return cols


def main(show_pb: bool = True):
    """
    """
    ### QoL ###
    warnings.filterwarnings("ignore")

    ### init the progress bar ###
    if show_pb: pb = tqdm(total=15)

    ### Init usefull var ###
    if show_pb: pb.set_description("Init usefull var")
    inp_path: str = "./dbs/intermittent/db.parquet"
    out_path: str = ["./dbs/cooked/tr.npz", "./dbs/cooked/ts.npz"]
    train_test_split: float = 0.8
    if show_pb: pb.update()

    ### Load the data ###
    if show_pb: pb.set_description("Load the data")
    df = get_data(inp_path)
    if show_pb: pb.update()

    ### Split the data to train and test to avoid data leakage ###
    if show_pb: pb.set_description("Split the data to train and test to avoid data leakage")
    train_test_split_index = int(df.shape[0] * train_test_split)
    tr = df.iloc[:train_test_split_index]
    ts = df.iloc[train_test_split_index:]
    if show_pb: pb.update()

    ### Add the date features ###
    if show_pb: pb.set_description("Add the date features")
    tr = add_date_features(tr)
    ts = add_date_features(ts)
    if show_pb: pb.update()

    ### Do sparse encoding on the category column ###
    if show_pb: pb.set_description("Do sparse encoding on the category column")
    tr["sparse_category"] = tr["category"].astype("category").cat.codes
    ts["sparse_category"] = ts["category"].astype("category").cat.codes
    if show_pb: pb.update()

    ### Quantize the values ###
    if show_pb: pb.set_description("Quantize the values")
    tr["quantized_value"] = pd.qcut(tr["value"], 10, labels=False)
    ts["quantized_value"] = pd.qcut(ts["value"], 10, labels=False)
    if show_pb: pb.update()

    ### Add Cluster Features ###
    if show_pb: pb.set_description("Add Cluster Features")
    tr = apply_kmeans_on_all(tr)
    ts = apply_kmeans_on_all(ts)
    if show_pb: pb.update()

    ### Deviation Analysis ###
    if show_pb: pb.set_description("Deviation Analysis")
    tr = apply_deviation_analysis(tr)
    ts = apply_deviation_analysis(ts)
    if show_pb: pb.update()

    ### Spectral Analysis ###
    if show_pb: pb.set_description("Spectral Analysis")
    tr = apply_spectral_analysis(tr)
    ts = apply_spectral_analysis(ts)
    if show_pb: pb.update()

    ### drop unneeded columns ###
    if show_pb: pb.set_description("drop unneeded columns")
    tr.drop(columns=["date", "category"], inplace=True)
    ts.drop(columns=["date", "category"], inplace=True)
    if show_pb: pb.update()

    ### split X and y ###
    if show_pb: pb.set_description("split X and y")
    tr_y = tr["y"]
    tr_x = tr.drop(columns=["y"])
    ts_y = ts["y"]
    ts_x = ts.drop(columns=["y"])
    if show_pb: pb.update()

    ### select features ###
    if show_pb: pb.set_description("select features")
    cols = select_features(tr_x, tr_y) # get the selected columns to avoid data leakage
    tr_x = tr_x.iloc[:, cols]
    ts_x = ts_x.iloc[:, cols]
    if show_pb: pb.update()

    ### apply PCA for dimensional reduction ###
    if show_pb: pb.set_description("apply PCA for dimensional reduction")
    num_components = int(tr_x.shape[1] // 1.7)
    pca = PCA(n_components=num_components)
    tr_x = pd.DataFrame(pca.fit_transform(tr_x))
    ts_x = pd.DataFrame(pca.transform(ts_x)) # no fit to avoid data leakage
    if show_pb: pb.update()

    ### convert to numpy ###
    if show_pb: pb.set_description("convert to numpy")
    tr_x = tr_x.values
    tr_y = tr_y.values
    ts_x = ts_x.values
    ts_y = ts_y.values
    if show_pb: pb.update()

    ### save the data ###
    if show_pb: pb.set_description("save the data")
    np.savez(out_path[0], x=tr_x, y=tr_y)
    np.savez(out_path[1], x=ts_x, y=ts_y)
    if show_pb: pb.update()
    
    ### close the progress bar ###
    if show_pb: pb.set_description("Done")
    if show_pb: pb.close()

    return 0


if (__name__ == "__main__"):    main()
#   __   _,_ /_ __, 
# _(_/__(_/_/_)(_/(_ 
#  _/_              
# (/                 
