import pandas as pd
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def load_dataset(id):
    # fetch dataset
    dataset = fetch_ucirepo(id=id)

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets
    dataset_name = dataset.metadata.name.replace(" ", "_").replace("'", "").lower()
    return X, y, dataset_name


def label_encoding(X, y):
    df = join_df(X, y)
    non_numeric_columns = list(df.select_dtypes(include=["object", "string"]).columns)
    encoders = {column_name: LabelEncoder() for column_name in non_numeric_columns}
    for column_name, le in encoders.items():
        le.fit(df[column_name])
        df[column_name] = le.transform(df[column_name])
    X, y = split_df(df)
    return X, y, encoders


def join_df(X, y):
    df = X
    df["y"] = y
    return df


def split_df(df):
    X = df.drop("y", axis=1)
    y = df["y"]
    return X, y


def load_and_prepare(id_dataset):
    X, y, dataset_name = load_dataset(id_dataset)
    inputer = SimpleImputer(strategy="most_frequent")
    X = pd.DataFrame(inputer.fit_transform(X), columns=X.columns)
    X, y, encoders = label_encoding(X, y)
    return X, y, encoders, dataset_name
