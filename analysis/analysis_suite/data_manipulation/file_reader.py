import pandas as pd


def read_csv_file(filename):
    """
    Reads in a CSV file with Pandas and
    returns a DataFrame.
    """
    return pd.read_csv(filename)
