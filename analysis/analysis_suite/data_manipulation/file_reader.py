import pandas as pd


def read_csv_file(filename, file_location=""):
    """
    Reads in a CSV file with Pandas and
    returns a DataFrame.
    """
    return pd.read_csv(file_location + filename)
