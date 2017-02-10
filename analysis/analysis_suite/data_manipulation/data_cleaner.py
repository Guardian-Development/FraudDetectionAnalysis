from sklearn.preprocessing import StandardScaler


def scale_column_to_range(data_frame, column_name):
    """
    Scales a column within a data frame
    :param data_frame: data frame containing the column
    :param column_name: string column name
    :return: data frame with scaled column
    """
    data_frame[column_name] = \
        StandardScaler().fit_transform(data_frame[column_name].values.reshape(-1, 1))
    return data_frame
