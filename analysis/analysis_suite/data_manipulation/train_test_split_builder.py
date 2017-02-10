from sklearn.model_selection import train_test_split
import numpy as np


class TrainTestSplitBuilder(object):

    def split_data(self):
        """
        Splits the data so you either have a random train test split,
        or you have an evenly distributed train test split for available categories
        This option is set by the use_column_distribution_split_if_needed flag.
        :return: x_train, x_test, y_train, y_test
        """
        if self.even_distribution:
            return self.__get_train_test_split_without_even_distribution__()
        else:
            return self.__get_train_test_split_with_even_distribution__()

    def use_column_distribution_split_if_needed(self, flag=True):
        """
        Sets whether or not to produce even distributions of classifications when
        splitting the training data.
        :param flag: boolean
        :return: self
        """
        self.even_distribution = flag
        return self

    def with_data_frame(self, data_frame, predicting_column):
        """
        Specifies the data frame to use when performing the train test split
        :param predicting_column: the column that has the binary category within it
        :param data_frame: DataFrame
        :return: self
        """
        self.data_frame = data_frame
        self.predicting_column = predicting_column
        return self

    def __get_train_test_split_without_even_distribution__(self):
        """
        Gets a train test split of the data frame without looking at the
        predicting column distribution of groups
        :return: x_train, x_test, y_train, y_test
        """
        y = self.data_frame[self.predicting_column]
        x = self.data_frame.drop(self.predicting_column, axis=1)
        return train_test_split(x, y, test_size=0.4, random_state=101)

    def __get_train_test_split_with_even_distribution__(self):
        """
        Performs a train test split on a data structure so you are left with
        an event split for the classes you are trying to predict.
        It randomly selects records to allow this to happen.
        :return: x_train, x_test, y_train, y_test
        """
        possible_categories = self.data_frame[self.predicting_column].unique()
        minimum_category_count = self.__get_minimum_category_count_for_even_split__(possible_categories)
        chosen_indices_by_category = self.__get_indices_by_category__(possible_categories, minimum_category_count)
        all_categories_indices = np.concatenate(chosen_indices_by_category)
        data_frame_subset = self.data_frame.iloc[all_categories_indices, :]

        self.__print_even_distributed_data_frame_info__(possible_categories, data_frame_subset)

        y = data_frame_subset[self.predicting_column]
        x = data_frame_subset.drop(self.predicting_column, axis=1)
        return train_test_split(x, y, test_size=0.4, random_state=101)

    def __get_minimum_category_count_for_even_split__(self, categories):
        """
        Gets the count of the category with the minimum amount of entries.
        :param categories: array of possible categories
        :return: int
        """
        minimum_count = 0

        for category in categories:
            rows_found = self.data_frame[self.data_frame[self.predicting_column] == category]
            count_rows_found = len(rows_found)

            if count_rows_found < minimum_count or minimum_count == 0:
                minimum_count = count_rows_found
        return minimum_count

    def __get_indices_by_category__(self, categories, minimum_count):
        """
        Gets an array, where each element is an array of indices where the predicting column is
        one of the available categories
        :param categories: the categories that are available to predict
        :param minimum_count: the amount of rows for each category we should include
        :return: array of array[int]
        """
        chosen_indices_by_category = []
        for category in categories:
            category_indices = self.data_frame[self.data_frame[self.predicting_column] == category].index
            random_indices = np.random.choice(category_indices, minimum_count, replace=False)
            category_indices_array = np.array(random_indices)
            chosen_indices_by_category.append(category_indices_array)
        return chosen_indices_by_category

    def __print_even_distributed_data_frame_info__(self, categories, under_sample_data):
        """
        Prints the total number of rows used to get an even split of data
        :param categories:  the categories that are available
        :param under_sample_data: the data that will be used in the even split
        :return: null
        """
        for category in categories:
            print("Percentage of category in data: ",
                  len(under_sample_data[under_sample_data[self.predicting_column] == category]) / len(
                      under_sample_data))
        print("Total number of transactions used: ", len(under_sample_data))


