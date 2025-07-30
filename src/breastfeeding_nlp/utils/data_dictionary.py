# # -*- coding: utf-8 -*-
# """
# Created on Sat Oct 07, 2023 12:58:50
# Modified on Mon Sep 23, 2024 09:00:39
# Making a data dictionary: First iteration
# @author: connorgrannis
# """

# # %% Import packages
# import pandas as pd
# import argparse

# # %% Functions and classes
# class CreateDataDictionary:
#     """Code (mostly) from https://github.com/p-easter/Python_DataDictionary/blob/main/README.md"""
#     def __init__(self):
#         '''This class provides functions to quickly develop a data dictionary for your data set'''
#         return None
#     def make_my_data_dictionary(self, dataFrame):
#         '''Create an initial data dictionary excluding definitions for meaning of features'''
#         col_ = dataFrame.columns
#         df_DataDict = {}
#         # loop through all columns and extract metadata
#         for col in col_:
#                 df_DataDict[col] = {
#                                'Type': str(dataFrame.dtypes[col]),
#                                'Number of observations': len(dataFrame[col]),
#                                "Number of valid observations": len(dataFrame[col]) - sum(dataFrame[col].isna()),
#                                'Number of missing observations': sum(dataFrame[col].isna()),
#                                'Size (bytes)': dataFrame.memory_usage()[col],
#                                'Definition': str(''),
#                                'Notes': str(''),
#                                'Possible Values': str(''),
#                                'Required?': str(''),
#                                'Accepts Null Values?': str(''),
#                                'Alignment to project goals': str('')
#                                 }
#         # convert to dataframe
#         df_DD = pd.DataFrame(df_DataDict)
#         return df_DD
#     def define_data_meaning(self, df_data_dictionary, add_def=True):
#         '''Quickly provide input regarding each columns meaning and transpose into a usable dictionary'''
#         col_ = df_data_dictionary.columns
#         d = 'Definition'    # adding a new column. Just a shortcut/nickname
#         # iteratively and interactively provide data definitions
#         for col in col_:
#             if add_def:
#                 df_data_dictionary[col][d] = input('Provide a data definition for {}'.format(col))
#             else:
#                 df_data_dictionary[col][d] = ''
#         # formatting
#         df_data_dictionary = df_data_dictionary.transpose()
#         return df_data_dictionary
#     def update_dd_definition(self, df_data_dictionary, attribute):
#         try:
#             df_dd = df_data_dictionary.transpose()
#             df_dd[attribute]['Definition'] = input('Provide a data definition for {}'.format(attribute))
#             df_dd = df_dd.transpose()
#             return df_dd
#         except:
#             print('Sorry, there was an error.  Check attribute name and try again')

# def make_chunked_data_dictionary(df, add_def):
#     # initialize object
#     dd = CreateDataDictionary()
#     # create initial data dictionary
#     df_dd = dd.make_my_data_dictionary(df)
#     # provide data definitions and adjust formatting
#     df_dd = dd.define_data_meaning(df_dd, add_def=add_def)
#     return df_dd


# # %% Run from command line
# if __name__ == "__main__":
#     # get the path to the input data
#     parser = argparse.ArgumentParser(description='Process data dictionary path.')
#     parser.add_argument('input_data_path', type=str, help='Path to the input data')
#     parser.add_argument('output_data_path', type=str, help='Path to the output data')
#     args = parser.parse_args()
#     input_data_path = args.input_data_path
#     output_data_path = args.output_data_path
#     # read the input data
#     input_data = pd.read_csv(input_data_path)
#     # make the data dictionary
#     dd = make_chunked_data_dictionary(input_data, add_def=False)
#     # save the data dictionary
#     dd.reset_index().to_csv(output_data_path, index=False)


import pandas as pd
from typing import Any

class CreateDataDictionary:
    """
    A class for creating a data dictionary from a pandas DataFrame.
    Provides functions to quickly generate metadata about your dataset.
    """
    def __init__(self) -> None:
        """Initialize the data dictionary object."""
        pass

    def make_my_data_dictionary(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        """
        Create an initial data dictionary with metadata for each column.

        Parameters:
            dataFrame (pd.DataFrame): The input DataFrame for which to create the data dictionary.

        Returns:
            pd.DataFrame: A DataFrame representing the data dictionary with metadata.
        """
        df_DataDict: dict[str, dict[str, Any]] = {}
        for col in dataFrame.columns:
            # Determine possible values
            unique_vals = dataFrame[col].unique()
            if len(unique_vals) <= 10:
                possible_values = ', '.join(map(str, unique_vals))
            else:
                sample_vals = ', '.join(map(str, unique_vals[:3]))
                possible_values = f"{sample_vals}, ..."
                
            df_DataDict[col] = {
                'Type': str(dataFrame.dtypes[col]),
                'Number of observations': len(dataFrame[col]),
                'Number of valid observations': dataFrame[col].count(),
                'Number of missing observations': dataFrame[col].isna().sum(),
                'Size (bytes)': dataFrame.memory_usage()[col],
                'Number of unique values': dataFrame[col].nunique(),
                'Definition': '',
                'Notes': '',
                'Possible Values': possible_values,
                'Required?': 'Yes' if dataFrame[col].isna().sum() == 0 else 'No',
                'Accepts Null Values?': 'Yes' if dataFrame[col].isna().sum() > 0 else 'No',
                'Alignment to project goals': ''
            }
        return pd.DataFrame(df_DataDict)
    
    def define_data_meaning(self, df_data_dictionary: pd.DataFrame, add_def: bool = True) -> pd.DataFrame:
        """
        Interactively provide data definitions for each column in the data dictionary.
        After collecting definitions, the DataFrame is transposed for easier viewing.

        Parameters:
            df_data_dictionary (pd.DataFrame): The initial data dictionary DataFrame.
            add_def (bool): If True, prompts the user to input definitions interactively.
                            If False, clears definitions.

        Returns:
            pd.DataFrame: The transposed data dictionary with updated definitions.
        """
        # Iterate through each column and update the 'Definition' field
        for col in df_data_dictionary.columns:
            definition = input(f'Provide a data definition for {col}: ') if add_def else ''
            df_data_dictionary.at['Definition', col] = definition
        return df_data_dictionary.transpose()

    def update_dd_definition(self, df_data_dictionary: pd.DataFrame, attribute: str) -> pd.DataFrame:
        """
        Update the definition for a specific attribute in the data dictionary.

        Parameters:
            df_data_dictionary (pd.DataFrame): The data dictionary DataFrame.
            attribute (str): The attribute (column) for which to update the definition.

        Returns:
            pd.DataFrame: The updated data dictionary.
        """
        try:
            # Ensure the DataFrame is transposed with attributes as the index
            if attribute not in df_data_dictionary.index:
                df_data_dictionary = df_data_dictionary.transpose()
            new_def = input(f'Provide a data definition for {attribute}: ')
            df_data_dictionary.at[attribute, 'Definition'] = new_def
            return df_data_dictionary
        except Exception as e:
            print(f'Error updating definition for {attribute}: {e}')
            return df_data_dictionary

def make_chunked_data_dictionary(df: pd.DataFrame, add_def: bool = True) -> pd.DataFrame:
    """
    Create a complete data dictionary for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        add_def (bool): Whether to interactively add definitions for each column.

    Returns:
        pd.DataFrame: The complete data dictionary.
    """
    dd = CreateDataDictionary()
    df_dd = dd.make_my_data_dictionary(df)
    df_dd = dd.define_data_meaning(df_dd, add_def=add_def)
    return df_dd