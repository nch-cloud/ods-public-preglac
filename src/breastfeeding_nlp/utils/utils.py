import re
import time
import smart_open
import pandas as pd
from typing import List

class StopWatch:
    def __init__(self):
        self.start_time = None  # When the stopwatch was started
        self.stop_time = None   # When the stopwatch was stopped
        self.running = False    # Is the stopwatch currently running
        self.laps = []          # List to store lap times
    def start(self):
        """Start the stopwatch."""
        if not self.running:
            if self.start_time is None:
                self.start_time = time.time()
            else:
                # Adjust start time if restarting without resetting
                self.start_time += (time.time() - self.stop_time)
            self.running = True
    def stop(self):
        """Stop the stopwatch."""
        if self.running:
            self.stop_time = time.time()
            self.running = False
    def reset(self):
        """Reset the stopwatch to zero and clear lap records."""
        self.start_time = None
        self.stop_time = None
        self.running = False
        self.laps.clear()
    def elapsed_time(self):
        """Return the total elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.running:
            return time.time() - self.start_time
        else:
            return self.stop_time - self.start_time
    def lap(self):
        """Record a lap and return the lap time."""
        if not self.running:
            print("Stopwatch is not running. Cannot record lap.")
            return None
        lap_time = self.elapsed_time()
        self.laps.append(lap_time)
        return lap_time
    def get_laps(self):
        """Return the list of all recorded lap times."""
        return self.laps
    def is_running(self):
        """Return whether the stopwatch is currently running."""
        return self.running
    
def load_dataframe(data, **kwargs):
    """
    Read data from a DataFrame or a file path (local or S3) and return a DataFrame.

    Args:
        data (pd.DataFrame or str): Either a DataFrame or a path to a file.
            - Supported file formats: .csv, .tsv, .parquet, .json
            - File paths can be local paths or S3 locations (starting with 's3://').
        **kwargs: Additional keyword arguments to pass to pandas file readers.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the file format is unsupported or the input type is invalid.
        FileNotFoundError: If the specified file path does not exist (for local files).
    """
    if isinstance(data, pd.DataFrame):
        # If it's already a DataFrame, return it as-is
        return data

    elif isinstance(data, str):
        # Infer file format from file extension
        file_format = data.split(".")[-1].lower()

        # Use smart_open to handle S3 or local paths
        try:
            with smart_open.open(data, "rb") as file:
                if file_format == "csv":
                    return pd.read_csv(file, **kwargs)
                elif file_format == "tsv":
                    return pd.read_csv(file, sep="\t", **kwargs)
                elif file_format == "parquet":
                    return pd.read_parquet(file, **kwargs)
                elif file_format == "json":
                    return pd.read_json(file, **kwargs)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            raise ValueError(f"Error loading file: {data}. Details: {e}")

    else:
        raise ValueError("Input must be a pandas DataFrame or a string path to a file.")


def save_dataframe(df, output_path, **kwargs):
    """
    Save a DataFrame to a specified file path (local or S3) in CSV, Parquet, or JSON format.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): Destination file path (local or S3).
            - Supported formats: .csv, .tsv, .parquet, .json
            - If an S3 path is provided, it will be saved to S3.
        **kwargs: Additional keyword arguments to pass to the Pandas writer.

    Raises:
        ValueError: If the file format is unsupported.
        Exception: If saving fails.
    """
    file_format = output_path.split(".")[-1].lower()

    try:
        with smart_open.open(output_path, "wb") as file:
            if file_format == "csv":
                df.to_csv(file, index=False, **kwargs)
            elif file_format == "tsv":
                df.to_csv(file, sep="\t", index=False, **kwargs)
            elif file_format == "parquet":
                df.to_parquet(file, index=False, **kwargs)
            elif file_format == "json":
                df.to_json(file, orient="records", lines=True, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        print(f"Data successfully saved to {output_path}")
    except Exception as e:
        raise ValueError(f"Error saving file to {output_path}. Details: {e}")


class Xlator(dict):
    """ Pronounced translator. All-in-one multiple-string-substitution class """
    def _make_regex(self):
        """ Build re object based on the keys of the current dictionary """
        return re.compile("|".join(map(re.escape, self.keys(  ))))
    def __call__(self, match):
        """ Handler invoked for each regex match """
        return self[match.group(0)]
    def xlat(self, text):
        """ Translate text, returns the modified text. """
        return self._make_regex(  ).sub(self, text)


class OptimizedDataFrame(pd.DataFrame):
    """
    Code is mostly taken from https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
    """
    def optimize_floats(self) -> pd.DataFrame:
        """ downcast float64 to as small as possible """
        floats = self.select_dtypes(include=['float64']).columns.tolist()
        self[floats] = self[floats].apply(pd.to_numeric, downcast='float')
        return self
    def optimize_ints(self) -> pd.DataFrame:
        """ downcast int64 to as small as possible """
        ints = self.select_dtypes(include=['int64']).columns.tolist()
        self[ints] = self[ints].apply(pd.to_numeric, downcast='integer')
        return self
    def optimize_objects(self, datetime_features: List[str]) -> pd.DataFrame:
        """ downcast objects like datetimes and characters to as small as possible """
        for col in self.select_dtypes(include=['object']):
            if col not in datetime_features:
                if not (type(self[col].iloc[0])==list):
                    num_unique_values = len(self[col].unique())
                    num_total_values = len(self[col])
                    if float(num_unique_values) / num_total_values < 0.5:
                        # get rid of hyphens
                        try:
                            self[col] = list(map(lambda x: Xlator({"-": " "}).xlat(x), self[col]))
                        except TypeError:
                            pass
                        self[col] = self[col].astype('category')
            else:
                self[col] = pd.to_datetime(self[col])
        return self
    def _optimize(self, datetime_features: List[str] = []) -> pd.DataFrame:
        """ optimize all columns """
        return self.optimize_floats().optimize_ints().optimize_objects(datetime_features)
    def optimize(self, datetime_features: List[str] = [], show_mem_reduction=False) -> pd.DataFrame:
        """ optimize everything and report the memory reduction """
        orig_mem = self.memory_usage(deep=True).sum()
        opt_df = self._optimize(datetime_features)
        opt_mem = opt_df.memory_usage(deep=True).sum()
        if show_mem_reduction:
            print(f"Memory usage reduced by {round(((orig_mem-opt_mem)/orig_mem)*100, 2)}%")
        return opt_df
    def count_values_table(self, col):
        """ displays the value, count, and % of total """
        count_val = self[col].value_counts()
        count_val_percent = 100 * count_val / len(self[col])
        count_val_table = pd.concat([count_val, count_val_percent.round(2)], axis=1)
        count_val_table.reset_index(inplace=True)
        # reset the column names
        count_val_table.columns = range(count_val_table.columns.size)
        count_val_table_ren_columns = count_val_table.rename(
                            columns = {0: 'Value', 1 : 'Count Values', 2 : '% of Total Values'}
        )
        return count_val_table_ren_columns
    def missing_values_table(self):
        mis_val = self.isnull().sum()
        mis_val_percent = 100 * self.isnull().sum() / len(self)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(self.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
        return mis_val_table_ren_columns
    def show_categorical_counts(self, cat_vars):
        print("Total Records:", self.shape[0], "- Value Counts per Categorical Attribute:\n")
        # todo: make a single dataframe with a multi-index
        for att in cat_vars:
            print(att)
            print((pd.DataFrame(self[att]
                                .value_counts(normalize = False, sort = True, ascending = False, bins = None, dropna = False))
                                .rename(columns={att:"Count"})
                    ))
            print("\n")

def filter_dataset(df, keep_split_info=False):
    # Drop the rows that are missing notes
    df.dropna(subset='NOTE_TEXT', inplace=True)

    # WIC request forms
    wic_ids = df.query("NOTE_TEXT.str.contains('Ohio WIC Prescribed Formula and Food Request Form')").row_ix

    # note types
    filter_out_note_types = ["Patient Instructions", "Discharge Instructions", "MR AVS Snapshot", "ED AVS Snapshot", 
                             "IP AVS Snapshot", "Training", "Operative Report", "D/C Planning", "Pharmacy"]

    # Filter
    cols_to_keep = ["PAT_ID", "BF1", "BF2", "NOTE_TYPE", "NOTE_TEXT", "row_ix"]
    if keep_split_info:
        cols_to_keep.append("split")
    df = df[cols_to_keep].copy()
    red_df = df[~df["NOTE_TYPE"].isin(filter_out_note_types)]
    return red_df.query("row_ix not in @wic_ids.tolist()")