import os
import pandas as pd
from smart_open import open as sopen
from joblib import Memory

# Define the cache directory for joblib
cache_dir = "./.cache"
memory = Memory(cache_dir, verbose=0)


@memory.cache
def read_file_from_s3(s3_path, bucket_name, key):
    with sopen(s3_path, "r") as file:
        return file.read()


def load_notes_to_dataframe(
    directory,
    file_extension=".txt",
    column_name="note_text",
    prefix=None,
    ignore_cache=False,
):
    """
    Load notes from a directory or S3 path into a DataFrame with optional caching.

    Args:
        directory (str): Path to the directory or S3 location containing the files.
        file_extension (str): File extension to filter files (e.g., ".txt").
        column_name (str): Name of the column for file content in the output DataFrame.
        prefix (str, optional): If provided, only include files whose filenames start with this prefix.
        ignore_cache (bool): If True, bypass the cache and read files directly.

    Returns:
        pd.DataFrame: A DataFrame with 'note_id' and file content in the specified column.
    """
    rows = []

    # Handle S3 paths
    if directory.startswith("s3://"):
        import boto3

        s3 = boto3.client("s3")
        bucket_name, prefix_path = directory[5:].split("/", 1)
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix_path)

        if "Contents" in objects:
            for obj in objects["Contents"]:
                key = obj["Key"]
                filename = os.path.basename(key)
                if key.endswith(file_extension) and (
                    not prefix or filename.startswith(prefix)
                ):
                    s3_path = f"s3://{bucket_name}/{key}"
                    if ignore_cache:
                        # Bypass cache and fetch directly
                        with sopen(s3_path, "r") as file:
                            content = file.read()
                    else:
                        # Use cache to fetch the file
                        content = read_file_from_s3(s3_path, bucket_name, key)
                    rows.append({"note_id": filename, column_name: content})

    # Handle local directories
    else:
        for filename in os.listdir(directory):
            if filename.endswith(file_extension) and (
                not prefix or filename.startswith(prefix)
            ):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as file:
                    content = file.read()
                rows.append({"note_id": filename, column_name: content})

    return pd.DataFrame(rows)
