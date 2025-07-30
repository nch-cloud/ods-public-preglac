import re

def clean_up_text(text: str, num_whitespace: int = 2) -> str: 
    """
    Clean up the input text by stripping whitespace, replacing HTML entities for quotes, and converting sequences of whitespace (based on num_whitespace) into newlines.

    Args:
        text (str): The input string to clean and normalize.
        num_whitespace (int): The exact number of consecutive whitespace characters to replace with a newline. Defaults to 2.

    Returns:
        str: The cleaned and normalized text.
    """
    # Remove leading and trailing whitespace
    text = text.strip()
    # Replace HTML entity &apos; with single quote
    text = text.replace("&apos;", "'")
    # Replace HTML entity &quot; with double quote
    text = text.replace("&quot;", '"')
    # Replace HTML entities for greater than and less than
    text = text.replace("&gt;", '>')
    text = text.replace("&lt;", '<')
    # Convert num_whitespace whitespace characters into a newline with a period to force a boundary
    text = re.sub(rf'\s{{{num_whitespace}}}', '.\n', text)
    # Remove lines that are JUST a period
    text = text.replace("\n.\n", "\n\n")
    text = text.replace("\n\n.\n", "\n\n\n")
    text = text.replace("\n\n.\n\n", "\n\n\n")
    # remove periods after colons
    text = text.replace(":.", ":")
    # remove new lines after colons
    text = text.replace(":\n", ": ")
    # add a space after all colons ONLY if not already followed by a space
    text = re.sub(r":(?! )", ": ", text)
    # replace multiple periods with a single period
    text = text.replace("..", ".")
    text = text.replace("...", ".")
    text = text.replace("..\n", ".\n")
    # remove periods after commas
    text = text.replace(",.", ",")
    # remove periods before commas
    text = text.replace(".,", ",")
    # remove newlines following commas
    text = text.replace(",\n", ', ')
    # replace question mark period with question mark
    text = text.replace("?.", "?")
    # Replace upside-down question marks with a tab, hyphen, and space.
    text = text.replace(" ¿ ", "\t- ")
    text = text.replace("¿ ", "\t- ")
    # If a new line starts with a SINGLE space, strip it
    text = text.replace("\n ", "\n")
    # Return the cleaned text
    return text