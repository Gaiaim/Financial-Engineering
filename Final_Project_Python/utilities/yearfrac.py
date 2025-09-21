from datetime import date
from enum import Enum
import pandas as pd
from typing import Iterable, Union, List, Tuple
import datetime as dt
# Define an enumeration for different day count conventions
class mod(Enum):
    ACT_360 = 2    # Actual/360 day count convention
    ACT_365 = 3    # Actual/365 day count convention
    EU_30_360 = 6  # European 30/360 day count convention

def yearfrac(start_date: Union[List[dt.datetime], pd.DatetimeIndex, dt.date], 
             end_date: Union[List[dt.datetime], pd.DatetimeIndex, dt.date],
             convention: mod) -> float:
    """
    Computes the fraction of a year between two dates using different day count conventions.
    This function replicates MATLAB's yearfrac function for ACT/360, ACT/365, and EU 30/360.

    :param start_date: The starting date as a datetime.date object.
    :param end_date: The ending date as a datetime.date object.
    :param convention: The day count convention to use (must be a value from the Conv Enum).
    
    :return: The fraction of the year between the two dates as a float.
    
    :raises ValueError: If an unsupported convention is provided.
    """

    # Actual/360: Calculates the fraction using a 360-day year assumption
    if convention == mod.ACT_360:
        return (end_date - start_date).days / 360.0

    # Actual/365: Calculates the fraction using a 365-day year assumption
    elif convention == mod.ACT_365:
        return (end_date - start_date).days / 365.0
    


    # European 30/360: Assumes each month has 30 days and a full year has 360 days
    elif convention == mod.EU_30_360:
        # Extract day, month, and year components of both dates
        d1, m1, y1 = start_date.day, start_date.month, start_date.year
        d2, m2, y2 = end_date.day, end_date.month, end_date.year

        # Apply the 30/360 European rule:
        # If the first date's day is greater than 30, set it to 30
        d1 = min(d1, 30)

        # If the first date's day is 30 and the second date's day is 31, set the second date's day to 30
        # Otherwise, ensure the second date's day does not exceed 30
        d2 = 30 if (d1 == 30 and d2 == 31) else min(d2, 30)

        # Compute the year fraction using the 30/360 formula:
        # (Years difference * 360) + (Months difference * 30) + (Days difference) / 360
        return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0

    # If the provided convention is not supported, raise an error
    else:
        raise ValueError("Unsupported convention")
