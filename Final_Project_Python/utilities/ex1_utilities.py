"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
import calendar
from scipy.stats import norm

#from bootstrap import bootstrap
#from readExcelData import readExcelData
from typing import Iterable, Union, List, Tuple
from utilities.yearfrac import yearfrac, mod

# Define an enumeration for the two types of swaptions
class SwapType(Enum):
    """
    Types of swaptions.
    """
    RECEIVER = "receiver"  # Option to receive fixed rate payments
    PAYER = "payer"        # Option to pay fixed rate payments


def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex, List[dt.date]],
    discount_factors: Iterable[float],
) -> List[float]:
    """
    Compute the zero rates from the discount factors.

    Parameters:
        dates (Union[List[float], pd.DatetimeIndex, List[dt.date]]): List of year fractions or dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        List[float]: List of zero rates.
    """
    #discount_factors = list(discount_factors)

    if isinstance(dates, pd.DatetimeIndex):
        base_date = dates[0]
        # year fractions ESCLUDENDO la prima data (t=0)
        year_fractions = [
            yearfrac(base_date, d, mod.ACT_365) for d in dates[1:]
        ]
        #discount_factors = discount_factors[1:]  # scarta t=0
    else:
        year_fractions = dates

    year_fractions = np.array(year_fractions, dtype=float)
    discount_factors = discount_factors.astype(np.float32)

    return (-np.log(discount_factors) / year_fractions).tolist()




def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp, dt.date],
    interp_date: Union[dt.datetime, pd.Timestamp, dt.date],
    dates: Union[List[dt.datetime], pd.DatetimeIndex, List[float], np.ndarray],
    discount_factors: Iterable[float],
) -> float:
    # Controllo che dates e discount_factors abbiano la stessa lunghezza
    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")

    # Se dates sono date o DatetimeIndex, calcola year fractions
    if isinstance(dates[0], (np.datetime64, dt.datetime, pd.Timestamp, dt.date)):
        # Converti tutte le date in datetime.datetime
        dates = [pd.to_datetime(d).to_pydatetime() for d in dates]
        year_fractions = [yearfrac(reference_date, T, mod.ACT_365) for T in dates[1:]]
        discount_factors_interp = discount_factors[1:]
    else:
        # Altrimenti assumiamo che siano già year fractions
        year_fractions = dates[1:]
        discount_factors_interp = discount_factors[1:]

    # Calcola la year fraction per la data di interpolazione
    inter_year_frac = yearfrac(reference_date, interp_date, mod.ACT_365)

    # Interpola il tasso zero lineare sui year fractions
    zero_rates = from_discount_factors_to_zero_rates(year_fractions, discount_factors_interp)
    rate = np.interp(inter_year_frac, year_fractions, zero_rates)

    return np.exp(-inter_year_frac * rate)



def business_date_offset(
    base_date: Union[dt.date, pd.Timestamp],
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> Union[dt.date, pd.Timestamp]:
    """
    Return the closest following business date to a reference date after applying the specified offset.

    Parameters:
        base_date (Union[dt.date, pd.Timestamp]): The starting date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Adjusted date moved to the closest following business day if needed.
    """
    # Adjust the year and month by converting the offset months into years and months
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Try to adjust the day; if the day is invalid (e.g., Feb 30), use the last valid day of the month
    day = base_date.day
    try:
        adjusted_date = base_date.replace(year=year, month=month, day=day) + dt.timedelta(days=day_offset)
    except ValueError:
        # Determine the last day of the month
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(year=year, month=month, day=last_day_of_month) + dt.timedelta(days=day_offset)

    # If the adjusted date falls on a weekend, shift it to the next business day
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date


def date_series(
    t0: Union[dt.date, pd.Timestamp], t1: Union[dt.date, pd.Timestamp], freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Generate a list of dates from t0 to t1 inclusive with a specified frequency (number of dates per year).

    Parameters:
        t0 (Union[dt.date, pd.Timestamp]): Start date.
        t1 (Union[dt.date, pd.Timestamp]): End date.
        freq (int): Number of dates per year.

    Returns:
        List of dates from t0 to t1.
    """
    # Start the series with the initial date
    dates = [t0]
    # Continue generating dates using business_date_offset until t1 is reached or exceeded
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // freq))
    # Remove any date that overshoots t1
    if dates[-1] > t1:
        dates.pop()
    # Ensure the final date is exactly t1
    if dates[-1] != t1:
        dates.append(t1)

    return dates
