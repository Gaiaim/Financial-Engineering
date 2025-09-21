import pandas as pd
from datetime import date, timedelta
from enum import Enum

class mod(Enum):
    Normal = "normal"
    Modified = "modified"

def is_business_day(date_obj: date) -> bool:
    """Check if a date is a business day (Monday to Friday)."""
    return date_obj.weekday() < 5

def adjust_to_business_day(date_obj: date, mod: mod) -> date:
    """Adjust the date to the nearest business day based on the mod type."""
    while not is_business_day(date_obj):
        if mod == mod.Normal:
            date_obj -= timedelta(days=1)  # Move to the previous business day
        elif mod == mod.Modified:
            date_obj += timedelta(days=1)  # Move to the next business day
    return date_obj

def add_Dates(start_date: date, years: int, mod: mod) -> pd.DataFrame:
    """
    Generates a pandas DataFrame with annual dates from the start_date up to n years,
    adjusting them to business days based on the given modification rule.

    :param start_date: The initial date (included in the output).
    :param years: The number of years to generate annual dates.
    :param mod: The modification rule (normal or modified) for adjusting to business days.
    :return: A pandas DataFrame containing the generated dates.
    """
    dates = [start_date.replace(year=start_date.year + i) for i in range(years + 1)]
    adjusted_dates = [adjust_to_business_day(d, mod) for d in dates]
    df = pd.DataFrame(adjusted_dates, columns=["Business Adjusted Dates"])
    return df

# Example usage
if __name__ == "__main__":
    start_date = date(2023, 2, 2)
    years = 50
    df = add_Dates(start_date, years, mod.Modified)
    print(df)
