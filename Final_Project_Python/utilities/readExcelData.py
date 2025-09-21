import pandas as pd
import os
from dataclasses import dataclass

def readExcelData(file_name="MktData_CurveBootstrap.xls"):
    """
    Load market data from an Excel file and return structured data classes.
    
    Returns:
        dates_set (DatesSet): Containing settle date, depos, future, and swap dates.
        rates_set (RatesSet): Containing depos, future, and swap rates.
    """
    # Get the current directory and file path
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_name}' was not found in the current directory!")

    # Load the Excel file
    df = pd.read_excel(file_path, engine="xlrd")
    print("File found and loaded successfully!")

    # Extract settlement date
    row_num = 7
    column_name = df.columns[4]
    settlement_date = df.at[row_num - 1, column_name]

    # Extract Depo dates
    start_row, end_row = 10, 15
    column_name = df.columns[3]
    Depo_dates = df.loc[start_row - 1:end_row - 1, column_name].to_frame(name="Settle Dates")

    # Extract Future dates
    start_row, end_row = 11, 19
    column1, column2 = df.columns[16], df.columns[17]
    Future_dates = df.loc[start_row - 1:end_row - 1, [column1, column2]]
    Future_dates.columns = ["Settle", "Expiry"]

    # Extract Swap dates
    start_row, end_row = 38, 55
    column_name = df.columns[3]
    Swap_dates = df.loc[start_row - 1:end_row - 1, column_name].to_frame(name="Swap Dates")

    # Extract Depo rates
    start_row, end_row = 10, 15
    column1, column2 = df.columns[7], df.columns[8]
    depo_rate = df.loc[start_row - 1:end_row - 1, [column1, column2]]
    depo_rate["Mean"] = depo_rate.mean(axis=1)
    depo_rate.columns = ["Bid", "Ask", "Mid"]

    # Extract Future rates
    start_row, end_row = 27, 35
    column1, column2 = df.columns[7], df.columns[8]
    future_rate = df.loc[start_row - 1:end_row - 1, [column1, column2]]
    future_rate["Mean"] = future_rate.mean(axis=1)
    future_rate.columns = ["Bid", "Ask", "Mid"]

    # Extract Swap rates
    start_row, end_row = 38, 54
    column1, column2 = df.columns[7], df.columns[8]
    swap_rate = df.loc[start_row - 1:end_row - 1, [column1, column2]]
    swap_rate["Mean"] = swap_rate.mean(axis=1)
    swap_rate.columns = ["Bid", "Ask", "Mid"]

    # Define dataclasses
    @dataclass
    class DatesSet:
        settle: str
        depos: pd.DataFrame
        future: pd.DataFrame
        swap: pd.DataFrame

    @dataclass
    class RatesSet:
        depos: pd.DataFrame
        future: pd.DataFrame
        swap: pd.DataFrame

    # Create objects for structured data
    dates_set = DatesSet(
        settle=settlement_date,
        depos=Depo_dates,
        future=Future_dates,
        swap=Swap_dates
    )

    rates_set = RatesSet(
        depos=depo_rate,
        future=future_rate,
        swap=swap_rate
    )

    return dates_set, rates_set