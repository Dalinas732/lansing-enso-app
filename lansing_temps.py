import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Temperature_Data:
    """
    Class for collecting, validating, and analyzing temperature data 
    across multiple years and months using daily records from 1960 to now.
    
    Assumes input is a dictionary of pandas DataFrames with each key as a year
    and each DataFrame containing 36 columns: 3 per month (high, average, low).
    """
    
    def __init__(self, dfs):
        """
        Initializes the Temperature_Data object with given dictionary of DataFrames.

        Parameters:
        dfs (dict): Keys are years (str), values are pandas DataFrames 
                    with daily temperature data organized as:
                    [high1, avg1, low1, high2, avg2, low2, ..., high12, avg12, low12]
        """
        years = []
        for i in dfs.keys():  # Store all years as strings in a list
            years.append(i)
            
        self.years = years
        self.dfs = dfs
        self.tag = False  # Flag to avoid recursive calls to _fix_zeros()

    def _check_zeros(self, data):
        """
        Checks if a month's temperature array contains any zero entries.

        Parameters:
        data (np.ndarray): Array of low temperatures for a month.

        Returns:
        bool: True if a zero is found; False otherwise.
        """
        for i in range(data.size - 1):
            if data[i] == 0:
                return True
        return False

    def _fix_zeros(self, frame, year):
        """
        Replaces zero values in the 'low' column with 2*avg - high,
        only for days where low == 0 and data is valid.

        Parameters:
        frame (pd.DataFrame): DataFrame for the given year.
        year (str): Year of the data.
        """
        months = np.arange(1, 13)  # Loop over all months

        for month in months:
            self.tag = True  # Important to avoid recursive _fix_zeros

            lows = self.collect_month(month, year, "low")
            self.tag = True
            avgs = self.collect_month(month, year, "average")
            self.tag = True
            highs = self.collect_month(month, year, "high")

            self.check_length_of_month(lows.size, month, year)
            self.check_length_of_month(avgs.size, month, year)
            self.check_length_of_month(highs.size, month, year)

            if self._check_zeros(lows):
                for i in range(lows.size):
                    if lows[i] == 0:
                        newlow = round(2 * avgs[i] - highs[i], 3)
                        lows[i] = newlow
                        assert lows[i] == newlow, f"The low was not changed for month={month}, index={i}."

            pd_lows = np.append(lows, [float(np.nan)] * (31 - lows.size))
            frame.iloc[:, (3 * month - 1)] = pd_lows.astype(float)

    def check_length_of_month(self, count, month, year):
        """
        Ensures that the number of days in the temperature data for a given
        month matches the expected number of days.

        Parameters:
        count (int): Number of temperature records for the month.
        month (int): Month number (1–12).
        year (str): The year being checked.

        Raises:
        AssertionError: If the number of days does not match the calendar.
        """
        correct = None
        if month in {1, 3, 5, 7, 8, 10, 12}:
            correct = 31
        elif month in {4, 6, 9, 11}:
            correct = 30
        elif month == 2:
            correct = 29 if int(year) % 4 == 0 else 28
        else:
            raise ValueError(f"Month = {month} was not read between 1-12. Year = {year}.")
        
        assert count == correct, (
            f"Number of days in month = {month} year = {year}. "
            f"{count} != expected {correct}."
        )

    def collect_month(self, month, year, mode=None):
        """
        Collects all non-NaN temperature values for a specific month and year.

        Parameters:
        month (int): Month number (1–12).
        year (int or str): Year of interest.
        mode (str or int): 'high', 'average', or 'low' (or 0, 1, 2 respectively).

        Returns:
        np.ndarray: Array of valid daily temperature values.
        """
        check = False
        if isinstance(year, int):
            year = str(year)
        elif not isinstance(year, str):
            raise ValueError("Year type must be str or int.")

        if mode == "high" or mode == 0:
            index = 3 * (month - 1)
        elif mode == "average" or mode == 1:
            index = 3 * month - 2
        elif mode == "low" or mode == 2:
            index = 3 * month - 1
            check = True
        else:
            raise ValueError("Mode must be 'high', 'average', or 'low' (or 0, 1, 2).")

        df = self.dfs[year].astype(float)

        if self.tag is not True and check: # If tagged from _fix_zeroes will skip to avoid recursion
            self._fix_zeros(df, year)

        temps_array = np.array(df.iloc[:, index], dtype=float)
        temps_list = []
        day = 1

        for temp in temps_array:
            if not np.isnan(temp):
                temps_list.append(temp)
            elif np.isnan(temp) and day <= 28:
                print(f"Warning: Read NaN in month={month}, year={year}, day={day}.")
            day += 1

        self.tag = False
        return np.array(temps_list, dtype=float)

    def collect_day(self, month, day, mode=None):
        """
        Collects the temperature values for the same calendar day across all years.

        Parameters:
        month (int): Month number (1–12).
        day (int): Day of the month (1–31).
        mode (str or int): 'high', 'average', or 'low' (or 0, 1, 2).

        Returns:
        np.ndarray: Array of daily values across all years.

        Raises:
        ValueError: If a NaN is encountered in the data.
        """
        check = False
        assert day <= 31, "Input for day cannot be above 31."

        if mode == "high" or mode == 0:
            index = 3 * (month - 1)
        elif mode == "average" or mode == 1:
            index = 3 * month - 2
        elif mode == "low" or mode == 2:
            index = 3 * month - 1
            check = True
        else:
            raise ValueError("Mode must be 'high', 'average', or 'low' (or 0, 1, 2).")

        temps_list = []
        day_index = day - 1  # Adjust for 0-based indexing

        for year in self.years:
            df = self.dfs[year].astype(float)
            if check:
                self._fix_zeros(df, year)
            temp = df.iloc[day_index, index]

            if not np.isnan(temp):
                temps_list.append(temp)
            elif np.isnan(temp):
                raise ValueError(f"NaN encountered in year={year}, month={month}, day={day}.")
            else:
                raise ValueError("Data was flagged as both NaN and not NaN.")

        return np.array(temps_list, dtype=float)

    def plot_hist(self, data, title="Title", bins=None):
        """
        Plots a histogram of the provided temperature data.

        Parameters:
        data (array-like): List or array of temperature values.
        title (str): Title of the histogram.
        bins (int or array-like): Optional number or range of bins.
        """
        if bins is None:
            bins = np.linspace(np.min(data), np.max(data), np.size(data))

        plt.hist(data, bins, edgecolor='black')
        plt.xlabel('Temperature')
        plt.ylabel('# of Times')
        plt.title(title)
        plt.show()
