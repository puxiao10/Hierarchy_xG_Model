"""
this file is constructed of small functions that used for xG model evaluation
"""
import pandas as pd

def import_epl_data(
    start_year: int,
    end_year: int,
    path_dir: str,
    prefix: str = "shots_epl_"
) -> pd.DataFrame:
    """import_epl_data is a function that imports the shots data from the csv files. 
    The csv files are expected to be named as "shots_epl_{season}.csv" (eg. "shots_epl_17-18.csv") and so on.

    Args:
        start_year (int): the start year of the first season. e.g. 2017-2018 season, the start year is 2017.
        end_year (int): the end year of the last season. e.g. 2017-2018 season, the end year is 2018.
        path_dir (str): the path of the directory that contains the csv files.
        prefix (str, optional): the prefix of the csv files' names. Defaults to "shots_epl_".

    Returns:
        pd.DataFrame: the concatenated dataframe of all the csv files of EPL shots from the start year to the end year.
    """
    df_list = []
    for yr in range(start_year, end_year):
        season = f"{yr % 100}-{yr % 100 + 1}" # get the season
        filename = f"{path_dir}{prefix}{season}.csv" # get the filename
        df_temp = pd.read_csv(filename) # read the csv file of that season
        df_temp["season"] = season # add the season column
        df_list.append(df_temp) # append the dataframe to the list
    return(pd.concat(df_list).reset_index(drop=True)) # return the concatenated dataframe

def evaluate(
    pred,
    y,
    threshold = 0.5
):
    
    # print out AUC / recall / precision

    # print out confusion matrix

    pass

def compare_scatter(
    pred,
    y,
    xG
):
    # draw the scatter plot

    pass