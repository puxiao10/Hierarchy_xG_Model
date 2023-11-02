"""
this file is constructed of small functions that used for xG model evaluation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

class DATASET:
    def __init__(self, X, y, random_state = None):
        self.X = X
        self.y = y
        self.X_over, self.y_over = RandomOverSampler(random_state = random_state).fit_resample(self.X, self.y)
        self.X_under, self.y_under = RandomUnderSampler(random_state = random_state).fit_resample(self.X, self.y)
        
    def fit(self, model, sampling = "original"):
        # check if sampling is valid
        if sampling not in ["original", "over", "under"]:
            raise ValueError("sampling must be 'original', 'over', or 'under'")
        # fit model
        if sampling == "original":
            model.fit(self.X, self.y)
        elif sampling == "over":
            model.fit(self.X_over, self.y_over)
        elif sampling == "under":
            model.fit(self.X_under, self.y_under)
        # return the fitted model
        return model

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
    input: np.ndarray, 
    target: np.ndarray, 
    threshold: int = 0.5, 
    verbose: bool = True, 
    cm: bool = True
):
    """evaluate() is a function that evaluates the performance of the xG model
    by printing out accuracy, precision, recall, f1 and auc (if verbose = True), 
    and confusion matrix (if cm = true).

    Args:
        input (np.ndarray): probability of the model's prediction.
        target (np.ndarray): the true label of the data.
        threshold (int, optional): the threshold of the model's prediction. Defaults to 0.5.
        verbose (bool, optional): if True, print out accuracy, precision, recall, f1, and auc. Defaults to True.
        cm (bool, optional): if True, print out the confusion matrix. Defaults to True.
    """
    pred = np.where(input > threshold, 1, 0)
    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred)
    recall = recall_score(target, pred)
    f1 = f1_score(target, pred)
    auc = roc_auc_score(target, input)
    
    if verbose:
        print(f"accuracy: {accuracy:.3f}")
        print(f"precision: {precision:.3f}")
        print(f"recall: {recall:.3f}")
        print(f"f1: {f1:3f}")
        print(f"auc: {auc:3f}")
    
    if cm:
        matrix = confusion_matrix(target, pred)
        display_labels = ["Miss", "Goal"]
        sns.heatmap(matrix, annot=True, fmt="d", xticklabels=display_labels, yticklabels=display_labels)
        plt.show()

def compare_scatter(
    pred,
    y,
    xG
):
    # draw the scatter plot

    pass