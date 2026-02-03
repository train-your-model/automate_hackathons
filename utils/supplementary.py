# ------------------
# IMPORTS
# ------------------

import os
import shutil
import numpy as np
import yaml
from pathlib import Path


# F1
def create_root_dir():
    """
    Creates the root directory in the parent directory where the hackathon specific dirs would
    be stored.
    """
    func_dir = Path(__file__).resolve().parents[1]
    y_path = os.path.join(func_dir, "files", "config.yaml")

    with open(y_path, 'r') as y:
        data = yaml.safe_load(y)

        par_dir = data["defaults"]["parent_dir"]
        root_dir_name = str(input("Enter the name for the root directory: "))
        full_path = os.path.join(par_dir, root_dir_name)
        os.mkdir(full_path)
        print("Root Directory has been created successfully!!")

        # Appending the Root Directory Path to the YAML file
        data.setdefault("defaults", {})["root_dir_path"] = full_path
        with open(y_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        print("YAML file has been updated with the ROOT Directory")

# F2


# C1
class DealDataFolders:
    """
    This class performs the following tasks:
        1. Locate the Zip Folder in the Project Directory
        2. Store the directory contents before and after unzipping the data folder
        3. Extracts the contents of the unzipped folder into the project directory and then the folder is removed.
    """
    def __init__(self, proj_dir_path=None, default_folder=None):
        self.b4_unzip_cont = []  # List of Project Directory Contents before unzipping
        self.af8_unzip_cont = []  # List of Project Directory Contents after unzipping

        self.proj_path = proj_dir_path
        self.default_folder = default_folder

        self.unzip_folder_path = None

    def get_proj_dir_contents(self, unzipped=0):
        if unzipped == 0:
            self.b4_unzip_cont.extend(os.listdir(self.proj_path))

        elif unzipped == 1:
            self.af8_unzip_cont.extend(os.listdir(self.proj_path))
            unzip_folder_name = list(map(str, set(self.af8_unzip_cont).difference(set(self.b4_unzip_cont))))
            unzip_full_path = os.path.join(self.proj_path, unzip_folder_name[0])
            self.unzip_folder_path = unzip_full_path

    def extract_from_unzip(self):
        for f in os.listdir(self.unzip_folder_path):
            f_path = os.path.join(self.proj_path, f)
            shutil.move(f_path, self.proj_path)

        # Remove the Now Empty Unzipped Folder
        os.rmdir(self.unzip_folder_path)

    def run(self, in_proj_dir = 0):
        if in_proj_dir == 1:
            self.get_proj_dir_contents()
            self.extract_from_unzip()

        else:
            ...

# C2
class CheckMultiRecord:

    def __init__(self, df, pred_lst: list):
        self.df = df
        self.pred_lst = pred_lst
        # Separator Based splits
        self.split_w_comma = []
        self.split_w_spaces = []
        self.multi_record_pred = []

    def multi_record_w_comma(self):
        for col in self.pred_lst:
            split_row_max = max(self.df[col].dropna().apply(lambda x: len(x.split(','))))
            if split_row_max > 1:
                self.split_w_comma.append(col)

    def multi_record(self, *lists, default=None):
        non_empty = (lst for lst in lists if len(lst) > 0)
        return next(non_empty, default)

    def run(self):
        self.multi_record_w_comma()

        self.multi_record_pred = self.multi_record(self.split_w_comma,
                                                   self.split_w_spaces)
        return self

class HandleNumbers:
    """
    Handles preprocessing related to Integer / Float dtypes
    """
    def __init__(self, df, int_dtype: list, flt_dtype: list):
        self.df = df
        self.int_dtype = int_dtype
        self.flt_dtype = flt_dtype
        self.predictors_with_outliers = []

    def check_outliers(self):
        """
        Checks the presence of outliers in numeric predictors.
        Outliers will be determined using the IQR technique

        :return: A class list containing the names of the predictors with outliers
        """
        for col in self.int_dtype + self.flt_dtype:
            q1, q3 = np.percentile(np.array(self.df[col]), [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (iqr * 1.5)
            upper_bound = q3 + (iqr * 1.5)
            outlier_sum = sum(np.where((np.array(self.df[col]) > upper_bound) | (np.array(self.df[col]) < lower_bound),
                                       1, 0))
            if outlier_sum != 0:
                self.predictors_with_outliers.append(col)
