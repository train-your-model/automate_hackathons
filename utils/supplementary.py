# ------------------
# IMPORTS
# ------------------

import os
import shutil
import numpy as np
import yaml
from pathlib import Path
from zipfile import ZipFile


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

        self.projdir_cont: list = []
        self.data_folder: str = ""

        self.zipped_folder_path: str = ""
        self.unzip_folder_path = None

    def check_for_zip_folders(self):
        """
        At this point, the newly created project directory only consists of Data Folders.
        """
        self.b4_unzip_cont = os.listdir(self.proj_path)
        for f in self.b4_unzip_cont:
            if f.endswith('.zip'):
                self.zipped_folder_path = os.path.join(self.proj_path, f)

            else:
                self.projdir_cont.append(f)

    def check_for_data_folder(self):
        """
        Used for cases, where there is NO zipped data folders, but there is presence of a directory with data files
        """
        for f in self.projdir_cont:
            f_path = os.path.join(self.proj_path, f)
            if os.path.isdir(f_path):
                self.data_folder = f_path

    def extract_contents(self):
        print("Contents of the Project Directory: ", end="\n")
        print(f"{self.data_folder}")
        user_input = int(input("Do you want to extract contents from the dir? 1/0: "))
        # Expects Only Directory to be present
        if user_input == 1:
            for f in os.listdir(self.data_folder):
                shutil.move(os.path.join(self.proj_path, f), self.proj_path)

            # Remove Directory
            os.rmdir(self.data_folder)

    def unzip_folder(self):
        z = ZipFile(self.zipped_folder_path)
        z.extractall(path=self.proj_path)
        z.close()

    def get_proj_dir_contents(self):
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

    def run(self):
        self.check_for_zip_folders()
        if len(self.projdir_cont) != 0:
            self.check_for_data_folder()
            self.extract_contents()
        else:
            self.unzip_folder()
            self.get_proj_dir_contents()
            self.extract_from_unzip()

# C2


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
