# --------
# IMPORTS
# --------
import os
import re
import pandas as pd
import numpy as np

from utils import supplementary as sp

# F1
def with_explore(func):
    def wrapper(self, *args, **kwargs):
        # Create DataFrame
        result = func(self, *args, **kwargs)

        # Post Condition Check
        if self.train_df is None or self.test_df is None:
            raise RuntimeError("DataFrames not available for exploration")

        # Run Exploration
        self.explorer = BasicExplore(
            tr_df=self.train_df,
            tst_df=self.test_df
        )
        self.explorer.run()

        # Dtype Exploration
        self.obj_explorer = sp.CheckMultiRecord(
            df=self.train_df, pred_lst=self.explorer.obj_pred_lst
        )
        self.obj_explorer.run()

        # Multi-Record Check
        multi_rec = getattr(self,"multi_record_pred")
        if multi_rec is None:
            print("No Predictor with Multi-Record Value")
        else:
            print(multi_rec)

        return result
    return wrapper

# F2
def with_preprocess(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        # Precondition
        if self.train_df is None:
            raise RuntimeError("Training DataFrame NOT available for preprocessing")

        if not hasattr(self, "explorer"):
            raise RuntimeError("Exploration MUST run before preprocessing")

        # Run Pre-processing
        self.preprocessor = BasicPreprocessing(
            df=self.train_df,
            duplicate_pres=self.explorer.has_duplicates,
            nan_pres=self.explorer.has_nan_pred
        )
        self.preprocessor.run()

        # Update DataFrame
        self.preprocessed_df = self.train_df

        return self.preprocessed_df
    return wrapper

# C1
class CreateDF:
    """
    This class performs the following tasks:
        1. Scans the project Directory for the Data Files to Create Pandas DataFrames
    """

    def __init__(self,  ext_options=(".csv", ".xlsx")):
        self.ext_options = ext_options
        self.files = []
        self.unq_file_ext = []

        self.sub_f_pres = 0
        self.sub_f_name = None
        self.train_f_name = None
        self.test_f_name = None

        self.train_df = None
        self.test_df = None
        self.sub_df = None

    def scan_directory(self, path='.'):
        """
        Scans current working directory for files with allowed extensions.
        """
        self.files = [
            f for f in os.listdir(path) if f.endswith(self.ext_options)
        ]

        # Scan Detection
        for f in self.files:
            if re.findall("submission", f, re.IGNORECASE):
                self.sub_f_pres = 1
                self.sub_f_name = f

        # Unique Extensions Detection
        self.unq_file_ext = list(
            {ext for f in self.files for ext in self.ext_options if f.endswith(ext)}
        )

        # Detect Train/Test Filenames
        for f in self.files:
            if re.findall("train", f, re.IGNORECASE):
                self.train_f_name = f
            elif re.findall("test", f, re.IGNORECASE):
                self.test_f_name = f

    def create_dataframes(self):
        """
        Load Train, Test (and Submission if Available)
        """
        if not self.unq_file_ext:
            raise RuntimeError("No recognized file extensions found.")

        # Pick Extension - Assuming only 1
        ext = self.unq_file_ext[0]

        # Extension Mapping
        loaders = {
            ".csv": pd.read_csv,
            ".xlsx": pd.read_excel
        }

        if ext not in loaders:
            raise ValueError(f"Unsupported Extension: {ext}")

        reader = loaders[ext]

        # Compulsory Assignment
        self.train_df = reader(self.train_f_name)
        self.test_df = reader(self.test_f_name)

        # Optional Assignment
        if self.sub_f_pres == 1:
            self.sub_df = reader(self.sub_f_name)
            print("Submission File is also present")

        print(f"Dataframes Loaded Successfully!!")

    def sanity_check(self):
        """
        Ensures Train Dataframe is Loaded correctly.
        """
        if self.train_df is None or self.train_df.empty:
            raise Exception("DataFrame Assignment was NOT successful")

        print("Sanity Check passed: train_df is loaded")

    @with_preprocess
    @with_explore
    def run(self, path='.'):
        """
        Full Pipeline: Scan, Load, and Check
        """
        self.scan_directory(path)
        self.create_dataframes()
        self.sanity_check()
        return self

# C2
class BasicExplore():
    """
    This class performs the following operations:
        1. Checks the presence of duplicated rows
        2. Checks for the presence of Predictors with Missing Values
        3. Checks for the presence of Predictors having multiple values for a single record
        4. Categorize predictors on the basis of dtype
    """
    def __init__(self, tr_df, tst_df):
        # DataFrames
        self.train_df = tr_df
        # Exploration
        self.has_duplicates: bool = False
        self.target_variable_idx: int = 0
        self.has_nan_pred: bool = False
        # Dtype Categorized Lists
        self.int_pred_lst = []
        self.obj_pred_lst = []
        self.flt_pred_lst = []

    def establish_target_idx(self):
        print(list(zip(range(len(self.train_df)), self.train_df.columns)))
        target_variable_index = int(input("Enter the index of the Target Variable: "))
        self.target_variable_idx = target_variable_index

    def check_dupl_rows(self):
        dupl_val = self.train_df.duplicated().sum()
        if dupl_val != 0:
            self.has_duplicates = True
            print('There is  presence of duplicated rows in the dataset')

        else:
            print('Dataset is free of Duplicated rows.')

    def check_na(self):
        nan_present = len(self.train_df.columns[self.train_df.isnull().any()].tolist())
        if nan_present != 0:
            self.has_nan_pred = True
            print("DataFrame contains NaN Values")
        else:
            print("DataFrame is FREE of NaN Values")

    def dtype_categorize(self):
        """
        A function to categorize predictors on the basis of their dytpes.
        :return: N-number of lists depending on the number of dtypes to be categorized
        """
        int_types = ['int8', 'int32', 'int64']
        flt_types = ['float32', 'float64']
        obj_types = ['object']

        cols = self.train_df.columns

        for col in cols:
            if self.train_df[col].dtypes in int_types:
                self.int_pred_lst.append(col)

            elif self.train_df[col].dtypes in obj_types:
                self.obj_pred_lst.append(col)

            elif self.train_df[col].dtypes in flt_types:
                self.flt_pred_lst.append(col)

    def run(self):
        self.establish_target_idx()
        self.check_dupl_rows()
        self.check_na()
        self.dtype_categorize()
        return self

class BasicPreprocessing:
    def __init__(self, df,  duplicate_pres, nan_pres):
        self.df = df
        self.dupl = duplicate_pres
        # Missing Variables
        self.nan_pres = nan_pres
        self.miss_vars = []
        self.miss_vars_prop = dict()

    def lower_pred_names(self):
        self.df.columns = self.df.columns.str.lower()

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()

    def deal_w_miss_vars(self):
        for ind, row in enumerate(self.df.isnull().sum()):
            if row != 0:
                self.miss_vars.append(self.df.columns[ind])
                self.miss_vars_prop.update({self.df.columns[ind]: np.round((row / len(self.df)) * 100, 2)})

    def run(self):
        self.lower_pred_names()
        if self.dupl:
            b4_removal = self.df.shape[0]
            print(f"Number of Rows before Deletion: {b4_removal}")
            self.remove_duplicates()
            af8_removal = self.df.shape[0]
            print(f"Number of Rows after Deletion: {af8_removal}")
            print(f"Number of Rows Deleted: {af8_removal-b4_removal}")
        if self.nan_pres:
            self.deal_w_miss_vars()

        return self.df
