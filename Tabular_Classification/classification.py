# --------
# IMPORTS
# --------
import re
import os
import pandas as pd
import numpy as np

# F1
def with_explore(func):
    def wrapper(self, *args, **kwargs):

        # Create DataFrame
        dataframe = func(self, *args, **kwargs)

        # Post Condition Check
        if self.train_df is None or self.test_df is None:
            raise RuntimeError("DataFrames not available for exploration")

        # Basic Exploration
        self.explorer = BasicExplore(tr_df=self.train_df)
        self.explorer.run()

        # Duplicate Check
        self.has_dupl = self.explorer.has_duplicates

        # Dtype Lists
        self.integer_list = self.explorer.int_pred_lst
        self.object_list = self.explorer.obj_pred_lst
        self.float_list = self.explorer.flt_pred_lst

        print("Lists of Predictors based on their dtypes: ", "\n")
        print(self.integer_list)
        print(self.object_list)
        print(self.float_list)

        self.multi_record = CheckMultiRecord(df=self.train_df,
                                                pred_lst=self.object_list)
        self.multi_record.run()

        # Multi-Record Predictors
        self.multi_record_preds = self.multi_record.multi_record_pred
        if len(self.multi_record_preds) != 0:
            print("List of Predictors with Multiple Records: ", end="\n")
            print(self.multi_record_preds)

        else:
            print("There is NO presence of Predictors having Multiple Record for a single record.")

        return dataframe
    return wrapper

# F2
def with_preprocess(func):
    def wrapper(self, *args, **kwargs):

        # Create DataFrame
        dataframe = func(self, *args, **kwargs)

        # Precondition
        if self.train_df is None:
            raise RuntimeError("Training DataFrame NOT available for preprocessing")

        if not hasattr(self, "explorer"):
            raise RuntimeError("Exploration MUST run before preprocessing")

        # Drop Duplicates
        dupl = getattr(self, "has_dupl")
        if dupl:
            self.train_df.drop_duplicates(keep='last', inplace=True, ignore_index=True)

        # Basic Preprocessing
        self.preprocess = BasicPreprocess(df=self.train_df)
        self.preprocess.run()

        # Updated DataFrame
        self.preprocessed_df = self.preprocess.df

        return self.preprocessed_df
    return wrapper


# C1
class CreateDF:
    """
    This class performs the following tasks:
        1. Scans the project Directory for the Data Files to Create Pandas DataFrames
    """

    def __init__(self, ext_options=(".csv", ".xlsx")):
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
class BasicExplore:
    """
    This class performs the following operations:
        1. Checks the presence of duplicated rows
        2. Checks for the presence of Predictors with Missing Values
        3. Checks for the presence of Predictors having multiple values for a single record
        4. Categorize predictors on the basis of dtype
    """

    def __init__(self, tr_df):
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

    def show_categories(self, *lists, default=None):
        """
        :return: Non-Empty List(s) based on dtype. Target Variable will be excluded from the consideration
        """
        non_empty = (lst for lst in lists if len(lst) > 0)
        return next(non_empty, default)

    def run(self):
        self.establish_target_idx()
        self.check_dupl_rows()
        self.check_na()
        self.dtype_categorize()
        return self

class BasicPreprocess:
    """
    This class performs the following operations:
        1. Lowers the predictor names.
        2. Collects the missing variable with their proportions
        3. Drop the un-wanted columns
    """
    def __init__(self, df):
        self.df = df

        # Missing Variables
        self.miss_vars = []
        self.miss_vars_prop = dict()

        # Threshold Missing Variables
        self.thr_miss_vars = []
        self.rej_miss_vars = []

    def lower_pred_names(self):
        self.df.columns = self.df.columns.str.lower()

    def collect_miss_vars(self):
        for ind, row in enumerate(self.df.isnull().sum()):
            if row != 0:
                self.miss_vars.append(self.df.columns[ind])
                self.miss_vars_prop.update({self.df.columns[ind]: np.round((row / len(self.df)) * 100, 2)})

    def threshold_miss_vars(self, threshold_val: float = 0.15):
        """
        Thresholding Missing Variables based on the missing variables proportion
        :param threshold_val: The cut-off proportion for the predictor to be omitted or considered

        :return: A tuple containing the names of the predictors that need to be considered for feature engineering and
        the ones that need to be rejected
        """
        for k in self.miss_vars_prop.keys():
            if self.miss_vars_prop[k] <= threshold_val:
                self.thr_miss_vars.append(k)
            else:
                self.rej_miss_vars.append(k)

    def run(self):
        self.lower_pred_names()
        self.collect_miss_vars()
        if len(self.miss_vars) != 0:
            self.threshold_miss_vars()
        if len(self.rej_miss_vars) != 0:
            self.df.drop(columns=self.rej_miss_vars, inplace=True)

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

    @staticmethod
    def multi_record(*lists, default=None):
        non_empty = (lst for lst in lists if len(lst) > 0)
        return next(non_empty, default)

    def run(self):
        self.multi_record_w_comma()

        self.multi_record_pred = CheckMultiRecord(self.split_w_comma,
                                                  self.split_w_spaces)
        return self