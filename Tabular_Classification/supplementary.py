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
