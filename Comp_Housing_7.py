# ========================== #
# 1. Import and Copy Dataset #
# ========================== #
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class ConfigSetting:
    def __init__(self, df=None):
        self.df = df

    # def build_config_table(self):
    #     self.df = pd.DataFrame([['Directory', os.getcwd()]], columns=['Parameter', 'Value'])

    def display_data_frame(self):
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', 100)

    def main(self):
        # self.build_config_table()
        self.display_data_frame()


config = ConfigSetting()
config.main()


class ExcelIO:
    def __init__(self, path=None, raw_train=None, raw_test=None, trans_train=None, trans_test=None, submission=None):
        self.path = path
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test
        self.submission = submission

    def read_csv(self):
        self.path = os.getcwd()
        path_train, path_test = self.path + '\\' + 'train.csv', self.path + '\\' + 'test.csv'
        self.raw_train, self.raw_test = pd.read_csv(path_train), pd.read_csv(path_test)

    def copy_data_frame(self):
        self.trans_train, self.trans_test = self.raw_train.copy(), self.raw_test.copy()

    def export_csv(self):
        self.submission = pd.DataFrame(model_df.trans_test[['Id', 'SalePrice']])
        self.submission.to_csv(self.path + '\\' + 'housing_data_submission.csv', index=False)

    def main(self):
        self.read_csv()
        self.copy_data_frame()


df_set = ExcelIO()
df_set.main()


# ====================== #
# 2. Describe Statistics #
# ====================== #
class BasicStatsProvider:
    def __init__(self, raw_train=None, raw_test=None, corr=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.corr = corr

    def print_data_frame_shape(self):
        print("Raw Train", self.raw_train.shape)
        print("Raw Test", self.raw_test.shape)

    def get_correlation(self):
        self.corr = self.raw_train.corr()

    def main(self):
        self.print_data_frame_shape()
        self.get_correlation()


describe_stats = BasicStatsProvider(df_set.raw_train, df_set.raw_test)
describe_stats.main()


# =========== #
# 3. Fill NA  #
# =========== #
class Filler:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test

    def fill_na_with_val(self, field, val):
        self.trans_train[field] = self.trans_train[field].fillna(val)
        self.trans_test[field] = self.trans_test[field].fillna(val)

    def fill_na_by_regression(self):
        # Set Factors
        lot_y = self.raw_train[self.raw_train['LotFrontage'].notnull()]['LotFrontage']
        lot_x = self.raw_train[self.raw_train['LotFrontage'].notnull()][['LotArea', 'TotalBsmtSF']]

        # Model
        lot_model = XGBRegressor()
        lot_model.fit(lot_x, lot_y)

        # Prediction
        self.raw_train['LotFrontage_predictions'] = lot_model.predict(self.raw_train[['LotArea', 'TotalBsmtSF']])
        self.raw_test['LotFrontage_predictions'] = lot_model.predict(self.raw_test[['LotArea', 'TotalBsmtSF']])

        # Fill_NA
        self.trans_train['LotFrontage'] = self.trans_train['LotFrontage'].fillna(self.raw_train['LotFrontage_predictions'])
        self.trans_test['LotFrontage'] = self.trans_test['LotFrontage'].fillna(self.raw_test['LotFrontage_predictions'])

    def fill_na_with_mode(self, field):
        self.trans_train[field] = self.trans_train[field].fillna(self.trans_train[field].mode())
        self.trans_test[field] = self.trans_test[field].fillna(self.trans_test[field].mode())

    def replace_0_to_mean(self, col):
        self.trans_train[col] = self.trans_train[col].replace(0, self.trans_train[col].mean())
        self.trans_test[col] = self.trans_test[col].replace(0, self.trans_test[col].mean())

    def main(self):
        self.fill_na_with_val('GarageCond', 'TA')
        self.fill_na_with_val('GarageType', 'Attchd')
        self.fill_na_with_val('GarageQual', 'TA')
        self.fill_na_by_regression()
        self.fill_na_with_mode('GarageYrBlt')

        for x in ['TotalBsmtSF']:
            self.replace_0_to_mean(x)


filled = Filler(df_set.raw_train, df_set.raw_test, df_set.trans_train, df_set.trans_test)
filled.main()


# ====================== #
# 4. Data Transformation #
# ====================== #
class DataTransformer:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test

    def mssub_2_chr(self, x):
        return {20: 'A', 30: 'B', 40: 'C', 45: 'D', 50: 'E',
                60: 'F', 70: 'G', 75: 'H', 80: 'I', 85: 'J',
                90: 'K', 120: 'L', 150: 'M', 160: 'N', 180: 'O', 190: 'P'}.get(x, 0)

    def irregular_extent(self, x):
        return {'Reg': 0, 'IR1': 25, 'IR2': 50, 'IR3': 75}.get(x, 0)

    def story(self, x):
        return {'1Story': 1, '1.5Fin': 1.5, '1.5Unf': 1.25, '2Story': 2, '2.5Fin': 2.5, '2.5Unf': 2.25}.get(x, 0)

    def exter_evaluate(self, x):
        return {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}.get(x, 0)

    def bsmt_qual(self, x):
        return {'Ex': 110, 'Gd': 95, 'TA': 85, 'Fa': 75, 'Po': 65}.get(x, 0)

    def break_down(self, df):
        df['MSSubClass'] = df['MSSubClass'].apply(self.mssub_2_chr)
        df['LotShape_Extent'] = df['LotShape'].apply(self.irregular_extent)
        df['HouseStyle'] = df['HouseStyle'].apply(self.story)
        df['ExterQual'] = df['ExterQual'].apply(self.exter_evaluate)
        df['ExterCond'] = df['ExterCond'].apply(self.exter_evaluate)
        df['BsmtQual'] = df['BsmtQual'].apply(self.bsmt_qual)
        df['BsmtCond'] = df['BsmtCond'].apply(self.exter_evaluate)
        return df

    def main(self):
        self.trans_train = self.break_down(self.trans_train)
        self.trans_test = self.break_down(self.trans_test)


transformed = DataTransformer(filled.raw_train, filled.raw_test, filled.trans_train, filled.trans_test)
transformed.main()


# ==================== #
# 5. Data Combination  #
# ==================== #
class DataCombination:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None, exterior_list=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test
        self.exterior_list = exterior_list

    # Combine Condition1 and Condition2
    def apply_condition_or_for_cond1_n_cond2(self, i, df, x):
        return ((df.loc[i, 'Condition1'] == x) or (df.loc[i, 'Condition2'] == x)) + 0

    def combine_cond1_n_cond2(self):
        for i in range(len(self.trans_train)):
            self.trans_train.loc[i, 'Cond_Artery'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'Artery')
            self.trans_train.loc[i, 'Cond_Feedr'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'Feedr')
            self.trans_train.loc[i, 'Cond_Norm'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'Norm')
            self.trans_train.loc[i, 'Cond_RRNn'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'RRNn')
            self.trans_train.loc[i, 'Cond_RRAn'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'RRAn')
            self.trans_train.loc[i, 'Cond_PosN'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'PosN')
            self.trans_train.loc[i, 'Cond_PosA'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'PosA')
            self.trans_train.loc[i, 'Cond_RRNe'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'RRNe')
            self.trans_train.loc[i, 'Cond_RRAe'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_train, 'RRAe')

        for i in range(len(self.trans_test)):
            self.trans_test.loc[i, 'Cond_Artery'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'Artery')
            self.trans_test.loc[i, 'Cond_Feedr'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'Feedr')
            self.trans_test.loc[i, 'Cond_Norm'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'Norm')
            self.trans_test.loc[i, 'Cond_RRNn'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'RRNn')
            self.trans_test.loc[i, 'Cond_RRAn'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'RRAn')
            self.trans_test.loc[i, 'Cond_PosN'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'PosN')
            self.trans_test.loc[i, 'Cond_PosA'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'PosA')
            self.trans_test.loc[i, 'Cond_RRNe'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'RRNe')
            self.trans_test.loc[i, 'Cond_RRAe'] = self.apply_condition_or_for_cond1_n_cond2(i, self.trans_test, 'RRAe')

    # Combine Exterior
    def define_exterior_list(self):
        self.exterior_list = ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd',
                              'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing']

    def apply_exterior_or(self, i, df, x):
        return ((df.loc[i, 'Exterior1st'] == x) or (df.loc[i, 'Exterior2nd'] == x)) + 0

    def combine_exterior(self):
        for i in range(len(self.trans_train)):
            for item in self.exterior_list:
                self.trans_train.loc[i, 'Exterior_' + item] = self.apply_exterior_or(i, self.trans_train, item)

        for i in range(len(self.trans_test)):
            for item in self.exterior_list:
                self.trans_test.loc[i, 'Exterior_' + item] = self.apply_exterior_or(i, self.trans_test, item)

    # Combine field to Percentage Field
    def get_rate(self):
        self.trans_train['Finish_Rate'] = self.trans_train['BsmtUnfSF'] / self.trans_train['TotalBsmtSF']
        self.trans_test['Finish_Rate'] = self.trans_test['BsmtUnfSF'] / self.trans_test['TotalBsmtSF']

    def main(self):
        self.combine_cond1_n_cond2()
        self.define_exterior_list()
        self.combine_exterior()
        self.get_rate()


combined = DataCombination(transformed.raw_train, transformed.raw_test, transformed.trans_train, transformed.trans_test)
combined.main()


# ============= #
# 6. Data Scope #
# ============= #
class Scope:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None, low_cardinality_cols=None, numeric_cols=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test
        self.low_cardinality_cols = low_cardinality_cols
        self.numeric_cols = numeric_cols

    def find_low_cardinality_cols(self):
        self.low_cardinality_cols = [cname for cname in self.raw_test.columns if self.raw_test[cname].nunique() < 20 and self.raw_test[cname].dtype == "object" and
                                     cname not in ['Utilities', 'Street', 'Alley', 'MiscFeature', 'PoolQC', 'Fence', 'RoofStyle', 'LandSlope', 'Utilities',
                                                   'Condition1', 'Condition2', 'LotShape', 'RootMatl', 'Exterior1st', 'Exterior2nd']]

    def find_numeric_cols(self):
        self.numeric_cols = [cname for cname in self.raw_test.columns if self.raw_test[cname].dtype in ['int64', 'float64'] and
                             cname not in ['MiscVal', 'PoolArea', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'LowQualFinSF', '2ndFlrSF', 'BsmtFinSF2', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF']] +\
                            ['Cond_Artery', 'Cond_Feedr', 'Cond_PosN', 'Cond_PosA', 'Cond_RRAe', 'LotShape_Extent'] +\
                            ['Exterior_AsbShng', 'Exterior_BrkComm', 'Exterior_CemntBd', 'Exterior_HdBoard', 'Exterior_ImStucc',
                             'Exterior_MetalSd', 'Exterior_Plywood', 'Exterior_VinylSd', 'Exterior_Wd Sdng', 'Exterior_WdShing'] + ['Finish_Rate']

    def main(self):
        self.find_low_cardinality_cols()
        self.find_numeric_cols()
        self.trans_train = self.trans_train[self.low_cardinality_cols + self.numeric_cols + ['SalePrice']]
        self.trans_test = self.trans_test[self.low_cardinality_cols + self.numeric_cols]


scoped = Scope(combined.raw_train, combined.raw_test, combined.trans_train, combined.trans_test)
scoped.main()


# ========================================================== #
# 5. One-Hot Encoding and Fillna with 0 for One-Hot Encoding #
# ========================================================== #
class OneHotEncoder:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test

    def one_hot_encoding(self):
        self.trans_train = pd.get_dummies(self.trans_train)
        self.trans_test = pd.get_dummies(self.trans_test)
        self.trans_train, self.trans_test = self.trans_train.align(self.trans_test, join='left', axis=1)

    def fill_with_zero(self):
        self.trans_train = self.trans_train.fillna(0)
        self.trans_test = self.trans_test.fillna(0)

    def print_result(self):
        print("Final Trans Train", self.trans_train.shape)
        print("Final Trans Test", self.trans_test.shape)

    def main(self):
        self.one_hot_encoding()
        self.fill_with_zero()
        self.print_result()


encoded = OneHotEncoder(scoped.raw_train, scoped.raw_test, scoped.trans_train, scoped.trans_test)
encoded.main()


# =================== #
# 7A. Model Parameter #
# =================== #
class Model:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None,
                 y=None, x=None, train_x=None, val_x=None, train_y=None, val_y=None,
                 housing_model=None, val_predictions=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test
        self.y = y
        self.x = x
        self.train_x = train_x
        self.val_x = val_x
        self.train_y = train_y
        self.val_y = val_y
        self.housing_model = housing_model
        self.val_predictions = val_predictions

    def set_parameter(self):
        self.y = self.trans_train.SalePrice
        self.x = self.trans_train.drop(columns=['SalePrice', 'Id', 'YrSold', 'BsmtHalfBath', 'MoSold'])
        print("X", self.x.shape, "Y", self.y.shape)

    def train_val_split(self):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.x, self.y, train_size=0.65, test_size=0.35, random_state=0)

    def fit_model(self):
        self.housing_model = XGBRegressor(n_estimators=200)
        self.housing_model.fit(self.train_x, self.train_y)

    def get_mae(self):
        self.val_predictions = self.housing_model.predict(self.val_x)
        print("MAE", mean_absolute_error(self.val_y, self.val_predictions))

    def main(self):
        self.set_parameter()
        self.train_val_split()
        self.fit_model()
        self.get_mae()


model_df = Model(encoded.raw_train, encoded.raw_test, encoded.trans_train, encoded.trans_test)
model_df.main()


# ======= #
# 8. Test #
# ======= #
class Predictor:
    def __init__(self, raw_train=None, raw_test=None, trans_train=None, trans_test=None, x_test=None):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.trans_train = trans_train
        self.trans_test = trans_test

        self.x_test = x_test

    def drop_column(self):
        self.x_test = self.trans_test.drop(columns=['SalePrice', 'Id', 'YrSold', 'BsmtHalfBath', 'MoSold'])

    def predict_n_round(self):
        self.trans_test['SalePrice'] = model_df.housing_model.predict(self.x_test)
        self.trans_test['SalePrice'] = (self.trans_test['SalePrice'] / 100).apply(np.floor).astype(int) * 100

    def main(self):
        self.drop_column()
        self.predict_n_round()


predicted = Predictor(model_df.raw_train, model_df.raw_test, model_df.trans_train, model_df.trans_test)
predicted.main()

# ========= #
# 9. Export #
# ========= #
df_set.export_csv()
