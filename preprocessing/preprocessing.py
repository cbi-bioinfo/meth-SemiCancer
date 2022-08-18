from sklearn.model_selection import train_test_split
import pandas as pd
import random
import sys

print('Data loading')

labeled_X = pd.read_csv(sys.argv[1], index_col=0)
labeled_Y = pd.read_csv(sys.argv[2], index_col=0)
unlabeled_X = pd.read_csv(sys.argv[3], index_col=0)

print('# Removing CpG sites having missing values for more than 20% of samples')
def removeMissingValue(data, percent):
    length = data.shape[1]
    data_T = data.T
    idx = data_T.isnull().sum() >= length*percent
    idx_True = idx[idx == True]
    data_T_na = data_T.drop(idx_True.index, axis=1)
    data_na = data_T_na.T
    return data_na

labeled_X_na = removeMissingValue(labeled_X, 0.2)
unlabeled_X_na = removeMissingValue(unlabeled_X, 0.2)

print('# Performing median imputation')
labeled_merge = pd.concat([labeled_Y,labeled_X_na.T],axis=1, join='inner')
fill_median_func = lambda g: g.fillna(g.median())
labeled_merge_imputation = labeled_merge.groupby('cancer_subtype').apply(fill_median_func)
labeled_merge_imputation_X = labeled_merge_imputation.drop(['cancer_subtype'], axis=1)
labeled_merge_imputation_X = labeled_merge_imputation_X.reset_index()
labeled_Y = labeled_merge_imputation_X['cancer_subtype']
labeled_imputation_X = labeled_merge_imputation_X.drop(['cancer_subtype'], axis=1)
labeled_imputation_X = labeled_imputation_X.rename(columns={'level_1':'cpg'})
labeled_imputation_X = labeled_imputation_X.set_index('cpg')

unlabeled_X_na_T = unlabeled_X_na.T
unlabeled_imputation_X = unlabeled_X_na_T.fillna(unlabeled_X_na_T.median())

print('# Extrating common CpG sites between labeled and unlabeled dataset')
common_cpg = list(set(list(labeled_imputation_X.columns)) & set(list(unlabeled_imputation_X.columns)))
common_labeled_X = labeled_imputation_X[common_cpg]
common_unlabeled_X = unlabeled_imputation_X[common_cpg]

labeled_Y_onehot = pd.get_dummies(labeled_Y)
subtype_list = list(labeled_Y_onehot.columns)
unlabeled_Y = [random.choice(subtype_list) for i in range(len(common_unlabeled_X))]
unlabeled_Y_onehot = pd.get_dummies(unlabeled_Y)

# Spliting train and test dataset
train_X, test_X, train_Y, test_Y = train_test_split(common_labeled_X, labeled_Y_onehot, test_size=0.1, shuffle=True, random_state=34)

# Saving dataset
train_X.to_csv("./train_X.csv", sep=',', index=False)
test_X.to_csv("./test_X.csv", sep=',', index=False)
train_Y.to_csv("./train_Y.csv", sep=',', index=False)
test_Y.to_csv("./test_Y.csv", sep=',', index=False)
common_unlabeled_X.to_csv("./unlabel_X.csv", sep=',', index=False)
unlabeled_Y_onehot.to_csv("./unlabel_Y.csv", sep=',', index=False)

