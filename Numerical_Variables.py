if __name__ == "__main__":
    import pandas as pd
import Ut_Functions
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# load the data set and separate the columns based on category
df = Ut_Functions.load_data("df_play.csv")
num_cols = df.select_dtypes(include=['float64', 'int64'])
num_cols = num_cols.iloc[:, 1:]
cat_cols = df.select_dtypes(include=['object'])

# One hot encode the targets
ordinal_encoder = OrdinalEncoder()
targets_enc = ordinal_encoder.fit_transform(cat_cols[['Claim Status']])
targets_enc = pd.DataFrame(targets_enc)
targets_enc.value_counts(sort=True)

# Create a copy of the num cols for safety stake
orig_nums = num_cols.copy()

# Lets work on numerical features
# num_cols.hist(bins=50, figsize=(40, 20))
# plt.show()

# dropping the columns with null values
num_cols = orig_nums.dropna(axis=1, how="all")
print(orig_nums.shape)
print(num_cols.shape)

# lets check the amount of null values in the dataset
num_cols.isna().sum() * 100 / len(num_cols)
print(num_cols.isin([' ', 'NULL', 0, 'nan']).mean())

# Lets remove the column with unique values ==1
num_cols = Ut_Functions.remove_one(num_cols)
print(num_cols.shape)
print(orig_nums.shape)

# Redundancy columns to be removed
redu = ['Division Code (Elig)', 'Enrollment Code (Elig)', 'Home County Code (Elig)',
        'Location Identifier (Elig)', 'Place Of Service', 'Plan Code (Elig)',
        'Plan Coverage (Elig)', 'Sales Tax', 'Service Line Index', 'Service Year',
        'Vaccine Admin Fee', 'Work County Code (Elig)', 'Location Identifier (CLM)']
num_cols = Ut_Functions.drop_col(num_cols, column_name=redu)
print(num_cols.shape)

sec_redu = [
    "AWP Amount",
    "Cancer Treatment",

    'Check Amount',

    'Claim Dependent Identifier',
    'Claim Sequence Number',
    'Claim Serial Number',
    'Coinsurance Amount',
    'Copay Amount',
    'DAW Code',
    'Deductible Amount',
    'Department Code (Elig)',
    'Dependent Code',
    'Dependent Identifier (Elig)',
    'Dispensing Fee Amount',
    'Division (Elig)',
    'Drug Identifier',
    'Drug Therapeutic Class Code AHSF',
    'Emergency Visit Count',
    'Employee Resp Not Covered Amount',
    'Enrollment Code',
    'Episode ID',
    'Fill Number',
    'Home County Code',
    'Ingredient Cost',
    'Inpatient Hospitalization Count',
    'Location Identifier',
    'Plan Code',
    'Plan Coverage',
    'Plan Payment Amount',
    'Predicted Resource Index - Benchmark',
    'Predicted Resource Index - Local',
    'Probability of IP Hospitalization',
    'Provider Reference Number',
    'ProviderFedTaxID',
    'Pseudo Claim Service Lines ID',
    'Readmission Count',
    'Readmission Probability',
    'Resource Utilization Band',
    'Samas Code (Elig)',
    'Section Code (Elig)',
    'Service Counter',
    'Subsection Code (Elig)',
    'Work County Code'
]
num_cols = Ut_Functions.drop_col(num_cols, column_name=sec_redu)
print(num_cols.shape)

num_cols['Target'] = targets_enc

tf_num = num_cols[['Age (risk)', 'Chronic Condition Count', 'Drug GCN Number', 'Number of Refills', 'Zip Code (Elig)',
               'Total Charge Amount', 'Target']]
tf_num['Total Charge Amount'] = abs(tf_num['Total Charge Amount'])
tf_num = tf_num.dropna(how='any')

tf_cat = cat_cols[['City (Elig)', 'Coverage Tier', 'Drug Class', 'Gender (risk)', 'Prescribing Provider Name']]
tf_cat = tf_cat.dropna(how='any')
tf = pd.concat([tf_cat,tf_num], axis=1)
tar = 'Target'
predictors_num = ['Age (risk)', 'Chronic Condition Count', 'Drug GCN Number', 'Number of Refills',
                  'Total Charge Amount']
predictors_cat = ['City (Elig)', 'Coverage Tier', 'Drug Class', 'Gender (risk)', 'Prescribing Provider Name']

predictors = predictors_num + predictors_cat
RFC_METRIC = 'gini'  # metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100  # number of estimators used for RandomForrestClassifier
NO_JOBS = 4  # number of parallel jobs used for RandomForrestClassifier

# TRAIN/VALIDATION/TEST SPLIT
# VALIDATION
VALID_SIZE = 0.20  # simple validation using train_test_split
TEST_SIZE = 0.20  # test size using_train_test_split

# CROSS-VALIDATION
NUMBER_KFOLDS = 5
train_df, test_df = train_test_split(tf, test_size=TEST_SIZE, random_state=42, stratify=tf['Target'])
train_df, valid_df = train_test_split(tf, test_size=VALID_SIZE, random_state=42, stratify=tf['Target'])

clf = RandomForestClassifier(n_jobs=NO_JOBS,
                             random_state=42,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)
clf.fit(train_df[predictors], train_df[tar].values)
preds = clf.predict(test_df[predictors])

cm = pd.crosstab(valid_df[tar].values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
sns.heatmap(cm,
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True, ax=ax1,
            linewidths=.2, linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()

roc_auc_score(valid_df[tar].values, preds)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print(accuracy_score(valid_df[tar].values, preds))
print(precision_recall_fscore_support(valid_df[tar].values, preds))
cm = confusion_matrix(valid_df[tar].values, preds, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()

tmp = pd.DataFrame({'Feature': predictors = predictors_num + predictors_cat, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance', ascending=False)
plt.figure(figsize=(20, 10))
plt.title('Features importance', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance', data=tmp)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.show()
