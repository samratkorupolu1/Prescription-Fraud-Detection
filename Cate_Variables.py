from Numerical_Variables import cat_cols
from Ut_Functions import print_unique_values, drop_col, remove_one
import pandas as pd

orig_cats = cat_cols.copy()
cat_cols_unique = print_unique_values(cat_cols)
print(cat_cols.shape)
print(orig_cats.shape)

cat_cols_unique.query('DistinctCount == 1').DistinctCount.count()

# Dropping the values with null values
cat_cols = orig_cats.dropna(axis=1, how="all")
print(orig_cats.shape)
print(cat_cols.shape)

# drop the speculated target column from the dataset
cat_cols = drop_col(cat_cols, column_name=['Claim Status'])
print(cat_cols.shape)

# check and remove the null columns
cat_cols.isna().sum() * 100 / len(cat_cols)
print((cat_cols.isin([' ', 'NULL', 0, 'nan']).mean()))

# removing columns with only one unique value
cat_cols = cat_cols[[c for c
                     in list(cat_cols)
                     if len(cat_cols[c].unique()) > 1]]
print(cat_cols.shape)
print(orig_cats.shape)

# Removal of redundant and non-useful rows
redu_cat = ['Drug Dosage Form', 'Drug Strength', 'Drug Type', 'Employer Identifier (Elig)',
            'From Date (Risk)', 'Gender (Elig)', 'Generation (Elig)', 'Hire Date (Elig)',
            'HMO Code', 'HMO Code (Elig)', "Hypertension", 'Location Name (Elig)', 'Medical Coverage? (Elig)',
            'Paid Date', 'Patient Birth Date', 'Patient Birth Date (Claim)', 'Patient Gender',
            'Plan Description', 'Plan Description (Elig)', 'Plan Service Sub Category',
            'Received Date', 'Relation To Employee (Elig)', 'Service Address2', 'Service City',
            'Service From Date', 'Service Through Date', 'State (Elig)', 'Through Date']

cat_cols = drop_col(cat_cols, column_name=redu_cat)
print(cat_cols.shape)

sec_redu_cat = ['Benefit Class (Elig)', 'BI Coverage Types (Elig)', 'Birth Date (Elig)',
                'Check Date', 'Coverage Tier (Elig)', 'Drug Description',
                'Specialty Drug Indicator']

cat_cols = drop_col(cat_cols, column_name=sec_redu_cat)
print(cat_cols.shape)
