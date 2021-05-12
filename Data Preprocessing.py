import pandas as pd
from dask import dataframe as dd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Clustering

from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

# load the dataset
df = pd.read_csv("PharmacyDatanew.csv")
# some info about the rows and columns of the data
df.info(verbose=False, memory_usage="deep")
len(df)


# Function to create a mini-dataset for playing around
def frac(dataframe, fraction, other_info=None):
    """Returns fraction of data"""
    return dataframe.sample(frac=fraction)


# creating a 10% mini dataset for fast exploration
df_play = frac(df, 0.01)
print(df_play.dtypes)
df_play.info()
df_play.describe()

# -----------------------------------------#-------------------------------------------------#
# Everything we will do here is for the play dataset which will be implemented in the full dataset.
# Checking data types of each column
pd.set_option('display.max_rows', 200)
print(df_play.dtypes)

# Percentage of missing values in each column and drop columns with more than 60% missing values
df_play.isna().sum() * 100 / len(df)
print(df_play.isin([' ', 'NULL', 0]).mean())
df_play = df_play.loc[:, df_play.isin([' ', 'NULL']).mean() < .6]

# looking at the start and end date for the transactions
df_play['Transaction Date'] = df_play['Transaction Date'].astype('str')
df_play['Transaction Date'] = pd.to_datetime(df_play['Transaction Date'], format='%m/%d/%Y')
# df['Transaction Date'] = df['Transaction Date'].map(lambda ts: ts.strftime("%d-%m-%Y"))
start_date = (df_play['Transaction Date'].min())
end_date = (df_play['Transaction Date'].max())

# Lets aggregate the distinct values for all categorical variables
df_play_unique = df_play.nunique().to_frame().reset_index()
df_play_unique.columns = ['Variable', 'DistinctCount']

# converting all object data types to category
df_play.loc[:, df_play.dtypes == 'object'] = \
    df_play.select_dtypes(['object']) \
        .apply(lambda x: x.astype('category'))

df_play.info()

# Lets look at the correlation matrix for this play dataset
corr = df_play.corr()
plt.figure(figsize=(16, 8))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
plt.show()

sns.displot(df_play, x='Plan Payment Amount', hue='Claim Status', multiple='stack')
plt.show()
# -----------------------------------------------------#-------------------------------------------------#

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
dataset = df_play[['Copay Amount', 'Plan Payment Amount', 'Drug Category', 'Plan Service Sub Category',
                   'Brand/Generic Flag', 'Plan Service Sub Category', 'Claim Status']]

# num_features = dataset[['Copay Amount', 'Plan Payment Amount']]
# cat_features = dataset[['Drug Category', 'Plan Service Sub Category',
#                         'Brand/Generic Flag', 'Plan Service Sub Category', 'Claim Status']]

X = np.array(dataset.iloc[:,:-1])
y = np.array(dataset.iloc[:,-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=6)
decision_tree = decision_tree.fit(X_train, y_train)
r = export_text(decision_tree, feature_names='Copay Amount', 'Plan Payment Amount', 'Drug Category', 'Plan Service Sub Category',
                   'Brand/Generic Flag', 'Plan Service Sub Category')
print(r)