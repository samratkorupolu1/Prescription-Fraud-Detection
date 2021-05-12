# Auxiliaries
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 200)


def load_data(filename="df_play.csv"):
    return pd.read_csv(filename, low_memory=False)


def remove_one(dataframe):
    for col in dataframe.columns:
        if len(dataframe[col].unique()) == 1:
            dataframe.drop(col, axis=1, inplace=True)
    return dataframe


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset

    print(dataset.shape)


def correlation_plot(dataframe, threshold=-1):
    corr = dataframe.corr()
    high_corr = corr[corr >= threshold]
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plt.figure(figsize=(30, 10))
    sns.heatmap(high_corr, cmap=cmap,
                square=True)
    return plt.show()



def print_unique_values(dataframe):
    dataframe = dataframe.nunique().to_frame().reset_index()
    dataframe.columns = ['Variable', 'DistinctCount']
    return dataframe


def drop_col(dataframe, axis=1, column_name=[]):
    return dataframe.drop(column_name, axis)
