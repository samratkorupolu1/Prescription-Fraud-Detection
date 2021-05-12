import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.mixture import BayesianGaussianMixture

# Data Description
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
print(df.dtypes)

columns = df.columns

# Checking for null values
df.isnull().sum()

# Data Visualization
plt.style.use('fivethirtyeight')
# lets build some custom histograms

plt.figure(1, figsize=(20, 10))
n = 0
for x in columns[2:]:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.distplot(df[x])
    plt.title('Distplot of {}'.format(x))
plt.show()

# Lets look at the Gender distribution
plt.figure(1, figsize=(15, 5))
sns.countplot(y='Gender', data=df)
plt.show()

# Plotting the relation between all the variables
sns.pairplot(df)
# checking the relation
plt.figure(1, figsize=(15, 7))
n = 0
for x in columns[2:]:
    for y in columns[2:]:
        n += 1
        plt.subplot(3, 3, n)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        sns.regplot(x=x, y=y, data=df)
        plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
plt.show()

# Age vs Annual Income with respect to gender Gender
plt.figure(1, figsize=(15, 6))
for gender in ['Male', 'Female']:
    plt.scatter(x='Age', y='Annual Income (k$)', data=df[df['Gender'] == gender], s=200, alpha=0.5, label=gender)
    plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')
    plt.title('Age vs Annual Income w.r.t Gender')
    plt.legend()
    plt.show()

# Annual Income vs Spending Score with respect to Gender
plt.figure(1, figsize=(15, 6))
for gender in ['Male', 'Female']:
    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)', data=df[df['Gender'] == gender], s=200, alpha=0.5,
                label=gender)
    plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')
    plt.title('Annual Income (k$) vs  Spending Score (1-100) w.r.t Gender')
    plt.legend()
    plt.show()

# Basket Analysis using unsupervised Algorithms
# 1. K-Means - Based on Annual Income  and Spending score
T1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].iloc[:, :].values
inertia = []
for i in range(1, 11):
    algo: KMeans = KMeans(n_clusters=i, n_init=10, max_iter=300, algorithm='elkan', random_state=0)
    algo.fit(T1)
    inertia.append(algo.inertia_)

# selecting cluster based on inertia (Squared distance between centroids and data points)
plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()

# Visualizing all the clusters
kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(T1)

plt.scatter(T1[y_kmeans == 0, 0], T1[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(T1[y_kmeans == 1, 0], T1[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(T1[y_kmeans == 2, 0], T1[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(T1[y_kmeans == 3, 0], T1[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(T1[y_kmeans == 4, 0], T1[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeansmodel.cluster_centers_[:, 0], kmeansmodel.cluster_centers_[:, 1], s=300, c='yellow',
            label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
KM = plt.show()


# 2. DB Scan - Based on Annual Income and Spending score
def dbscan(X, eps, min_samples):
    # ss = StandardScaler()
    # X = ss.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    y_pred = db.fit_predict(X)
    print(db.labels_)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
    plt.title("DBSCAN")
    plt.show()


dbscan(T1, eps=5, min_samples=4)
# Now the problem with DB scan is to decide the optimal value of epsilon and for that we need to use the nearest
# neighbour logic - This is one method and the min samples come from dimension of he dataset. But if we have no
# domain knowledge we should choose D+1 that is dimension plus one.

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(T1)
distances, indices = nbrs.kneighbors(T1)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()
# the elbow curve for nearest neighbour gives optimal value for epsilon at maximum curvature is 5.8 approximately
dbscan(T1, eps=5.7, min_samples=4)


# The difference between the plots for DB Scan and K-means Even the the members in the clusters provided in db scan
# are same as k-means but as db scan relies on the density the outliers of the less dense core instance are assigned
# with -1 as anomalies

# 3. Gaussian mixture Model -Based on Annual Income and Spending Score

# function to plot and visualize GMM
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


gm = GaussianMixture(n_components=5, n_init=100, covariance_type='full')
plot_gmm(gm, T1)
plt.show()

# But how to decide whether what is the optimal number of customers try to find the model that minimizes a theoretical
# information criterion such as the BIC(Bayesian information criterion) and AIC Akaike information criterion(AIC)

aic = []
bic = []
for i in range(1, 11):
    gm = GaussianMixture(n_components=i, n_init=15, covariance_type='full')
    gm.fit(T1)
    aic.append(gm.aic(T1))
    bic.append(gm.bic(T1))

plt.figure()
plt.plot(np.arange(1, 11), aic, 'o')
plt.plot(np.arange(1, 11), aic, '-', alpha=0.5, label='AIC')
plt.plot(np.arange(1, 11), bic, 'o')
plt.plot(np.arange(1, 11), bic, '-', alpha=0.5, label='BIC')
plt.xlabel('Number of Clusters'), plt.ylabel('AIC and BIC')
plt.legend()
plt.show()

# After looking at the all the three unsupervised techniques the optimal number of clusters comes out to be around
# 4-5 where it was confirmed 5 by two techniques
# Just the last model to check cluster range
# 4. Bayesian Gaussian Mixture - Based on Annual Income and spending score
bgm = BayesianGaussianMixture(n_components=10, n_init=15)
bgm.fit(T1)
print(np.round(bgm.weights_, 1))
