'''
Module to plot results of clustering methods

Code adapted from Data Science 3, by Fenna Feenstra
Author: Jan Rombouts
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def plot_cluster(X, df):
    #add x, y features to the original df
    df['x'] = X[:,0]
    df['y'] = X[:,1]
    ax = sns.scatterplot(x = 'x', y = 'y', hue = 'histology', data = df, 
                        legend = False, alpha = 1, palette=['red', 'blue', 'green'])


def plot_sub_cluster(X, df):
    #add x, y features to the original df
    df['x'] = X[:,0]
    df['y'] = X[:,1]
    ax = sns.scatterplot(x = 'x', y = 'y', hue = 'histology', data = df, 
                        legend = False, alpha = 0.5, palette=['red', 'blue', 'green'])
    return ax

def optic_plot(clust, X):
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in zip(range(0, 5), colors):
        Xk = X[clust.labels_ == klass]
        plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    plt.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
    plt.title("OPTICS")
    plt.show()

def plot_pca_comp(pca, X_df, PC):
    loadings = pca.components_
    num_pc = pca.n_features_
    pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = X_df.columns.values
    loadings_df = loadings_df.set_index('variable')
    
    loadings_pc = loadings_df.sort_values(by=[PC])
    top_pc = loadings_pc.iloc[:10,0:1]
    bottom_pc = loadings_pc.iloc[-10:,0:1]
    max_loadings_pc = top_pc.append(bottom_pc)
    
    # create a correlation matrix
    c = X_df[max_loadings_pc.index].corr().abs()
    sns.heatmap(c)
