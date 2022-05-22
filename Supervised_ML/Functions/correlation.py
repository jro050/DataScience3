'''
Code used from:
https://github.com/krishnaik06/Complete-Feature-Selection/blob/master/2-Feature%20Selection-%20Correlation.ipynb
By Jan Rombouts
'''

def correlation(dataset, threshold):
    '''
    Function to find correlations between features above a given threshold
    Parameters:
        Input:  dataset with features, threshold for correlations
        Return: set of variables with correlations to other above threshold
    '''
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr