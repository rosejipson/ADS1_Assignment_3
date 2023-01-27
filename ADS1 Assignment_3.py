# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn import preprocessing
import itertools as iter


def read_and_filter_csv(csv_file):
    """
    This function is used to read the csv file from the directory and to
    import the data for the For curve and fitting

    file_name :- the name of the csv file with data.   
    """

    file_data = pd.read_csv(csv_file)
    dataFrame = pd.DataFrame(file_data)
    dataFrame = dataFrame[['2000', '2001', '2002', '2003', '2004', '2005', 
                           '2006', '2007', '2008', '2009', '2010', '2011']]
    dataFrame = dataFrame.iloc[61:76]
    print(file_data)
    print(dataFrame)
    return file_data, dataFrame

# pair plot comparing the two data
def pairGraph_plot():
    """
    The function helps is creating the pair plot of renewable energy
    and the electricity production. As well to find the k- means clusters of 
    the data
    """

    data_frame_plot = pd.DataFrame()
    renewable = []
    electricity = []

    for i in electricity_data:
        electricity.extend(electricity_data[i])

    for i in renewable_data:
        renewable.extend(renewable_data[i])

    data_frame_plot['renewable'] = renewable
    data_frame_plot['electricity'] = electricity

    type(data_frame_plot)

    # Plottting the data
    sns.pairplot(data_frame_plot[['electricity', 'renewable']])
    sns.set(font_scale = 1.5)
    plt.savefig("pair plot.png")

    # function for finding K means clusttering
    kmeans1 = KMeans(n_clusters=3, random_state=0).fit(
        data_frame_plot[['electricity', 'renewable']])
    kmeans1.inertia_
    kmeans1.cluster_centers_
    data_frame_plot['cluster'] = kmeans1.labels_
    print(data_frame_plot)
    return data_frame_plot

# Scatter plot with K means clustering before and afer normalization
def scatter_kmean_plot(data_kmean):
    """ 
    To find the K-mean clusters and also to construct the 
    scatter plot with and without normalizing the data.
    data_kmean :- the data frame with renewable energy and electricity 
    prodction data
    """

    # plot for K means clusttering before normalisation
    plt.figure()
    sns.scatterplot(x='renewable', y='electricity', hue='cluster',
                    data=data_kmean)
    plt.title("K-Means before normalisation", fontsize = 15, color='red')
    plt.savefig("Scatter K-mean.png")
    plt.show()

    data_plot = data_fr.drop(['cluster'], axis=1)
    names = ['renewable', 'electricity']
    a = preprocessing.normalize(data_plot, axis=0)
    data_aft = pd.DataFrame(a, columns=names)
    kmeans2 = KMeans(n_clusters=3, random_state=0).fit(
        data_aft[['electricity', 'renewable']])
    kmeans2.inertia_
    kmeans2.cluster_centers_
    data_aft['cluster'] = kmeans2.labels_

    # plot for K means clusttering after normalisation
    plt.figure()
    sns.scatterplot(x='renewable', y='electricity', hue='cluster', 
                    data=data_aft)
    plt.title("K-Means after normalisation", fontsize = 15, color='red')
    plt.savefig("Scatter K-mean normalized.png")
    plt.show()
    return


# function to calculate the error limits'''
def func(x, a, b, c):
    return a * np.exp(-(x-b)**2 / c)


# finding upper and lowe limit or ranges of the function
def err_ranges(x, func, param, sigma):
    """
    The function helps to calculate the upper and lower limits for the 
    function,parameters and sigmas for single value or array x. Function 
    values are calculated for all the combinations of positive/negative 
    sigma and the minimum and maximum is determined whcich can be used for 
    all number of parameters and sigmas >=1.  
    """

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper


# Scatter plot of data without normalizing
def scatter_plot():
    """
    Used to create scatter plot without fitting the curve
    """

    data_n = pd.DataFrame()
    renewable = []
    electricity = []

    for i in electricity_data:
        electricity.extend(electricity_data[i])

    for i in renewable_data:
        renewable.extend(renewable_data[i])

    data_n['renewable'] = renewable
    data_n['electricity'] = electricity

    # plot for scattering
    plt.scatter(data_n['renewable'], data_n['electricity'])
    plt.title('Scatter plot without curve fitting', fontsize=15, 
              color='purple')
    plt.ylabel('Renewable resources', fontsize=16)
    plt.xlabel('Electricity production', fontsize=16)
    plt.savefig("scatter plot.png")
    plt.show()
    return

# adding an exponential function'''
def expoFunc(x, a, b):
    return a**(x+b)


# The scatter plot with both cluster and curve fit
def scatter_plot_fitting():
    """ 
    The function is used to fit the curve over the cluster for the same data.
    """

    xaxis_data = data_fr['renewable']
    yaxis_data = data_fr['electricity']
    popt, pcov = curve_fit(expoFunc, xaxis_data, yaxis_data, p0=[1, 0])
    ab_opt, bc_opt = popt
    x_mod = np.linspace(min(xaxis_data), max(xaxis_data), 100)
    y_mod = expoFunc(x_mod, ab_opt, bc_opt)

    # plot for scattering after fitting the curve
    plt.scatter(xaxis_data, yaxis_data)
    plt.plot(x_mod, y_mod, color='r')
    plt.title('Scatter plot with the curve fitting', fontsize=15, 
              color='purple')
    plt.ylabel('Renewable energy', fontsize=16)
    plt.xlabel('Electricity production', fontsize=16)
    plt.savefig("Curve and Cluster.png")
    plt.show()
    return

#Sorting the data for required countries and years
countries = ['Thailand', 'Belgium', 'Bulgaria', 'chile', 'Ireland', 'Portugal']

# data for total greenhouse emissions from a period of year 2000-2011
org_renewable_data, renewable_data = read_and_filter_csv(
    "Renewable energy.csv")


# data for total methane gas emission from a period of year 2000-2011
org_electricity_data, electricity_data = read_and_filter_csv(
    "Electricity production.csv")

# Invoking the functions
data_fr = pairGraph_plot()
scatter_kmean_plot(data_fr)
scatter_plot()
scatter_plot_fitting()
