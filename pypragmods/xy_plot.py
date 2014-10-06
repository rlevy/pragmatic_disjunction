#!/usr/bin/env python

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

"""
Markers: http://matplotlib.org/api/markers_api.html#module-matplotlib.markers

Lines: http://matplotlib.org/api/artist_api.html#matplotlib.lines.Line2D.set_linestyle

"""

def histogram(vals, nbins=10, title='', xlab='', ylab=''):
    n, bins, patches = plt.hist(vals, nbins, normed=True)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
def scatterplot(xvals, 
                yvals, 
                title="", 
                xlab="", 
                ylab="",                
                marker='o',
                markersize=5,
                markerfacecolor='black'):
    generic_xy_plot(xvals, yvals, 
                    title=title, xlab=xlab, ylab=ylab, 
                    linestyle='', 
                    marker=marker, markersize=markersize, markerfacecolor=markerfacecolor)

def lineplot(xvals, 
             yvals, 
             title="", 
             xlab="", 
             ylab="",   
             linestyle='-',
             linewidth=2,
             linecolor='black',
             marker='',
             markersize=5,
             markerfacecolor='black'):
    generic_xy_plot(xvals, yvals, 
                    title=title, xlab=xlab, ylab=ylab, 
                    linestyle=linestyle, linewidth=linewidth, linecolor=linecolor,
                    marker=marker)

def generic_xy_plot(xvals, 
                    yvals, 
                    title="", 
                    xlab="", 
                    ylab="", 
                    linestyle='', 
                    linewidth=2,
                    linecolor='black',
                    marker='o',
                    markersize=5,
                    markerfacecolor='black'):
    # A bit of error handling:
    if len(xvals) != len(yvals):
        raise TypeError("Vectors must be aligned; xvals has length %s, but yvals has length %s" % (len(xvals), len(yvals)))
    for x in xvals + yvals:
        if not (isinstance(x, int) or isinstance(x, float)):
            raise TypeError("generic_xy_plot requires numeric values")
    # Now to the plotting:
    plt.plot(xvals, 
             yvals, 
             marker=marker, 
             markersize=markersize,
             markerfacecolor=markerfacecolor, 
             markeredgecolor=markerfacecolor, 
             linestyle=linestyle,
             color=linecolor)   
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
######################################################################

def lm(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    coef, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    return {'coef':coef, 'intercept':intercept, 'rvalue':rvalue, 'pvalue': pvalue, 'se':se}

def add_lm(x, y,
           linestyle='-',
           linewidth=2,
           linecolor='red'):
    fit = lm(x, y) 
    vals = [fit['intercept'] + (fit['coef']*i) for i in x]
    plt.plot(x, vals, linestyle=linestyle, linewidth=linewidth, color=linecolor, marker='')

######################################################################
    
if __name__ == '__main__':
   
    # Quick test of scatterplot:
    import random
    xvals = range(1,20)
    yvals = [random.uniform(-2,2)+x for x in xvals]

    scatterplot(xvals, yvals) #, title="Linear", xlab="X", ylab="Y")
    add_lm(xvals, yvals)
    plt.show()    
