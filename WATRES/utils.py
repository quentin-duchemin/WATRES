
import numpy as np
import torch
import random
from datetime import datetime, timedelta

def fractional_year_to_datetime(fractional_year):
    """
    Converts a fractional year (e.g., 2023.5) to a datetime object rounded to the nearest hour.
    The fractional part is interpreted as the fraction of the year that has elapsed.
    """
    # Extract the integer part (year) and the fractional part
    year = int(fractional_year)
    fractional_part = fractional_year - year
    
    # Calculate the number of days in the given year
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year + 1, 1, 1)
    days_in_year = (end_of_year - start_of_year).days
    
    # Calculate the total number of seconds represented by the fractional part
    seconds_in_year = days_in_year * 24 * 3600
    elapsed_seconds = fractional_part * seconds_in_year
    
    # Get the date and time by adding the elapsed seconds to the start of the year
    date_time = start_of_year + timedelta(seconds=elapsed_seconds)
    
    # Round to the nearest hour
    if date_time.minute >= 30:
        date_time += timedelta(hours=1)
    rounded_date_time = date_time.replace(minute=0, second=0, microsecond=0)
    
    return rounded_date_time

def datetime2year_month(dt):
    """
    Converts a datetime object to a string in the format 'YYYY-M'.
    """
    a = str(dt.year)
    a = a+'-'+str(dt.month)
    return(a)

def get_clusters_time_points(pathdata, lst, nbclusts=4, use_cout=False):
    """
    Clusters time points in 'lst' into 'nbclusts' clusters based on quantiles of Q or Cout values.
    Loads data from pathdata, and returns a list of clusters (each cluster is a list of indices).
    """
    import pickle
    f = open(pathdata+"data.pkl","rb")
    data = pickle.load(f)
    f.close()
    
    if use_cout:
        Q = data['Cout']
    else:
        Q = np.load(pathdata+'Q.npy')
    sortedQ = np.sort(Q)
    quantiles = [sortedQ[int(k*0.25*len(sortedQ))] for k in range(4)] + [sortedQ[-1]+1]
    def find_level(x):
        k = 0
        while k<nbclusts:
            if quantiles[k]<=x<quantiles[k+1]:
                return k
            else: 
                k += 1 
    clusters = [[] for k in range(nbclusts)]
    n = len(lst)
    for i, t in enumerate(lst):
         clusters[find_level(Q[t])].append(i)    
    return clusters

def get_clusters_time_points_subsequence(pathdata, lst, nbclusts=4, use_cout=False):
    """
    Clusters time points in 'lst' into 'nbclusts' clusters by sorting Q or Cout values and splitting into equal-sized groups.
    Returns a list of clusters (each cluster is a numpy array of indices).
    """
    import pickle
    f = open(pathdata+"data.pkl","rb")
    data = pickle.load(f)
    f.close()
    
    if use_cout:
        Q = data['Cout']
    else:
        Q = np.load(pathdata+'Q.npy')
    timesorted = np.argsort(np.array(Q)[lst])
    clusters = []
    n = len(lst)
    for l in range(nbclusts):
         clusters.append(timesorted[int(n*l/nbclusts):int(n*(l+1)/nbclusts)])    
    return clusters

def fit_sin(tt, yy):
    """
    Fits a sinusoidal function to the input time sequence (tt, yy).
    Returns a dictionary with amplitude, phase, offset, fit function, max covariance, and raw results.
    """
    tt = numpy.array(tt)
    yy = numpy.array(yy)
    ff = numpy.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(numpy.fft.fft(yy))
    guess_freq = abs(ff[numpy.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = 4. #numpy.std(yy) * 2.**0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_amp,  0., guess_offset])

    def sinfunc(t, A, p, c):  return A * numpy.sin(2*np.pi*t/(24*365) + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, p, c = popt
    #f = w/(2.*numpy.pi)
    fitfunc = lambda t: A * numpy.sin(2*np.pi*t/(24*365) + p) + c
    return {"amp": A, "phase": p, "offset": c, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}

def get_ampli(name_data, data, figure=True, use_noiseless_CJ=False):
    """
    Computes the amplitude of a sinusoidal fit to the data[name_data] time series.
    Optionally plots the data and fit. If use_noiseless_CJ is True, uses fixed coefficients for the fit.
    Returns the absolute value of the amplitude.
    """
    import pylab as plt

    N, amp, omega, phase, offset, noise = 500, 10., 2., .5, 4., 3
    #N, amp, omega, phase, offset, noise = 50, 1., .4, .5, 4., .2
    #N, amp, omega, phase, offset, noise = 200, 1., 20, .5, 4., 1
    lst_sin =  np.arange(0,len(data[name_data]),1)[np.where(1-np.isnan(data[name_data]))[0]]
    if use_noiseless_CJ:
        sincoeff = -0.0464  
        coscoeff = -2.19 
        offset = -10
        yynoise = sincoeff*np.sin(2*np.pi*lst_sin/(365*24))  + coscoeff*np.cos(2*np.pi*lst_sin/(365*24))  + offset
    else:
        yynoise = data[name_data][lst_sin]
    res = fit_sin(lst_sin, yynoise)
    #print( "Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res )
    plt.plot(lst_sin, yynoise, "ok", label="y with noise")
    plt.plot(lst_sin, res["fitfunc"](lst_sin), "r-", label="y fit curve", linewidth=2)
    plt.title(name_data)
    plt.show()
    return abs(res['amp'])
