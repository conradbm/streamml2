# Confidence Intervals
import numpy as np
import scipy as sp
import scipy.stats


def confidence_interval(data, confidence=0.95):
	"""
	Given a data array of averages, find the 95% confidence interval bounds.
	Returns tuple with lower and upper bound.
	"""
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m-h, m+h

    
