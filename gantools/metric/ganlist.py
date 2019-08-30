"""List of metrics for the GAN models."""

from .core import StatisticalMetric, Statistic, StatisticalMetricLim, MetricSum
import numpy as np
from scipy import stats


def mean(x):
    """Compute the mean."""
    return np.mean(x.flatten())


def var(x):
    """Compute the variance."""
    return np.var(x.flatten())


def min(x):
    """Compute the minmum."""
    return np.min(x.flatten())


def max(x):
    """Compute the maximum."""
    return np.max(x.flatten())


def kurtosis(x):
    """Compute the kurtosis."""
    return stats.kurtosis(x.flatten())


def skewness(x):
    """Compute the skewness."""
    return stats.skew(x.flatten())


def median(x):
    """Compute the median."""
    return np.median(x.flatten())

def do_nothing(x):
    """Do nothing."""
    return x

def gan_stat_list(subname='', size=2):
    """Return a list of statistic for a GAN."""
    stat_list = []

    # While the code of the first statistic might not be optimal,
    # it is likely to be negligible compared to all the rest.

    # The order the statistic is important. If it is changed, the test cases
    # need to be adapted accordingly.

    if not (subname == ''):
        subname = '_' + subname

    stat_list.append(Statistic(mean, name='mean'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(var, name='var'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(min, name='min'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(max, name='max'+subname, group='descriptives', stype=0))    
    stat_list.append(Statistic(kurtosis, name='kurtosis'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(skewness, name='skewness'+subname, group='descriptives', stype=0))
    stat_list.append(Statistic(median, name='median'+subname, group='descriptives', stype=0))

    return stat_list


def gan_metric_list(recompute_real=False, size=2):
    """Return a metric list for a GAN."""

    stat_list = gan_stat_list(size=size)
    metric_list = [StatisticalMetric(statistic=stat, recompute_real=recompute_real) for stat in stat_list]

    return metric_list
