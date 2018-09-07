"""Least-square fitting tools."""


# --- import --------------------------------------------------------------------------------------


from ._utilities import Timer

import numpy as np

from scipy import optimize as scipy_optimize


# --- define --------------------------------------------------------------------------------------


__all__ = ["leastsqfitter"]


# --- functions -----------------------------------------------------------------------------------


def leastsqfitter(p0, datax, datay, function, verbose=False, cov_verbose=False):
    """Conveniently call scipy.optmize.leastsq().

    Returns fit parameters and their errors.

    Parameters
    ----------
    p0 : list
        list of guess parameters to pass to function
    datax : array
        array of independent values
    datay : array
        array of dependent values
    function : function
        function object to fit data to. Must be of the callable form function(p, x)
    verbose : bool
        toggles printing of fit time, fit params, and fit param errors
    cov_verbose : bool
        toggles printing of covarience matrix

    Returns
    -------
    pfit_leastsq : list
        list of fit parameters. s.t. the error between datay and function(p, datax) is minimized
    perr_leastsq : list
        list of fit parameter errors (1 std)
    """
    timer = Timer(verbose=False)
    with timer:
        # define error function
        def errfunc(p, x, y):
            return y - function(p, x)

        # run optimization
        pfit_leastsq, pcov, infodict, errmsg, success = scipy_optimize.leastsq(
            errfunc, p0, args=(datax, datay), full_output=1, epsfcn=0.0001
        )
        # calculate covarience matrix
        # original idea https://stackoverflow.com/a/21844726
        if (len(datay) > len(p0)) and pcov is not None:
            s_sq = (errfunc(pfit_leastsq, datax, datay) ** 2).sum() / (len(datay) - len(p0))
            pcov = pcov * s_sq
            if cov_verbose:
                print(pcov)
        else:
            pcov = np.inf
        # calculate and write errors
        error = []
        for i in range(len(pfit_leastsq)):
            try:
                error.append(np.absolute(pcov[i][i]) ** 0.5)
            except BaseException:
                error.append(0.00)
        perr_leastsq = np.array(error)
    # exit
    if verbose:
        print("fit params:       ", pfit_leastsq)
        print("fit params error: ", perr_leastsq)
        print("fitting done in %f seconds" % timer.interval)
    return pfit_leastsq, perr_leastsq
