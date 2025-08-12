import numpy as np
import pandas as pd
import arviz as az
from scipy import stats
from BiCEP import model_circle_fast, model_circle_slow
from BiCEP.criteria import auto_interpret


def extract_values(fit, var):
    """
    Extracts the data for variable in the BiCEP fit to a 1d numpy array

    Parameters
    ------
    fit arviz InferenceData:
    site level BiCEP fit object

    var str:
    name of variable in fit

    Returns
    -------
    values: array
    1d array of values for that variable in the fit
    """
    values = fit.posterior[var].stack(sample=["chain", "draw"]).values
    return values


def get_sampler_success(site, abs_tol=16, perc_tol=40, acceptSkew=False):
    """
    Checks for a BiCEP site result has a full width of its credible
    interval <abs_tol μT or perc_tol% of the median. Additionally, checks that
    the sampler did not fail the rhat criterion (if it does, you may
    need to rerun this function on the site). The function also
    outputs a warning if the skewness of the distribution of the
    site mean intensity is highly skewed, which can be indicative of
    a result which cannot exclude zero. These results have a higher
    chance of being inaccurate because the mean is restricted to be
    greater than zero and so this may artificially reduce the
    uncertainty in the mean value.

    Parameters
    ------
    site: BiCEP specimenCollection object
    site/sample used to check for success.

    abs_tol: float
    absolute maximum limit for accurate specimen

    perc_tol: float
    absolute maximum percentage tolerance for accurate specimen

    acceptSkew: bool
    Whether we accept A- / B- results (default False)

    Returns
    -------
    True/False: bool
    Whether the site has a successful result or not (an A or a B)
    """

    # Get the Rhats for BiCEP
    rhat_worst = site.get_specimen_rhats()

    # Get the distribution of possible site mean intensities
    int_site = extract_values(site.fit, "int_site")

    # Get the percentiles of the intensity
    minB, medB, maxB = np.percentile(int_site, (2.5, 50, 97.5))

    # Look at the percentile criteria
    perc_dev = (maxB - minB) / medB
    abs_dev = maxB - minB

    # Get skewness of the distribution
    skew = stats.skew(int_site)

    # Warn if high skewness (can lead to low absolute )
    if (perc_dev >= perc_tol / 100) & (abs_dev < abs_tol) & (skew > 1):
        raise Warning(
            "Warning "
            + site.name
            + " has a distribution which cannot rule out zero field strength, this may cause a passing grade even when it shouldn't"
        )
        return acceptSkew

    # If it's a good sample (0.9<rhat<1.1) and passes our criteria for precision, it passes
    elif (rhat_worst < 1.1) & (rhat_worst > 0.9) & ((perc_dev < 0.4) | (abs_dev < 16)):
        print(
            site.name
            + " successful! Estimated intensity %2.1f" % medB
            + " with bounds of %2.1f" % minB
            + " - %2.1f" % maxB
            + " μT"
        )
        return True

    elif (rhat_worst >= 1.1) | (rhat_worst <= 0.9):
        raise Warning("Warning " + site.name + " Failed, sampler did not converge!")
        return False
    # Otherwise, it fails
    else:
        return False


def run_site_fit(
    site, min_specs=4, save_nc=False, model=None, n_samples=10000, **kwargs
):
    """
    Runs a BiCEP fit to a site, automatically choosing
    specimen level interpretations.

    Parameters
    ------
    site: BiCEP specimenCollection object.
    site/specimen to run fit on

    Returns
    -------
    None
    """

    wd = site.parentData.wd
    # Check if 4 or more specimens. BiCEP almost always fails with less.
    if len(site.specimens) >= min_specs:
        auto_interpret(site, 5, 10, 10, **kwargs)  # Get interpretations

    # Skip site otherwise
    else:
        print("skipping site " + str(site.name) + ", not enough specimens in site")
        return

    actives = 0
    # Check how many specimens had a passing interpretation

    for specimen in site.specimens.values():
        if specimen.active == True:
            actives += 1

    # If more than 4 had a passing interpretation, run BiCEP
    if actives >= min_specs:
        # We use 10000 n_samples and the fast model here. If you're having issues
        # Run with higher n_samples and/or the "slow" model
        site.BiCEP_fit(model=model, n_samples=n_samples)
        success = get_sampler_success(site, **kwargs)  # Check we passed

        if success == True:  # If succesful, save the BiCEP fit
            if save_nc == True:
                site.fit.to_netcdf(wd + site.name + ".nc")
                # site.save_magic_tables()

        else:  # Otherwise say it failed
            print(site.name + " did not have a passing estimate with BiCEP")

    # If less than 4 specimens pass minimal criteria, ignore site.
    else:
        print("skipping site " + site.name + ", not enough specimens passed criteria")
        return
