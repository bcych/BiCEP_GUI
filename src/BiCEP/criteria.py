import numpy as np
import pandas as pd
import pickle
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def sufficient_statistics(ptrm, nrm):
    """
    inputs list of ptrm and nrm data and computes sufficent statistcs
    needed for computations

    Inputs
    ------
    ptrm: list
    list of ptrm data

    nrm: list
    list of nrm data

    Returns
    -------
    dict containing mean ptrm and nrm, and covariances in xx, xy and yy.
    """

    corr = np.cov(np.stack((ptrm, nrm), axis=0))

    return {
        "xbar": np.mean(ptrm),
        "ybar": np.mean(nrm),
        "S2xx": corr[0, 0],
        "S2yy": corr[1, 1],
        "S2xy": corr[0, 1],
    }


def TaubinSVD(x, y):
    """
    Function from PmagPy
    algebraic circle fit
    input: list [[x_1, y_1], [x_2, y_2], ....]
    output: a, b, r.  a and b are the center of the fitting circle, and
    r is the radius

     Algebraic circle fit by Taubin
      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                  Space Curves Defined By Implicit Equations, With
                  Applications To Edge And Range Image Segmentation",
      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    """
    X = np.array(list(map(float, x)))
    Xprime = X
    Y = np.array(list(map(float, y)))
    Yprime = Y
    XY = np.array(list(zip(X, Y)))
    XY = np.array(XY)
    X = XY[:, 0] - np.mean(XY[:, 0])  # norming points by x avg
    Y = XY[:, 1] - np.mean(XY[:, 1])  # norming points by y avg
    centroid = [np.mean(XY[:, 0]), np.mean(XY[:, 1])]
    Z = X * X + Y * Y
    Zmean = np.mean(Z)
    Z0 = (Z - Zmean) / (2.0 * np.sqrt(Zmean))
    ZXY = np.array([Z0, X, Y]).T
    U, S, V = np.linalg.svd(ZXY, full_matrices=False)  #
    V = V.transpose()
    A = V[:, 2]
    A[0] = A[0] / (2.0 * np.sqrt(Zmean))
    A = np.concatenate([A, [-1.0 * Zmean * A[0]]], axis=0)
    a, b = (-1 * A[1:3]) / A[0] / 2 + centroid
    r = np.sqrt(A[1] * A[1] + A[2] * A[2] - 4 * A[0] * A[3]) / abs(A[0]) / 2
    errors = []
    for i in list(range(0, len(Xprime) - 1)):
        errors.append((np.sqrt((Xprime[i] - a) ** 2 + (Yprime[i] - b) ** 2) - r) ** 2)
    sigma = np.sqrt((sum(errors)) / (len(Xprime) - 1))
    return a, b, r, sigma


def bestfit_line(ptrm, nrm):
    """
    Returns the slope and intercept of the best fit line to a set of
    pTRM and NRM data using York Regression

    Inputs
    ------
    ptrm: list or array
    list of pTRM data

    nrm: list or array
    list of NRM data

    Returns
    -------
    dictionary of slope and intercept for best fitting line.
    """
    stat = sufficient_statistics(ptrm, nrm)

    w = 0.5 * (stat["S2xx"] - stat["S2yy"]) / stat["S2xy"]
    m = -w - np.sqrt(w**2 + 1)
    b = stat["ybar"] - m * stat["xbar"]
    return {"slope": m, "intercept": b}


def get_drat(IZZI, IZZI_trunc, P):
    """Calculates the difference ratio (DRAT) of pTRM checks
    (Selkin and Tauxe, 2000) to check for alteration

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of all in field and
    zero field measurements for a specimen.

    IZZI_trunc: pandas.DataFrame
    DataFrame object- same as IZZI but truncated only for temperatures
    used in interpretation

    P: pandas.DataFrame
    DataFrame object containing pTRM checks up to the
    maximum temperature of an interpretation

    Returns
    -------
    absdiff: float
    maximum DRAT for all pTRM check measurements.
    returns zero if the interpretation is not valid.
    """
    try:
        IZZI_reduced = IZZI[IZZI.temp_step.isin(P.temp_step)]
        a = np.sum(
            (IZZI_trunc.PTRM - np.mean(IZZI_trunc.PTRM))
            * (IZZI_trunc.NRM - np.mean(IZZI_trunc.NRM))
        )
        b = (
            a
            / np.abs(a)
            * np.sqrt(
                np.sum((IZZI_trunc.NRM - np.mean(IZZI_trunc.NRM)) ** 2)
                / np.sum((IZZI_trunc.PTRM - np.mean(IZZI_trunc.PTRM)) ** 2)
            )
        )
        yint = np.mean(IZZI_trunc.NRM) - b * np.mean(IZZI_trunc.PTRM)
        line = {"slope": b, "intercept": yint}

        xprime = 0.5 * (
            IZZI_trunc.PTRM + (IZZI_trunc.NRM - line["intercept"]) / line["slope"]
        )
        yprime = 0.5 * (
            IZZI_trunc.NRM + line["slope"] * IZZI_trunc.PTRM + line["intercept"]
        )
        scalefactor = np.sqrt(
            (min(xprime) - max(xprime)) ** 2 + (min(yprime) - max(yprime)) ** 2
        )
        absdiff = (
            max(np.abs(P.PTRM.values - IZZI_reduced.PTRM.values) / scalefactor) * 100
        )
        return absdiff
    except:
        return 0


def get_mad(IZZI, pca):
    """
    Calculates the free Maximum Angle of Deviation (MAD) of Kirshvink et
    al (1980)

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of in field and
    zero field measurements for a specimen (interpretation).

    pca: scikitlearn.decomposition.PCA object
    pca used to fit the vector direction.

    Returns
    -------
    mad: float
    maximum angle of deviation for that intepretation
    """
    try:
        fit = pca.fit(IZZI.loc[:, "NRM_x":"NRM_z"].values).explained_variance_
        return np.degrees(np.arctan(np.sqrt((fit[2] + fit[1]) / (fit[0]))))
    except:
        return 0


def get_dang(NRM_trunc_dirs, pca):
    """
    Calculates the Deviation Angle
    Inputs
    ------
    NRM_trunc_dirs: numpy.ndarray
    Vector directions for zero field measurements for specimen

    pca: scikitlearn.decomposition.PCA object
    pca used to fit the vector direction.

    Returns
    -------
    dang: float
    deviation angle for that intepretation
    """
    try:
        length, vector = pca.explained_variance_[0], pca.components_[0]
        NRM_vect = np.mean(NRM_trunc_dirs, axis=0)
        NRM_mean_magn = np.sqrt(sum(NRM_vect**2))
        vector_magn = np.sqrt(sum(vector**2))
        return np.degrees(
            np.arccos(np.abs(np.dot(NRM_vect, vector) / (NRM_mean_magn * vector_magn)))
        )
    except:
        return 0


def get_frac(IZZI, IZZI_trunc):
    """
    Calculates the NRM Fraction from VDS (Shaar and Tauxe, 2013)

    Inputs
    ------
    specimen: BiCEP specimen object
    specimen to obtain FRAC from

    Returns
    -------
    FRAC: float
    calculated FRAC
    """
    # Calculate VDS of all steps
    totalsum = np.linalg.norm(
        np.sum(np.diff(IZZI.loc[:, "NRM_x":"NRM_z"].values, axis=0), axis=0)
    ) + np.linalg.norm(IZZI.loc[:, "NRM_x":"NRM_z"].values[-1])

    # Calculate VDS of just our steps
    truncsum = np.linalg.norm(
        np.sum(np.diff(IZZI_trunc.loc[:, "NRM_x":"NRM_z"].values, axis=0), axis=0)
    )

    if all(IZZI_trunc.iloc[-1, :-3] == IZZI.iloc[-1, :-3]):
        truncsum += np.linalg.norm(IZZI.loc[:, "NRM_x":"NRM_z"].values[-1])

    # Calculate FRAC based on sum of points
    FRAC = truncsum / totalsum
    return FRAC


def auto_interpret(site, mad, dang, drat, mad_type="mad_free"):
    """
    Auto-Interpreter for BiCEP. Calculates MAD_coe, DANG and DRAT
    for every possible interpretation with 4 or more points on
    the Arai plot. Selects the interpretation that passes the
    criteria which has the highest FRAC.

    Inputs
    ------
    site: BiCEP specimenCollection object
    Site/sample to calculate interpretations for

    mad: float
    MAD_Coe criterion threshold

    dang: float
    DANG criterion threshold

    DRAT: float
    DRAT criterion threshold

    Returns
    -------
    None
    """
    # Loop through specimens
    for specimen in site.specimens.values():
        # Make lists of lower, upper and fracs
        interps_lower = []
        interps_upper = []
        fracs = []
        lowerTemps = specimen.temps[:-4]

        # Loop through interpretation
        for i in range(len(lowerTemps)):
            lowerTemp = lowerTemps[i]  # Get lower temperature bound
            for upperTemp in specimen.temps[
                i + 3 :
            ]:  # Get upper temperature bound (min 4 steps)

                # Change specimen temperatures
                specimen.change_temps(lowerTemp, upperTemp)

                # Calculate MAD or MAD_Coe
                pca = PCA(n_components=3)
                if mad_type == "mad_coe":
                    mad_steps = specimen.IZZI_trunc[
                        specimen.IZZI_trunc.steptype == "ZI"
                    ]
                elif mad_type == "mad_free":
                    mad_steps = specimen.IZZI_trunc
                else:
                    raise ValueError(
                        "Error: mad_type must be one of mad_coe or mad_free"
                    )
                mad_est = get_mad(mad_steps, pca)

                # Check whether intepretation fits criteria
                if (mad_est < mad) & (specimen.dang < dang) & (specimen.drat < drat):
                    fracs.append(get_frac(specimen.IZZI, specimen.IZZI_trunc))
                    interps_lower.append(lowerTemp)
                    interps_upper.append(upperTemp)

        # Make into arrays
        interps_lower = np.array(interps_lower)
        interps_upper = np.array(interps_upper)
        fracs = np.array(fracs)

        # If nothing passes, (specimens labeled aa were bad in one dataset)
        # Exclude from analysis
        if len(fracs) == 0 or "aa" in specimen.name:
            specimen.active = False  # Exclude

            # Set temperature range to 0,0
            lowerTemp = specimen.temps[0]
            upperTemp = specimen.temps[0]

        # If some things pass, set lower and upper temps to what maximizes FRAC
        if len(fracs) > 0:
            # Add specimen to analysis if previously excluded.
            specimen.active = True
            # Passing interpretation that maximizes FRAC
            lowerTemp = interps_lower[fracs == max(fracs)][0]
            upperTemp = interps_upper[fracs == max(fracs)][0]

        # Save changes to specimens
        specimen.change_temps(lowerTemp, upperTemp)
        specimen.save_changes()


def calculate_anisotropy_correction(IZZI):
    """
    Calculates anisotropy correction factor for a
    paleointensity interpretation, given an s tensor

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of in field and
    zero field measurements for a specimen (interpretation).

    Returns
    -------
    c: float
    Anisotropy correction factor
    """

    # Convert the s tensor into a numpy array
    strlist = IZZI["s_tensor"].iloc[0].split(":")
    slist = []
    for stringo in strlist:
        slist.append(float(stringo.strip()))
    stensor = np.array(
        [
            [slist[0], slist[3], slist[5]],
            [slist[3], slist[1], slist[4]],
            [slist[5], slist[4], slist[2]],
        ]
    )

    # Fit a PCA to the IZZI directions
    NRM_trunc_dirs = IZZI.loc[:, "NRM_x":"NRM_z"]
    pca = PCA(n_components=3)
    pca = pca.fit(NRM_trunc_dirs)

    # Calculate the anisotropy correction factor (see Standard Paleointensity Definitions)
    vector = pca.components_[0]
    vector = vector / np.sqrt(np.sum(vector**2))
    ancvector = np.matmul(np.linalg.inv(stensor), vector)
    ancvector = ancvector / np.sqrt(np.sum(ancvector**2))
    labmag = np.matmul(stensor, np.array([0, 0, -1]))
    ancmag = np.matmul(stensor, ancvector)
    c = np.sqrt(np.sum(labmag**2)) / np.sqrt(np.sum(ancmag**2))
    return c


def calculate_NLT_correction(IZZI, c):
    """
    Calculates the correction for non linear TRM for a paleointensity interpretation,
    given the anisotropy and cooling rate corrections

    Inputs
    ------
    IZZI: pandas.DataFrame
    DataFrame object in BiCEP format of in field and
    zero field measurements for a specimen (interpretation).

    c: float
    Combined Anisotropy and Cooling Rate correction-
    needed because the nonlinearity is applied after this.

    Returns
    -------
    c: float
    NLT correction factor
    """
    a = np.sum((IZZI.PTRM - np.mean(IZZI.PTRM)) * (IZZI.NRM - np.mean(IZZI.NRM)))
    b = (
        a
        / np.abs(a)
        * np.sqrt(
            np.sum((IZZI.NRM - np.mean(IZZI.NRM)) ** 2)
            / np.sum((IZZI.PTRM - np.mean(IZZI.PTRM)) ** 2)
        )
    )
    beta = IZZI["NLT_beta"].iloc[0]
    correction = c * IZZI.correction.iloc[0]
    B_lab = IZZI.B_lab.iloc[0] * 1e6
    total_correction = (np.arctanh(correction * np.abs(b) * np.tanh(beta * B_lab))) / (
        np.abs(b) * beta * B_lab
    )
    return total_correction
