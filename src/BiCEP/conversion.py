import pmagpy.pmag as pmag
import numpy as np
from pmagpy import contribution_builder as cb
import pandas as pd
from scipy.optimize import curve_fit
import pmagpy.ipmag as ipmag
from IPython.display import clear_output


def sortarai(datablock, s, Zdiff, **kwargs):
    """
     sorts data block in to first_Z, first_I, etc.

    Parameters
    _________
    datablock : Pandas DataFrame with Thellier-Tellier type data
    s : specimen name
    Zdiff : if True, take difference in Z values instead of vector difference
            NB:  this should always be False
    **kwargs :
        version : data model.  if not 3, assume data model = 2.5

    Returns
    _______
    araiblock : [first_Z, first_I, ptrm_check,
                 ptrm_tail, zptrm_check, GammaChecks]
    field : lab field (in tesla)
    """
    if "version" in list(kwargs.keys()) and kwargs["version"] == 3:
        dec_key, inc_key, csd_key = "dir_dec", "dir_inc", "dir_csd"
        Mkeys = ["magn_moment", "magn_volume", "magn_mass", "magnitude", "dir_csd"]
        meth_key = "method_codes"
        temp_key, dc_key = "treat_temp", "treat_dc_field"
        dc_theta_key, dc_phi_key = "treat_dc_field_theta", "treat_dc_field_phi"
        # convert dataframe to list of dictionaries
        datablock = datablock.to_dict("records")
    else:
        dec_key, inc_key, csd_key = (
            "measurement_dec",
            "measurement_inc",
            "measurement_csd",
        )
        Mkeys = [
            "measurement_magn_moment",
            "measurement_magn_volume",
            "measurement_magn_mass",
            "measurement_magnitude",
        ]
        meth_key = "magic_method_codes"
        temp_key, dc_key = "treatment_temp", "treatment_dc_field"
        dc_theta_key, dc_phi_key = "treatment_dc_field_theta", "treatment_dc_field_phi"
    first_Z, first_I, zptrm_check, ptrm_check, ptrm_tail = [], [], [], [], []
    field, phi, theta = "", "", ""
    starthere = 0
    Treat_I, Treat_Z, Treat_PZ, Treat_PI, Treat_M = [], [], [], [], []
    ISteps, ZSteps, PISteps, PZSteps, MSteps = [], [], [], [], []
    GammaChecks = []  # comparison of pTRM direction acquired and lab field
    rec = datablock[0]
    for key in Mkeys:
        if key in list(rec.keys()) and rec[key] != "":
            momkey = key
            break
    # first find all the steps
    for k in range(len(datablock)):
        rec = datablock[k]
        temp = float(rec[temp_key])
        methcodes = []
        tmp = rec[meth_key].split(":")
        for meth in tmp:
            methcodes.append(meth.strip())
        if (
            "LT-T-I" in methcodes
            and "LP-TRM" not in methcodes
            and "LP-PI-TRM" in methcodes
        ):
            Treat_I.append(temp)
            ISteps.append(k)
            if field == "":
                field = float(rec[dc_key])
            if phi == "":
                phi = float(rec[dc_phi_key])
                theta = float(rec[dc_theta_key])
        # stick  first zero field stuff into first_Z
        if "LT-NO" in methcodes:
            Treat_Z.append(temp)
            ZSteps.append(k)
        if "LT-T-Z" in methcodes:
            Treat_Z.append(temp)
            ZSteps.append(k)
        if "LT-PTRM-Z" in methcodes:
            Treat_PZ.append(temp)
            PZSteps.append(k)
        if "LT-PTRM-I" in methcodes:
            Treat_PI.append(temp)
            PISteps.append(k)
        if "LT-PTRM-MD" in methcodes:
            Treat_M.append(temp)
            MSteps.append(k)
        if "LT-NO" in methcodes:
            dec = float(rec[dec_key])
            inc = float(rec[inc_key])
            str = float(rec[momkey])
            if csd_key not in rec.keys():
                sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            elif rec[csd_key] != None:
                sig = np.radians(float(rec[csd_key])) * np.sqrt(3) / np.sqrt(2) * str
            else:
                sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            first_I.append([273, 0.0, 0.0, 0.0, 0.0, 1])
            first_Z.append([273, dec, inc, str, sig, 1])  # NRM step
    for temp in Treat_I:  # look through infield steps and find matching Z step
        if temp in Treat_Z:  # found a match
            istep = ISteps[Treat_I.index(temp)]
            irec = datablock[istep]
            methcodes = []
            tmp = irec[meth_key].split(":")
            for meth in tmp:
                methcodes.append(meth.strip())
            # take last record as baseline to subtract
            brec = datablock[istep - 1]
            zstep = ZSteps[Treat_Z.index(temp)]
            zrec = datablock[zstep]
            # sort out first_Z records
            if "LP-PI-TRM-IZ" in methcodes:
                ZI = 0
            else:
                ZI = 1
            dec = float(zrec[dec_key])
            inc = float(zrec[inc_key])
            str = float(zrec[momkey])
            if csd_key not in rec.keys():
                sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            elif rec[csd_key] != None:

                sig = np.radians(float(rec[csd_key])) * np.sqrt(3) / np.sqrt(2) * str
            else:
                sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            first_Z.append([temp, dec, inc, str, sig, ZI])

            # sort out first_I records
            idec = float(irec[dec_key])
            iinc = float(irec[inc_key])
            istr = float(irec[momkey])
            X = pmag.dir2cart([idec, iinc, istr])
            BL = pmag.dir2cart([dec, inc, str])
            I = []
            for c in range(3):
                I.append((X[c] - BL[c]))
            if I[2] != 0:
                iDir = pmag.cart2dir(I)
                if csd_key not in rec.keys():
                    isig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
                elif rec[csd_key] != None:
                    isig = (
                        np.radians(float(rec[csd_key])) * np.sqrt(3) / np.sqrt(2) * istr
                    )
                else:
                    isig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * istr
                isig = np.sqrt(isig**2 + sig**2)

                if Zdiff == 0:
                    first_I.append([temp, iDir[0], iDir[1], iDir[2], isig, ZI])
                else:
                    first_I.append([temp, 0.0, 0.0, I[2], 0.0, isig, ZI])

                gamma = pmag.angle([iDir[0], iDir[1]], [phi, theta])
            else:
                first_I.append([temp, 0.0, 0.0, 0.0, 0.0, ZI])
                gamma = 0.0
            # put in Gamma check (infield trm versus lab field)
            if 180.0 - gamma < gamma:
                gamma = 180.0 - gamma
            GammaChecks.append([temp - 273.0, gamma])
    for temp in Treat_PI:  # look through infield steps and find matching Z step
        step = PISteps[Treat_PI.index(temp)]
        rec = datablock[step]
        dec = float(rec[dec_key])
        inc = float(rec[inc_key])
        str = float(rec[momkey])

        brec = datablock[step - 1]  # take last record as baseline to subtract
        btemp = float(brec[temp_key])
        pdec = float(brec[dec_key])
        pinc = float(brec[inc_key])
        pint = float(brec[momkey])
        X = pmag.dir2cart([dec, inc, str])
        prevX = pmag.dir2cart([pdec, pinc, pint])
        I = []
        for c in range(3):
            I.append(X[c] - prevX[c])
        dir1 = pmag.cart2dir(I)
        if csd_key not in rec.keys():
            sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            psig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * dir1[2]
        elif rec[csd_key] != None:
            sig = np.radians(float(rec[csd_key])) * np.sqrt(3) / np.sqrt(2) * str
            psig = np.radians(float(brec[csd_key])) * np.sqrt(3) / np.sqrt(2) * dir1[2]
        else:
            sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            psig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * dir1[2]
        psig = np.sqrt(sig**2 + psig**2)
        if Zdiff == 0:
            ptrm_check.append([temp, dir1[0], dir1[1], dir1[2], sig, btemp])
        else:
            ptrm_check.append([temp, 0.0, 0.0, I[2]], sig)
    # in case there are zero-field pTRM checks (not the SIO way)
    for temp in Treat_PZ:
        logstring += str(temp)
        logstring += str(treat_pz) + "\n"
        step = PZSteps[Treat_PZ.index(temp)]
        rec = datablock[step]
        dec = float(rec[dec_key])
        inc = float(rec[inc_key])
        str = float(rec[momkey])
        brec = datablock[step - 1]
        btemp = float(brec[temp_key])
        pdec = float(brec[dec_key])
        pinc = float(brec[inc_key])
        pint = float(brec[momkey])
        X = pmag.dir2cart([dec, inc, str])
        prevX = pmag.dir2cart([pdec, pinc, pint])
        I = []
        for c in range(3):
            I.append(X[c] - prevX[c])
        dir2 = pmag.cart2dir(I)
        if csd_key not in rec.keys():
            sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            psig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * dir1[2]
        elif rec[csd_key] != None:
            sig = np.radians(float(rec[csd_key])) * np.sqrt(3) / np.sqrt(2) * str
            psig = np.radians(float(brec[csd_key])) * np.sqrt(3) / np.sqrt(2) * dir2[2]
        else:
            sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
            psig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * dir1[2]
        psig = np.sqrt(sig**2 + psig**2)
        zptrm_check.append([temp, dir2[0], dir2[1], dir2[2], psig, btemp])
    # get pTRM tail checks together -
    for temp in Treat_M:
        # tail check step - just do a difference in magnitude!
        step = MSteps[Treat_M.index(temp)]
        rec = datablock[step]
        dec = float(rec[dec_key])
        inc = float(rec[inc_key])
        str = float(rec[momkey])
        brec = datablock[step - 1]  # take last record as baseline to subtract
        btemp = float(brec[temp_key])
        if csd_key not in rec.keys():
            sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
        elif rec[csd_key] != None:
            sig = np.radians(float(rec[csd_key])) * np.sqrt(3) / np.sqrt(2) * str
        else:
            sig = np.radians(2) * np.sqrt(3) / np.sqrt(2) * str
        if temp in Treat_Z:
            step = ZSteps[Treat_Z.index(temp)]
            brec = datablock[step]
            pint = float(brec[momkey])
            #        X=dir2cart([dec,inc,str])
            #        prevX=dir2cart([pdec,pinc,pint])
            #        I=[]
            #        for c in range(3):I.append(X[c]-prevX[c])
            #        d=cart2dir(I)
            #        ptrm_tail.append([temp,d[0],d[1],d[2]])
            # difference - if negative, negative tail!
            ptrm_tail.append([temp, dec, inc, str, sig, btemp])
        else:
            logstring += (
                s
                + "  has a tail check with no first zero field step - check input file! for step"
                + str(temp - 273.0)
                + "\n"
            )

    #
    # final check
    #
    if len(first_Z) != len(first_I):
        logstring += " Something wrong with this specimen! Better fix it or delete it "
    araiblock = (first_Z, first_I, ptrm_check, ptrm_tail, zptrm_check, GammaChecks)
    return araiblock, field


def NLTsolver(fields, a, b):
    """Makes the non linear TRM correction"""
    return a * np.tanh(b * fields)


def convert_intensity_measurements(measurements):
    """Converts a measurements table with only intensity experiments into the internal data format used by the BiCEP method"""
    specimens = list(
        measurements.specimen.unique()
    )  # This function constructs the 'temps' dataframe (used to plot Arai plots)
    # this may take a while to run depending on the number of specimens.
    # Constructs initial empty 'temps' dataframe
    data_array = np.empty(shape=(17, 0))
    logstring = ""
    for specimen in specimens:
        logstring += "Working on: " + specimen + "\n"
        clear_output(wait=True)
        print("Working on: " + specimen)
        try:
            araiblock, field = sortarai(
                measurements[measurements.specimen == specimen],
                specimen,
                Zdiff=False,
                version=3,
            )  # Get arai data
            sitename = measurements[measurements.specimen == specimen].site.unique()
            first_Z, first_I, ptrm_check, ptrm_tail, zptrm_check, GammaChecks = (
                araiblock  # Split NRM and PTRM values into step types
            )
            B_lab = np.full(len(first_Z), field)  # Lab field used
            m = len(first_Z)
            NRM_dec_max = first_Z[m - 1][1]
            NRM_inc_max = first_Z[m - 1][2]
            NRM_int_max = first_Z[m - 1][3]
            PTRM_dec_max = first_I[m - 1][1]
            PTRM_inc_max = first_I[m - 1][2]
            PTRM_int_max = first_I[m - 1][3]
            NRM_vector_max = pmag.dir2cart([NRM_dec_max, NRM_inc_max, NRM_int_max])
            PTRM_vector_max = pmag.dir2cart([PTRM_dec_max, PTRM_inc_max, PTRM_int_max])
            PTRM_vector_max = PTRM_vector_max - NRM_vector_max
            NRMS = [first_Z[i][3] for i in list(range(len(first_Z)))]
            first_Z = np.array(first_Z)
            first_I = np.array(first_I)

            if min(NRMS) / NRMS[0] < 0.25:
                if (len(first_Z)) > 1:
                    sample = np.full(
                        len(first_Z),
                        measurements[measurements.specimen == specimen][
                            "sample"
                        ].unique()[0],
                    )  # Get sample name
                    site = np.full(
                        len(first_Z),
                        measurements[measurements.specimen == specimen].site.unique()[
                            0
                        ],
                    )  # Get site name
                    specarray = np.full(len(first_Z), specimen)
                    temp_step = first_Z[:, 0]  # Gets the temperature in kelvin we use
                    NRM = first_Z[:, 3]  # NRM value (in first_Z dataframe)
                    zbinary = first_Z[:, 5]  # Is it a ZI or an IZ step?
                    zbinary = zbinary.astype("object")
                    zbinary[zbinary == 1] = "ZI"
                    zbinary[zbinary == 0] = "IZ"
                    steptype = zbinary
                    PTRM = first_I[:, 3]  # PTRM value (in first_I dataframe)
                    PTRM_sigma = first_I[:, 4]

                    NRM_dec = first_Z[:, 1]
                    NRM_inc = first_Z[:, 2]
                    NRM_int = NRM
                    NRM_sigma = first_Z[:, 4]

                    NRM_vector = pmag.dir2cart(np.array([NRM_dec, NRM_inc, NRM_int]).T)
                    PTRM_vector = pmag.dir2cart(
                        np.array([first_I[:, 1], first_I[:, 2], first_I[:, 3]]).T
                    )

                    NRM_x = NRM_vector[:, 0]
                    NRM_y = NRM_vector[:, 1]
                    NRM_z = NRM_vector[:, 2]

                    PTRM_x = PTRM_vector[:, 0]
                    PTRM_y = PTRM_vector[:, 1]
                    PTRM_z = PTRM_vector[:, 2]
                    baseline_temp = np.full(len(specarray), np.nan)

                    newarray = np.array(
                        [
                            specarray,
                            sample,
                            site,
                            NRM,
                            PTRM,
                            NRM_x,
                            NRM_y,
                            NRM_z,
                            PTRM_x,
                            PTRM_y,
                            PTRM_z,
                            NRM_sigma,
                            PTRM_sigma,
                            B_lab,
                            steptype,
                            temp_step,
                            baseline_temp,
                        ]
                    )
                    data_array = np.concatenate((data_array, newarray), axis=1)

                    # Doing PTRM Checks Part
                    ptrm_check = np.array(ptrm_check)
                    temp_step = ptrm_check[:, 0]
                    smallarray = data_array
                    sample = np.full(
                        len(ptrm_check),
                        measurements[measurements.specimen == specimen][
                            "sample"
                        ].unique()[0],
                    )  # Get sample name
                    site = np.full(
                        len(ptrm_check),
                        measurements[measurements.specimen == specimen].site.unique()[
                            0
                        ],
                    )  # Get site name
                    specarray = np.full(len(ptrm_check), specimen)
                    B_lab = np.full(len(ptrm_check), field)
                    PTRM = ptrm_check[:, 3]
                    PTRM_sigma = ptrm_check[:, 4]
                    intersect = data_array[
                        :,
                        (data_array[0] == specimen)
                        & (
                            np.in1d(
                                data_array[-2].astype("float"),
                                temp_step.astype("float"),
                            )
                        ),
                    ]
                    NRM_vector = np.array([intersect[5], intersect[6], intersect[7]])
                    NRM_sigma = intersect[11]
                    PTRM_vector = pmag.dir2cart(
                        np.array(
                            [ptrm_check[:, 1], ptrm_check[:, 2], ptrm_check[:, 3]]
                        ).T
                    )
                    NRM_x = NRM_vector[0]
                    NRM_y = NRM_vector[1]
                    NRM_z = NRM_vector[2]

                    PTRM_x = PTRM_vector[:, 0]
                    PTRM_y = PTRM_vector[:, 1]
                    PTRM_z = PTRM_vector[:, 2]
                    NRM = intersect[3]
                    steptype = np.full(len(ptrm_check), "P")
                    baseline_temp = ptrm_check[:, -1]

                    if len(NRM) == len(PTRM):

                        newarray = np.array(
                            [
                                specarray,
                                sample,
                                site,
                                NRM,
                                PTRM,
                                NRM_x,
                                NRM_y,
                                NRM_z,
                                PTRM_x,
                                PTRM_y,
                                PTRM_z,
                                NRM_sigma,
                                PTRM_sigma,
                                B_lab,
                                steptype,
                                temp_step,
                                baseline_temp,
                            ]
                        )
                        data_array = np.concatenate((data_array, newarray), axis=1)
                    else:
                        diff = np.setdiff1d(temp_step, intersect[-2])
                        for i in diff:
                            logstring += (
                                "PTRM check at "
                                + str(i)
                                + "K has no corresponding zero field measurement, ignoring"
                                + "\n"
                            )
                        newarray = np.array(
                            [
                                specarray[temp_step != diff],
                                sample[temp_step != diff],
                                site[temp_step != diff],
                                NRM,
                                PTRM[temp_step != diff],
                                NRM_x,
                                NRM_y,
                                NRM_z,
                                PTRM_x[temp_step != diff],
                                PTRM_y[temp_step != diff],
                                PTRM_z[temp_step != diff],
                                NRM_sigma,
                                PTRM_sigma[temp_step != diff],
                                B_lab[temp_step != diff],
                                steptype[temp_step != diff],
                                temp_step[temp_step != diff],
                                baseline_temp[temp_step != diff],
                            ]
                        )
                        data_array = np.concatenate((data_array, newarray), axis=1)

                    # Add PTRM tail checks
                    ptrm_tail = np.array(ptrm_tail)

                    if len(ptrm_tail) > 1:
                        temp_step = ptrm_tail[:, 0]
                        sample = np.full(
                            len(ptrm_tail),
                            measurements[measurements.specimen == specimen][
                                "sample"
                            ].unique()[0],
                        )  # Get sample name
                        site = np.full(
                            len(ptrm_tail),
                            measurements[
                                measurements.specimen == specimen
                            ].site.unique()[0],
                        )  # Get site name
                        specarray = np.full(len(ptrm_tail), specimen)
                        B_lab = np.full(len(ptrm_tail), field)
                        intersect = data_array[
                            :,
                            (data_array[0] == specimen)
                            & (
                                np.in1d(
                                    data_array[-2].astype("float"),
                                    temp_step.astype("float"),
                                )
                            )
                            & (data_array[-3] != "P"),
                        ]

                        NRM = ptrm_tail[:, 3]
                        NRM_sigma = ptrm_tail[:, 4]
                        NRM_vector = pmag.dir2cart(
                            np.array(
                                [ptrm_tail[:, 1], ptrm_tail[:, 2], ptrm_tail[:, 3]]
                            ).T
                        )
                        PTRM_vector = np.array(
                            [intersect[8], intersect[9], intersect[10]]
                        )
                        PTRM_sigma = intersect[12]
                        PTRM_x = PTRM_vector[0]
                        PTRM_y = PTRM_vector[1]
                        PTRM_z = PTRM_vector[2]
                        NRM_x = NRM_vector[:, 0]
                        NRM_y = NRM_vector[:, 1]
                        NRM_z = NRM_vector[:, 2]
                        PTRM = intersect[4]

                        steptype = np.full(len(ptrm_tail), "T")
                        baseline_temp = ptrm_tail[:, -1]

                        if len(PTRM) == len(NRM):
                            newarray = np.array(
                                [
                                    specarray,
                                    sample,
                                    site,
                                    NRM,
                                    PTRM,
                                    NRM_x,
                                    NRM_y,
                                    NRM_z,
                                    PTRM_x,
                                    PTRM_y,
                                    PTRM_z,
                                    NRM_sigma,
                                    PTRM_sigma,
                                    B_lab,
                                    steptype,
                                    temp_step,
                                    baseline_temp,
                                ]
                            )
                            data_array = np.concatenate((data_array, newarray), axis=1)
                        else:
                            diff = np.setdiff1d(temp_step, intersect[-2])
                            for i in diff:
                                logstring += (
                                    "PTRM tail check at "
                                    + str(i)
                                    + "K has no corresponding in field measurement, ignoring"
                                    + "\n"
                                )
                            newarray = np.array(
                                [
                                    specarray[temp_step != diff],
                                    sample[temp_step != diff],
                                    site[temp_step != diff],
                                    NRM[temp_step != diff],
                                    PTRM,
                                    NRM_x[temp_step != diff],
                                    NRM_y[temp_step != diff],
                                    NRM_z[temp_step != diff],
                                    PTRM_x,
                                    PTRM_y,
                                    PTRM_z,
                                    NRM_sigma[temp_step != diff],
                                    PTRM_sigma,
                                    B_lab[temp_step != diff],
                                    steptype[temp_step != diff],
                                    temp_step[temp_step != diff],
                                    baseline_temp[temp_step != diff],
                                ]
                            )
                            data_array = np.concatenate((data_array, newarray), axis=1)

                else:
                    logstring += (
                        specimen
                        + " in site "
                        + sitename[0]
                        + " Not included, not a thellier experiment"
                        + "\n"
                    )
            else:
                logstring += (
                    specimen
                    + " in site "
                    + sitename[0]
                    + " Not included, demagnetization not completed"
                    + "\n"
                )
        except:
            logstring += (
                "Something went wrong with specimen "
                + specimen
                + ". Could not convert from MagIC format"
                + "\n"
            )
    temps = pd.DataFrame(
        data_array.T,
        columns=[
            "specimen",
            "sample",
            "site",
            "NRM",
            "PTRM",
            "NRM_x",
            "NRM_y",
            "NRM_z",
            "PTRM_x",
            "PTRM_y",
            "PTRM_z",
            "NRM_sigma",
            "PTRM_sigma",
            "B_lab",
            "steptype",
            "temp_step",
            "baseline_temp",
        ],
    )
    return (temps, logstring)


def generate_arai_plot_table(outputname, wd="./"):
    """
    Generates a DataFrame with Thellier Data for a Dataset, stores it as a csv.

    Inputs
    ------
    outputname: (str)
    name of file to output (no extension)

    Returns
    -------
    None
    """
    # This cell constructs the 'measurements' dataframe with samples and sites added
    logstring = ""
    status, measurements = cb.add_sites_to_meas_table(wd)
    try:
        measurements = measurements[measurements.specimen.str.contains("#") == False]
    except AttributeError:
        raise FileNotFoundError(
            "No MagIC data files found, please put measurements, specimens, samples and sites tables in this folder"
        )
    measurements_old = measurements
    measurements = measurements[measurements.method_codes.str.contains("LP-PI-TRM")]
    temps, output = convert_intensity_measurements(measurements)

    logstring += output
    clear_output(wait=True)

    temps["correction"] = 1
    temps["s_tensor"] = np.nan
    temps["aniso_type"] = np.nan

    spec = pd.read_csv(wd + "specimens.txt", skiprows=1, sep="\t")

    # Create the anisotropy tensors if they don't already exist.
    logstring += "Couldn't find Anisotropy Tensors, Generating..." + "\n"

    # Tensor for ATRM
    ipmag.atrm_magic("measurements.txt", dir_path=wd)
    try:
        spec_atrm = pd.read_csv(wd + "specimens.txt", sep="\t", skiprows=1)
        spec_atrm = spec_atrm.dropna(subset=["method_codes"])
        spec_atrm = spec_atrm[spec_atrm.method_codes.str.contains("LP-AN-TRM")]
        for specimen in spec_atrm.specimen.unique():
            temps.loc[temps.specimen == specimen, "s_tensor"] = spec_atrm.loc[
                spec_atrm.specimen == specimen, "aniso_s"
            ].iloc[0]
            temps.loc[temps.specimen == specimen, "aniso_type"] = ":DA-AC-ATRM"
    except:
        pass
    # Tensor for AARM
    ipmag.aarm_magic("measurements.txt", dir_path=wd)
    try:
        spec_aarm = pd.read_csv(wd + "specimens.txt", sep="\t", skiprows=1)
        spec_aarm = spec_aarm.dropna(subset=["method_codes"])
        spec_aarm = spec_aarm[spec_aarm.method_codes.str.contains("LP-AN-ARM")]
        for specimen in spec_aarm.specimen.unique():
            temps.loc[temps.specimen == specimen, "s_tensor"] = spec_aarm.loc[
                spec_aarm.specimen == specimen, "aniso_s"
            ].iloc[0]
            temps.loc[temps.specimen == specimen, "aniso_type"] = ":DA-AC-AARM"
    except:
        pass

    # Get the best fitting hyperbolic tangent for the NLT correction.
    temps["NLT_beta"] = np.nan
    NLTcorrs = measurements_old[measurements_old["method_codes"] == "LP-TRM:LT-T-I"]
    for specimen in NLTcorrs.specimen.unique():
        meas_val = NLTcorrs[NLTcorrs["specimen"] == specimen]
        try:
            meas_val["magn_moment"] = meas_val["magn_moment"].astype(float)
            meas_val["treat_dc_field"] = meas_val["treat_dc_field"].astype(float)
            ab, cov = curve_fit(
                NLTsolver,
                meas_val["treat_dc_field"].values * 1e6,
                meas_val["magn_moment"].values / meas_val["magn_moment"].iloc[-1],
                p0=(
                    max(meas_val["magn_moment"] / meas_val["magn_moment"].iloc[-1]),
                    1e-2,
                ),
            )
            temps.loc[temps.specimen == specimen, "NLT_beta"] = ab[1]
        except RuntimeError:
            logstring += (
                "-W- WARNING: Can't fit tanh function to NLT data for "
                + specimen
                + "\n"
            )

    # Get the cooling rate correction
    try:
        meas_cool = measurements_old[
            measurements_old.method_codes.str.contains("CR-TRM")
        ].dropna(subset=["description"])
        meas_cool = meas_cool[
            meas_cool.method_codes.str.contains("LT-T-Z") == False
        ]  # Ignores zero field cooling rate measurements
        samples = pd.read_csv(wd + "samples.txt", skiprows=1, sep="\t")
        samples = samples.dropna(
            subset=["cooling_rate"]
        )  # Get only things with cooling rates from sample table
        for specimen in meas_cool.specimen.unique():
            specframe = meas_cool[meas_cool.specimen == specimen]
            vals = specframe.description.str.split(
                ":"
            ).values  # Lab cooling rates used in "description" column.
            crs = np.array([])
            for val in vals:
                crs = np.append(crs, float(val[1]))
            magn_moments = specframe["magn_moment"].astype(float).values
            avg_moment = np.mean(magn_moments[crs == max(crs)])
            norm_moments = magn_moments / avg_moment
            croven = max(crs)
            crlog = np.log(croven / crs)
            try:
                specframe["cooling_rate"] = specframe.cooling_rate.astype(
                    float
                )  # Original cooling rate from samples table
            except AttributeError:
                try:
                    m, c = np.polyfit(crlog, norm_moments, 1)
                    sample = specframe["sample"].iloc[0]
                    cr_real = (
                        samples[samples["sample"] == sample].cooling_rate.values
                        / 5.256e11
                    )
                    cr_reallog = np.log(croven / cr_real)
                    cfactor = 1 / (c + m * cr_reallog)[0]
                    temps.loc[temps.specimen == specimen, "correction"] *= cfactor
                except AttributeError:
                    logstring += (
                        "Cooling rate correction for specimen "
                        + specimen
                        + " could not be calculated, original cooling rate unknown. Please add the original cooling rate (K/min) to a cooling_rate column in the specimens table. \n"
                    )
                except:
                    logstring += (
                        "Something went wrong with estimating the cooling rate correction for specimen "
                        + specimen
                        + ". Check that you used the right cooling rate."
                        + "\n"
                    )
    except KeyError:
        logstring += "Measurements file does not contain a description for cooling rate corrections. Ignoring corrections. \n"
    # Save the dataframe to output.
    logfile = open(wd + "thellier_convert.log", "w")
    logfile.write(logstring)
    logfile.close()
    clear_output(wait=True)
    print("Data conversion finished- check thellier_convert.log for errors")
    temps = temps.dropna(subset=["site"])
    temps.to_csv(wd + outputname + ".csv", index=False)
