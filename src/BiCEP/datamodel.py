import nest_asyncio
from numpy.linalg import LinAlgError

nest_asyncio.apply()
import pandas as pd
import numpy as np
import arviz as az
import pmagpy.pmag as pmag
import matplotlib.pyplot as plt
from BiCEP.criteria import *
from BiCEP.extract import *
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from BiCEP import model_circle_fast, model_circle_slow, __version__
from os.path import dirname
from cmdstanpy import CmdStanModel


class ThellierData:
    """
    Class which supports several methods using the BiCEP method in pandas.
    Inputs
    ------
    datafile: string for file name in BiCEP format
    """

    def __init__(self, datafile):
        self.data = pd.read_csv(datafile)
        self.wd = dirname(datafile) + "/"
        if self.wd == "/":
            self.wd = "./"
        self.groupType = "site"
        try:
            self.redo = pd.read_csv(
                self.wd + "bicep_gui.redo", delim_whitespace=True, header=None
            )
        except:
            self.redo = None

        self.collections = {
            siteName: SpecimenCollection(self, siteName, self.groupType)
            for siteName in self.data[self.groupType].unique()
        }

    def __repr__(self):
        reprstr = (
            "Set of Thellier Data Containing the " + "S" + self.groupType[1:] + "s:\n"
        )
        for key in self.collections.keys():
            reprstr += (
                key
                + "\t("
                + str(len(self.collections[key].specimens))
                + " specimens)\n"
            )
        return reprstr

    def __getitem__(self, item):
        return self.collections[item]

    def switch_grouping(self):
        if self.groupType == "site":
            self.groupType = "sample"
        else:
            self.groupType = "site"
        self.collections = {
            siteName: SpecimenCollection(self, siteName, self.groupType)
            for siteName in self.data[self.groupType].unique()
        }


class SpecimenCollection:
    """
    Collection of specimens (site or sample, from ThellierData Dataset)

    Parameters
    ----------

    parentData: ThellierData object
    Set of Thellier Data the site/sample is derived from

    collectionName: string
    Name of specimen/site

    key: string
    Either 'site' for site or 'sample' for sample
    """

    def __init__(self, parentData, collectionName, key):
        self.name = collectionName
        self.key = key
        self.parentData = parentData
        self.data = parentData.data[parentData.data[self.key] == collectionName]
        self.specimens = {
            specimenName: Specimen(self, specimenName)
            for specimenName in self.data.specimen.unique()
        }
        try:
            self.fit = az.from_netcdf(self.parentData.wd + self.name + ".nc")
        except:
            self.fit = None
        self.methcodes = "IE-BICEP"
        self.artist = None

    def __repr__(self):
        reprstr = "Site containing the specimens:\n"
        for specimenName in self.specimens.keys():
            reprstr += specimenName + "\n"
        return reprstr

    def __getitem__(self, item):
        return self.specimens[item]

    def BiCEP_fit(self, n_samples=30000, priorstd=5, model=None, **kwargs):
        """
        Performs the fitting routine using the BiCEP method for a given list of specimens from a single site.
        Inputs
        ------
        Specimenlist: iterable of specimen names (strings)

        """
        minPTRMs = []
        minNRMs = []
        B_lab_list = []
        klist = []
        NRM0s = []
        pTRMsList = np.array([])
        NRMsList = np.array([])
        lengths = []
        philist = []
        dist_to_edgelist = []
        B_ancs = []
        dmaxlist = []
        PTRMmaxlist = []
        centroidlist = []
        i = 0
        # try:
        for specimen in self.specimens.values():
            if specimen.active == True:
                specimen.save_changes()
                minPTRM, minNRM, PTRMmax, k, phi, dist_to_edge, sigma, PTRMS, NRMS = (
                    specimen.BiCEP_prep()
                )
                NRM0 = specimen.NRM0
                minPTRMs.append(minPTRM)
                minNRMs.append(minNRM)
                line = bestfit_line(specimen.IZZI_trunc.PTRM, specimen.IZZI_trunc.NRM)
                B_anc = (
                    -line["slope"]
                    * specimen.B_lab
                    * specimen.IZZI_trunc.correction.iloc[0]
                )
                B_ancs.append(B_anc)
                Pi, Pj = np.meshgrid(PTRMS, PTRMS)
                Ni, Nj = np.meshgrid(NRMS, NRMS)
                dmax = np.amax(np.sqrt((Pi - Pj) ** 2 + (Ni - Nj) ** 2))
                centroid = np.sqrt(np.mean(PTRMS) ** 2 + np.mean(NRMS) ** 2)
                B_lab_list.append(specimen.B_lab)
                klist.append(k)
                philist.append(phi)
                dist_to_edgelist.append(dist_to_edge)
                NRM0s.append(NRM0)
                pTRMsList = np.append(pTRMsList, PTRMS)
                NRMsList = np.append(NRMsList, NRMS)
                lengths.append(int(len(PTRMS)))
                dmaxlist.append(dmax)
                PTRMmaxlist.append(PTRMmax)
                centroidlist.append(centroid)
                i += 1

        if model == None:
            if i < 7:
                model_circle = model_circle_slow
            else:
                model_circle = model_circle_fast
        else:
            model_circle = model

        model_data = {
            "I": len(pTRMsList),
            "M": len(lengths),
            "PTRM": pTRMsList,
            "NRM": NRMsList,
            "N": lengths,
            "PTRMmax": PTRMmaxlist,
            "B_labs": B_lab_list,
            "dmax": np.sqrt(dmaxlist),
            "centroid": centroidlist,
            "priorstd": priorstd,
        }

        model_init = [
            {
                "k_scale": np.array(klist) * np.array(dist_to_edgelist),
                "phi": philist,
                "dist_to_edge": dist_to_edgelist,
                "int_real": B_ancs,
            }
        ] * 4

        fit_circle = model.sample(
            data=model_data,
            inits=model_init,
            iter_sampling=n_samples,
            iter_warmup=int(n_samples / 2),
            **kwargs,
        )

        self.fit = az.from_cmdstanpy(fit_circle)

    def save_magic_tables(self):
        """
        Saves data from the currently displayed site to the GUI

        Inputs
        ------
        None

        Returns
        -------
        None
        """
        fit = self.fit
        wd = self.parentData.wd
        sitestable = pd.read_csv(wd + self.key + "s.txt", skiprows=1, sep="\t")
        # If no method codes row add one
        for key in [
            "method_codes",
            "int_abs",
            "int_abs_sigma",
            "int_abs_sigma_perc",
            "int_abs_min",
            "int_abs_max",
            "vadm",
            "vadm_sigma",
        ]:
            if key not in sitestable.columns:
                sitestable[key] = np.nan

        # Check if a row already exists for this site
        sitesfilter = (sitestable[self.key] == self.name) & (
            (
                sitestable["method_codes"]
                .astype(str)
                .str.contains("IE-BICEP")
                .fillna(False)
            )
            | (sitestable.method_codes.apply(type) == float)
        )
        # If there are no BiCEP data, add a new line
        if len(sitestable[sitesfilter]) == 0:

            new_row_dict = {}
            for key in sitestable.columns:
                req_keys = [
                    "site",
                    "location",
                    "citations",
                    "geologic_classes",
                    "lithologies",
                    "geologic_types",
                    "lat",
                    "lon",
                    "age",
                    "age_low",
                    "age_high",
                    "age_unit",
                ]
                if key in req_keys:
                    value = sitestable.loc[sitestable[self.key] == self.name, key].iloc[
                        0
                    ]
                elif key == "method_codes":
                    value = self.methcodes
                elif key == "software_packages":
                    value = "BiCEP_GUI-" + __version__
                else:
                    value = np.nan
                new_row_dict[key] = value
            sitestable = pd.concat(
                [sitestable, pd.DataFrame(new_row_dict, index=[0])], ignore_index=True
            )

        # We have to redo the filter because it's the wrong shape otherwise
        sitesfilter = (sitestable[self.key] == self.name) & (
            (sitestable["method_codes"].str.contains("IE-BICEP").fillna(False))
            | (sitestable.method_codes.apply(type) == float)
        )

        extract_and_round = lambda param, percentile: round(
            np.percentile(extract_values(fit, param), percentile), 1
        )

        sitestable.loc[sitesfilter, "int_abs_min"] = (
            extract_and_round("int_site", 2.5) / 1e6
        )
        sitestable.loc[sitesfilter, "int_abs_max"] = (
            extract_and_round("int_site", 97.5) / 1e6
        )
        sitestable.loc[sitesfilter, "int_abs"] = extract_and_round("int_site", 50) / 1e6
        sitestable.loc[sitesfilter, "int_abs_sigma"] = (
            sitestable.loc[sitesfilter, "int_abs_max"]
            - sitestable.loc[sitesfilter, "int_abs_min"]
        ) / 4
        sitestable.loc[sitesfilter, "int_abs_sigma_perc"] = np.round(
            sitestable.loc[sitesfilter, "int_abs_sigma"]
            / sitestable.loc[sitesfilter, "int_abs"]
            * 100,
            2,
        )

        if not np.any(np.isnan(sitestable.loc[sitesfilter, "lat"])):
            sitestable.loc[sitesfilter, "vadm"] = np.round(
                pmag.b_vdm(
                    sitestable.loc[sitesfilter, "int_abs"],
                    sitestable.loc[sitesfilter, "lat"],
                ),
                2,
            )
            sitestable.loc[sitesfilter, "vadm_sigma"] = np.round(
                pmag.b_vdm(
                    sitestable.loc[sitesfilter, "int_abs_sigma"],
                    sitestable.loc[sitesfilter, "lat"],
                ),
                2,
            )

        sitestable.loc[sitesfilter, "software_packages"] = "BiCEP_GUI-" + __version__

        specimenstable = pd.read_csv(wd + "specimens.txt", skiprows=1, sep="\t")
        speclist = [
            spec
            for spec in self.specimens.keys()
            if self.specimens[spec].active == True
        ]

        for i in range(len(speclist)):
            specimen = speclist[i]
            specfilter = (specimenstable.specimen == specimen) & (
                (specimenstable.method_codes.str.contains("IE-BICEP").fillna(False))
                | (specimenstable.method_codes.apply(type) == float)
            )

            # If there are no BiCEP data, use make a new line in the dataframe.
            if len(specimenstable[specfilter]) == 0:
                new_row_dict = {}
                for key in specimenstable.columns:
                    req_keys = [
                        "citations",
                        "geologic_classes",
                        "lithologies",
                        "geologic_types",
                        "sample",
                        "specimen",
                    ]
                    if key in req_keys:
                        value = specimenstable.loc[
                            specimenstable.specimen == specimen, key
                        ].iloc[0]
                    elif key == "method_codes":
                        value = self[specimen].methcodes
                    elif key == "software_packages":
                        value = "BiCEP_GUI-" + __version__
                    else:
                        value = np.nan
                    new_row_dict[key] = value
                specimenstable = pd.concat(
                    [specimenstable, pd.DataFrame(new_row_dict, index=[0])],
                    ignore_index=True,
                )

                specfilter = (specimenstable.specimen == specimen) & (
                    (specimenstable.method_codes.str.contains("IE-BICEP").fillna(False))
                    | (specimenstable.method_codes.apply(type) == float)
                )
            extract_and_round = lambda param, percentile, j, prec: round(
                np.percentile(extract_values(fit, param)[j], percentile), prec
            )

            specimenstable.loc[specfilter, "int_abs_min"] = (
                extract_and_round("int_real", 2.5, i, 1) / 1e6
            )
            specimenstable.loc[specfilter, "int_abs_max"] = (
                extract_and_round("int_real", 97.5, i, 1) / 1e6
            )
            specimenstable.loc[specfilter, "int_abs"] = (
                extract_and_round("int_real", 2.5, i, 1) / 1e6
            )
            specimenstable.loc[specfilter, "int_k_min"] = (
                extract_and_round("k", 2.5, i, 3) / 1e6
            )
            specimenstable.loc[specfilter, "int_k_max"] = (
                extract_and_round("k", 97.5, i, 3) / 1e6
            )
            specimenstable.loc[specfilter, "int_k"] = (
                extract_and_round("k", 50, i, 3) / 1e6
            )
            specimenstable.loc[specfilter, "meas_step_min"] = self[
                specimen
            ].savedLowerTemp
            specimenstable.loc[specfilter, "meas_step_max"] = self[
                specimen
            ].savedUpperTemp
            specimenstable.loc[specfilter, "software_packages"] = (
                "BiCEP_GUI-" + __version__
            )

            # Sort out methodcodes
            method_codes = self[specimen].methcodes.split(":")
            method_codes = list(set(method_codes))
            newstr = ""
            for code in method_codes[:-1]:
                newstr += code
                newstr += ":"
            newstr += method_codes[-1]
            specimenstable.loc[specfilter, "method_codes"] = self[specimen].methcodes

            extra_columns = self[specimen].extracolumnsdict
            for col in extra_columns.keys():
                specimenstable.loc[specfilter, col] = extra_columns[col]

        sitestable.loc[sitesfilter, "method_codes"] = self.methcodes
        specimenstable["meas_step_unit"] = "Kelvin"
        sitestable = sitestable.fillna("")
        specimenstable = specimenstable.fillna("")
        sitesdict = sitestable.to_dict("records")
        specimensdict = specimenstable.to_dict("records")
        pmag.magic_write(wd + "sites.txt", sitesdict, "sites")
        pmag.magic_write(wd + "specimens.txt", specimensdict, "specimens")

    def regplot(self, ax, legend=False, title=None):
        """
        Plots B vs k for all specimens in a site given a BiCEP or unpooled fit

        Inputs
        ------
        ax: matplotlib axis
        axes to plot to

        legend: bool
        If set to True, plots a legend

        title: str
        Title for plot. Does not plot title if set to None.
        """
        B_lab_list = []
        for specimen in self.specimens.values():
            B_lab_list.append(specimen.B_lab)
        try:
            Bs = extract_values(self.fit, "int_real").T
            ks = extract_values(self.fit, "k").T
            int_sites = extract_values(self.fit, "int_site")
            cs = extract_values(self.fit, "c")
            mink, maxk = np.amin(ks), np.amax(ks)
            minB, maxB = cs * mink + int_sites, cs * maxk + int_sites
            c = np.random.choice(range(len(minB)), 100)
            ax.plot([mink, maxk], [minB[c], maxB[c]], color="skyblue", alpha=0.12)
        except:
            Bs = extract_values(self.fit, "slope").T * np.array(B_lab_list).T
            ks = extract_values(self.fit, "k").T
        ax.set_xlabel(r"$\vec{k}$")
        ax.plot(
            np.percentile(ks, (2.5, 97.5), axis=0),
            [np.median(Bs, axis=0), np.median(Bs, axis=0)],
            "k",
        )
        ax.plot(
            [np.median(ks, axis=0), np.median(ks, axis=0)],
            np.percentile(Bs, (2.5, 97.5), axis=0),
            "k",
        )
        ax.plot(
            np.median(ks, axis=0),
            np.median(Bs, axis=0),
            "o",
            markerfacecolor="lightgreen",
            markeredgecolor="k",
        )
        ax.axvline(0, color="k", linewidth=1)
        if title != None:
            ax.set_title(title, fontsize=20, loc="left")

    def get_specimen_rhats(self):
        """
        Finds the worst Rhat va; for each specimen and assigns it to that specimen
        """
        try:
            rhats_orig = az.rhat(self.fit)
            rhats = rhats_orig.to_stacked_array("rhats", sample_dims=[]).values
            worst_rhat = rhats[(1 - rhats) ** 2 == max((1 - rhats) ** 2)][0]
            specrhatsarray = []
            specrhats = rhats_orig.drop_vars(["int_site", "sd_site", "c"]).data_vars
            for i in specrhats:
                specrhatsarray.append(specrhats[i].values)
            specrhatsarray = np.array(specrhatsarray)
            speclist = [
                spec
                for spec in self.specimens.keys()
                if self.specimens[spec].active == True
            ]
            for j in range(len(speclist)):
                spec = self[speclist[j]]
                rhats_spec = specrhatsarray[:, j]
                try:
                    worst_rhat_spec = rhats_spec[
                        (1 - rhats_spec) ** 2 == max((1 - rhats_spec) ** 2)
                    ][0]
                except IndexError:
                    worst_rhat_spec = np.nan
                spec.rhat = worst_rhat_spec
        except (TypeError, ValueError) as e:
            worst_rhat = 1.00
            print("Error caught:", e)
        return worst_rhat

    def histplot(self, ax, **kwargs):
        """
        Plots a histogram of the site level paleointensity estimate.

        Inputs
        ------
        **kwargs:
        arguments to be passed to the histogram plot

        Returns
        -------
        None
        """
        ax.hist(
            extract_values(self.fit, "int_site"),
            bins=100,
            color="skyblue",
            density=True,
        )
        minB, maxB = np.percentile(extract_values(self.fit, "int_site"), (2.5, 97.5))
        ax.plot([minB, maxB], [0, 0], "k", linewidth=4)
        ax.set_xlabel("Intensity ($\mu$T)")
        ax.set_ylabel("Probability Density")


class Specimen:
    """
    Specimen from a given site or sample SpecimenCollection object.

    Parameters
    ----------

    parentCollection: SpecimenCollection object
    Site/Sample the specimen is derived from.

    specimenName: string
    Name of specimen
    """

    def __init__(self, parentCollection, specimenName):

        # Inherent properties
        self.parentCollection = (
            parentCollection  # Site or sample this specimen belongs to
        )
        self.name = specimenName
        self.data = parentCollection.data[
            parentCollection.data.specimen == specimenName
        ]

        self.methcodes = "IE-BICEP"  # Appended to when saving.
        self.extracolumnsdict = {}  # Extra columns for e.g. corrections
        self.rhat = 1.0

        # Important constants for the specimen
        self.B_lab = self.data.B_lab.iloc[0] * 1e6
        self.NRM0 = self.data.NRM.iloc[0]
        self.pTRMmax = max(self.data.PTRM)
        self.temps = self.data.temp_step.unique()

        # Try importing from redo file. Otherwise initiliaze to default interpretation using all measurements
        redo = parentCollection.parentData.redo
        try:
            # Interpretation temperatures
            self.lowerTemp = float(redo.loc[redo[0] == specimenName, 1].iloc[0])
            self.upperTemp = float(redo.loc[redo[0] == specimenName, 2].iloc[0])
            if self.lowerTemp == self.upperTemp:
                self.active = False  # Used for BiCEP GUI- set to false if specimen excluded from analysis
            else:
                self.active = True

            # Saved interpretation temperatures- these are saved when the BiCEP_fit method is run.
            self.savedLowerTemp = float(redo.loc[redo[0] == specimenName, 1].iloc[0])
            self.savedUpperTemp = float(redo.loc[redo[0] == specimenName, 2].iloc[0])
        except:
            # Interpretation temperatures
            self.lowerTemp = min(self.temps)
            self.upperTemp = max(self.temps)

            # Saved interpretation temperatures- these are saved when the BiCEP_fit method is run.
            self.savedLowerTemp = min(self.temps)
            self.savedUpperTemp = max(self.temps)
            self.active = True

        # Definitions of Thellier Experiment Measurements (for plotting)
        self.IZZI = self.data[
            (self.data.steptype == "IZ") | (self.data.steptype == "ZI")
        ]
        self.P = self.data[self.data.steptype == "P"]
        self.T = self.data[self.data.steptype == "T"]
        self.IZZI_trunc = self.IZZI[
            (self.IZZI.temp_step >= self.lowerTemp)
            & (self.IZZI.temp_step <= self.upperTemp)
        ]
        self.IZ = self.IZZI_trunc[self.IZZI_trunc.steptype == "IZ"]
        self.ZI = self.IZZI_trunc[self.IZZI_trunc.steptype == "ZI"]
        self.saved_IZZI_trunc = self.IZZI[
            (self.IZZI.temp_step >= self.lowerTemp)
            & (self.IZZI.temp_step <= self.upperTemp)
        ]
        self.saved_IZ = self.saved_IZZI_trunc[self.IZZI_trunc.steptype == "IZ"]
        self.saved_ZI = self.saved_IZZI_trunc[self.IZZI_trunc.steptype == "ZI"]

        # Definitions of Zijderveld Measurements (for plotting)
        self.NRM_dirs = self.IZZI.loc[:, "NRM_x":"NRM_z"].values
        self.NRM_trunc_dirs = self.IZZI_trunc.loc[:, "NRM_x":"NRM_z"].values
        self.plotting_temps = self.calc_plotting_temps()

        # SPD parameters/PCA fit to direction
        self.drat = get_drat(
            self.IZZI, self.IZZI_trunc, self.P[(self.P.baseline_temp <= self.upperTemp)]
        )
        pca = PCA(n_components=3)
        try:
            self.pca = pca.fit(self.NRM_trunc_dirs)
        except:
            self.pca = pca.fit(self.NRM_dirs)
        self.mad = get_mad(self.IZZI_trunc, self.pca)
        self.dang = get_dang(self.NRM_trunc_dirs, self.pca)

    def __repr__(self):
        return (
            "Specimen "
            + self.name
            + " in "
            + self.parentCollection.key
            + " "
            + self.parentCollection.name
        )

    def change_temps(self, lowerTemp, upperTemp):
        """
        Changes temperature range (interpretation for specimen).
        Recalculates SPD statistic and PCA for said specimen.

        Inputs
        ------
        lowerTemp: float
        Lower temperature (inclusive) for interpretation

        upperTemp: float
        Upper temperature (inclusive) for interpretation

        Returns
        -------
        None
        """

        self.lowerTemp = lowerTemp
        self.upperTemp = upperTemp
        self.IZZI_trunc = self.IZZI[
            (self.IZZI.temp_step >= self.lowerTemp)
            & (self.IZZI.temp_step <= self.upperTemp)
        ]
        self.IZ = self.IZZI_trunc[self.IZZI_trunc.steptype == "IZ"]
        self.ZI = self.IZZI_trunc[self.IZZI_trunc.steptype == "ZI"]
        self.drat = get_drat(
            self.IZZI, self.IZZI_trunc, self.P[(self.P.baseline_temp <= self.upperTemp)]
        )
        pca = PCA(n_components=3)
        self.NRM_trunc_dirs = self.IZZI_trunc.loc[:, "NRM_x":"NRM_z"].values
        try:
            self.pca = pca.fit(self.NRM_trunc_dirs)
        except:
            self.pca = pca.fit(self.NRM_dirs)
        self.mad = get_mad(self.IZZI_trunc, self.pca)
        self.dang = get_dang(self.NRM_trunc_dirs, self.pca)

    def save_changes(self):
        """
        Commits temperature changes for use with the BiCEP method

        Inputs
        ------
        None

        Returns
        -------
        None
        """
        self.savedLowerTemp = self.lowerTemp
        self.savedUpperTemp = self.upperTemp
        self.saved_IZZI_trunc = self.IZZI_trunc
        self.saved_IZ = self.IZ
        self.saved_ZI = self.ZI
        if type(self.parentCollection.parentData.redo) == type(None):
            redo = pd.DataFrame({0: [], 1: [], 2: []})
        else:
            redo = self.parentCollection.parentData.redo
        if self.name in redo[0].unique():
            redo.loc[redo[0] == self.name, 1] = self.lowerTemp
            redo.loc[redo[0] == self.name, 2] = self.upperTemp
        else:
            redo = pd.concat(
                [
                    redo,
                    pd.DataFrame(
                        {0: [self.name], 1: [self.lowerTemp], 2: [self.upperTemp]}
                    ),
                ]
            )
        self.parentCollection.parentData.redo = redo
        wd = self.parentCollection.parentData.wd
        redo.to_csv(wd + "bicep_gui.redo", header=None, index=False, sep=" ")

    def plot_arai(self, ax=None, temps=True):
        """
        Plots data onto the Arai plot.

        Inputs
        ------
        ax: matplotlib axis
        axis for plot to be plotted on to

        temps: bool
        if True, plots temperatures on the Arai plot

        Returns
        -------
        None
        """
        if ax == None:
            fig, ax = plt.subplots()
        # IZZI_trunc=self.IZZI[(self.IZZI.temp_step>=self.lowerTemp)&(self.IZZI.temp_step<=self.upperTemp)]
        lines = ax.plot(
            self.IZZI.PTRM / self.NRM0, self.IZZI.NRM / self.NRM0, "k", linewidth=1
        )
        ptrm_base = self.IZZI[self.IZZI.temp_step.isin(self.P.baseline_temp)]
        for i in range(len(ptrm_base)):
            step_PTRMs = [
                ptrm_base.iloc[i].PTRM / self.NRM0,
                self.P.iloc[i].PTRM / self.NRM0,
                self.P.iloc[i].PTRM / self.NRM0,
            ]
            step_NRMs = [
                ptrm_base.iloc[i].NRM / self.NRM0,
                ptrm_base.iloc[i].NRM / self.NRM0,
                self.P.iloc[i].NRM / self.NRM0,
            ]
            ax.plot(step_PTRMs, step_NRMs, "k", lw=1, alpha=0.5)
        emptydots = ax.plot(
            self.IZZI.PTRM / self.NRM0,
            self.IZZI.NRM / self.NRM0,
            "o",
            markerfacecolor="None",
            markeredgecolor="black",
            label="Not Used",
        )
        ptrm_check = ax.plot(
            self.P.PTRM / self.NRM0,
            self.P.NRM / self.NRM0,
            "^",
            markerfacecolor="None",
            markeredgecolor="black",
            markersize=10,
            label="PTRM Check",
        )
        md_check = ax.plot(
            self.T.PTRM / self.NRM0,
            self.T.NRM / self.NRM0,
            "s",
            markerfacecolor="None",
            markeredgecolor="black",
            markersize=10,
        )

        ax.set_ylim(0, max(self.IZZI.NRM / self.NRM0) * 1.1)
        ax.set_xlim(0, self.pTRMmax / self.NRM0 * 1.1)
        if self.active == True:
            iz_plot = ax.plot(
                self.IZ.PTRM / self.NRM0,
                self.IZ.NRM / self.NRM0,
                "o",
                markerfacecolor="b",
                markeredgecolor="black",
                label="I step",
            )
            zi_plot = ax.plot(
                self.ZI.PTRM / self.NRM0,
                self.ZI.NRM / self.NRM0,
                "o",
                markerfacecolor="r",
                markeredgecolor="black",
                label="Z step",
            )
        ax.set_ylabel("NRM/NRM$_0$")
        ax.set_xlabel("pTRM/NRM$_0$")
        for i in self.plotting_temps:
            tempRow = self.IZZI.iloc[i]
            temp = tempRow.temp_step
            ax.text(
                tempRow.PTRM / self.NRM0,
                tempRow.NRM / self.NRM0,
                str(temp - 273),
                alpha=0.5,
            )

    def plot_zijd(self, ax=None, temps=True):
        """
        Plots data onto the Zijderveld plot. Does not fit a line to this data.

        Inputs
        ------
        ax: matplotlib axis
        axis for plot to be plotted on to

        temps: bool
        if True, plots temperature values as text on plot.

        Returns
        -------
        None
        """
        if ax == None:
            fig, ax = plt.subplots()
        # Get the NRM data for the specimen
        # Plot axis
        ax.axvline(0, color="k", linewidth=1)
        ax.axhline(0, color="k", linewidth=1)

        # Plot NRM directions
        ax.plot(self.NRM_dirs[:, 0], self.NRM_dirs[:, 1], "k")
        ax.plot(self.NRM_dirs[:, 0], self.NRM_dirs[:, 2], "k")

        # Plot NRM directions in currently selected temperature range as closed symbols
        ax.plot(self.NRM_trunc_dirs[:, 0], self.NRM_trunc_dirs[:, 1], "ko")
        ax.plot(self.NRM_trunc_dirs[:, 0], self.NRM_trunc_dirs[:, 2], "rs")

        # Plot open circles for all NRM directions as closed symbols
        ax.plot(
            self.NRM_dirs[:, 0],
            self.NRM_dirs[:, 1],
            "o",
            markerfacecolor="None",
            markeredgecolor="k",
        )
        ax.plot(
            self.NRM_dirs[:, 0],
            self.NRM_dirs[:, 2],
            "s",
            markerfacecolor="None",
            markeredgecolor="k",
        )
        length, vector = self.pca.explained_variance_[0], self.pca.components_[0]
        vals = self.pca.transform(self.NRM_trunc_dirs)[:, 0]
        v = np.outer(vals, vector)

        # Plot PCA line fit
        ax.plot(self.pca.mean_[0] + v[:, 0], self.pca.mean_[1] + v[:, 1], "g")
        ax.plot(self.pca.mean_[0] + v[:, 0], self.pca.mean_[2] + v[:, 2], "g")

        ax.set_xlabel("x, $Am^2$")
        ax.set_ylabel("y,z, $Am^2$")
        ax.axis("equal")
        mins = np.amin(self.NRM_dirs, axis=0)
        maxes = np.amax(self.NRM_dirs, axis=0)
        x_margins = (maxes[0] - mins[0]) / 20
        y_margins = (max(maxes[1:]) - min(mins[1:])) / 20

        ax.set_xlim(mins[0] - x_margins, maxes[0] + x_margins)
        ax.set_ylim(max(maxes[1:]) + y_margins, min(mins[1:]) - y_margins)
        ax.ticklabel_format(scilimits=(0, 0))
        # Plot Temperature text on Zijderveld plot (uses clustering to do this).
        if temps == True:
            zijd_data = self.NRM_dirs

            zrange = np.max(zijd_data[:, 2]) - np.min(zijd_data[:, 2])
            yrange = np.max(zijd_data[:, 1]) - np.min(zijd_data[:, 1])
            if yrange > zrange:
                textindex = 1
            else:
                textindex = 2
            # Gradients are reasonably accurate, curvatures aren't
            grads = np.gradient(zijd_data, axis=0)
            gradgrads = np.gradient(grads, axis=0)
            # If we have 0 in other directions, make gradient large
            gradgrads[gradgrads == -np.inf] = -1e38
            gradgrads[gradgrads == np.inf] = 1e38
            max_norm = np.sqrt(
                np.diff(ax.get_xlim()) ** 2 + np.diff(ax.get_ylim()) ** 2
            )
            for j in self.plotting_temps:
                # Calculate normal to gradient
                diff = 1 / grads[j]
                diff[0] = -diff[0]
                # Gradient should be in the direction away from curvature
                diff = (
                    -np.sign(np.dot(gradgrads[j, [0, textindex]], diff[[0, textindex]]))
                    * diff
                )
                diff = diff / np.linalg.norm(diff[[0, textindex]])

                tic_loc = zijd_data[j] + max_norm * diff * 0.04

                # Calculate text alignment
                v_angle = np.degrees(np.arctan2(diff[textindex], np.abs(diff[0])))
                h_angle = np.degrees(np.arctan2(np.abs(diff[textindex]), diff[0]))
                if v_angle > 45:
                    va = "top"
                elif v_angle < -45:
                    va = "bottom"
                else:
                    va = "center"
                if h_angle < 45:
                    ha = "left"
                elif h_angle > 135:
                    ha = "right"
                else:
                    ha = "center"

                ax.text(
                    tic_loc[0],
                    tic_loc[textindex],
                    str(int(self.temps[j] - 273.0)),
                    alpha=0.5,
                    ha=ha,
                    va=va,
                )
                ax.plot(
                    [zijd_data[j, 0], tic_loc[0]],
                    [zijd_data[j, textindex], tic_loc[textindex]],
                    "k",
                    lw=1,
                    zorder=-1,
                )

    def calc_plotting_temps(self):
        """
        Uses k-means clustering to find the set of temperatures
        to plot as text on the Arai/Zijderveld plot, without things
        overlapping
        """
        zijd_data = self.NRM_dirs

        zrange = np.max(zijd_data[:, 2]) - np.min(zijd_data[:, 2])
        yrange = np.max(zijd_data[:, 1]) - np.min(zijd_data[:, 1])
        if yrange > zrange:
            textindex = 1
        else:
            textindex = 2

        n_clusters = min(int(len(zijd_data) / 2), 8)
        clusters = KMeans(n_clusters, random_state=0).fit_predict(
            zijd_data[:, [0, textindex]]
        )

        indices = []
        for i in range(n_clusters):
            points = np.where(clusters == i)[0]
            indices.append(points[int(len(points) / 2)])
        return indices

    def BiCEP_prep(self):
        """
        Returns the needed data for a paleointensity interpretation to
        perform the BiCEP method, calculates all corrections for a specimen.
        Performs scaling on the PTRM and NRM data. It performs the Taubin SVD circle fit
        to find the maximum likelihood circle fit to initialize the BiCEP method sampler.

        Inputs
        ------
        None

        Returns
        -------
        minPTRM: float
        Minimum scaled pTRM

        maxNRM: float
        Minimum scaled NRM

        PTRMmax: float
        Maximum total pTRM (scaled)

        k: float
        Best fitting k value using Taubin circle fit.

        phi: float
        Best fitting phi value using Taubin circle fit

        dist_to_edge: float
        Best fitting D value using Taubin circle fit.

        sigma: float
        Best fitting sigma value using Taubin circle fit.

        PTRMS: numpy.ndarray()
        Scaled and translated pTRM values

        NRMS: numpy.ndarray()
        Scaled and translated NRM values.
        """

        # Calculate Anisotropy Correction:
        if len(self.IZZI.dropna(subset=["s_tensor"])) > 0:
            c = calculate_anisotropy_correction(self.saved_IZZI_trunc)
            self.extracolumnsdict["int_corr_aniso"] = c
            # Get method code depending on anisotropy type (AARM or ATRM)
            self.methcodes += self.IZZI["aniso_type"].iloc[0]
        else:
            c = 1

        # Get Cooling Rate Correction
        if self.IZZI.correction.iloc[0] != 1:
            self.methcodes += ":DA-CR-TRM"  # method code for cooling rate correction
            self.extracolumnsdict["int_corr_cooling_rate"] = self.IZZI.correction.iloc[
                0
            ]

        # Calculate nonlinear TRM Correction
        if len(self.IZZI.dropna(subset=["NLT_beta"])) > 0:
            self.methcodes += ":DA-NL"  # method code for nonlinear TRM correction
            total_correction = calculate_NLT_correction(
                self.saved_IZZI_trunc, c
            )  # total correction (combination of all three corrections)
            self.extracolumnsdict["int_corr_nlt"] = total_correction / (
                c * self.IZZI.correction.iloc[0]
            )  # NLT correction is total correction/original correction.
        else:
            total_correction = c * self.IZZI.correction.iloc[0]

        # Converting Arai plot data to useable form
        NRMS = self.saved_IZZI_trunc.NRM.values / self.NRM0
        PTRMS = (
            self.saved_IZZI_trunc.PTRM.values / self.NRM0 / total_correction
        )  # We divide our pTRMs by the total correction, because we scale the pTRM values so that the maximum pTRM is one, this doesn't affect the fit and just gets scaled back when converting the circle tangent slopes back to intensities as would be expected, but it's easier to apply this here.

        PTRMmax = max(
            self.IZZI.PTRM / self.NRM0 / total_correction
        )  # We scale by our maximum pTRM to perform the circle fit.
        line = bestfit_line(
            self.IZZI.PTRM / self.NRM0 / total_correction, self.IZZI.NRM / self.NRM0
        )  # best fitting line to the pTRMs

        PTRMS = PTRMS / PTRMmax  # Scales the pTRMs so the maximum pTRM is one

        # We subtract the minimum pTRM and NRM to maintain aspect ratio and make circle fitting easier.
        minPTRM = min(PTRMS)
        minNRM = min(NRMS)
        PTRMS = PTRMS - minPTRM
        NRMS = NRMS - minNRM

        # We perform the Taubin least squares circle fit to get values close to the Bayesian maximum likelihood to initialize our MCMC sampler at, this makes sampling a lot easier than initializing at a random point (which may have infinitely low probability).

        try:
            x_c, y_c, R, sigma = TaubinSVD(PTRMS, NRMS)  # Calculate x_c,y_c and R
        except LinAlgError:
            raise LinAlgError(
                "Could not get initial guess for circle fit to specimen "
                + self.name
                + "It has "
                + len(PTRMS)
                + "temperature steps included in the interpretation. Did you mean to exclude this specimen?"
            )

        dist_to_edge = abs(np.sqrt(x_c**2 + y_c**2) - R)  # Calculate D (dist_to_edge)
        phi = np.radians(np.degrees(np.arctan(y_c / x_c)) % 180)

        # Calculate (and ensure the sign of) k
        if y_c < 0:
            k = -1 / R
        else:
            k = 1 / R

        return (minPTRM, minNRM, PTRMmax, k, phi, dist_to_edge, sigma, PTRMS, NRMS)

    def plot_circ(self, ax, legend=False, linewidth=2, title=None, tangent=False):
        """
        Plots circle fits sampled from the posterior distribution
        (using the BiCEP method) to the Arai plot data. Plots tangent
        to the circle as a slope if tangent=True

        Inputs
        ------
        ax: matplotlib axis
        axis to be used for plot.

        legend: bool
        If True plots a legend.

        linewidth: float
        Width of circle fit lines on plot

        title: str
        Title for plot

        tangent: bool
        If set to True, plots best fitting tangent to circle.
        """

        # Get information on maximum pTRM for rescaling of circle
        fit = self.parentCollection.fit
        speclist = np.array(
            [
                specimen
                for specimen in self.parentCollection.specimens.keys()
                if self.parentCollection.specimens[specimen].active == True
            ]
        )
        try:
            i = np.where(speclist == self.name)[0][0]
        except:
            return
        if fit != None:
            minNRM = min(self.saved_IZZI_trunc.NRM / self.NRM0)
            minPTRM = min(self.saved_IZZI_trunc.PTRM / self.NRM0)

            # Parameters for the circle fit
            Rs = extract_values(fit, "R")[i]
            x_cs = extract_values(fit, "x_c")[i]
            y_cs = extract_values(fit, "y_c")[i]
            c = np.random.choice(range(len(Rs)), 100)
            thetas = np.linspace(0, 2 * np.pi, 1000)
            NRM0 = self.NRM0

            # Circle x and y values for circle plot.
            xs = (
                x_cs[c][:, np.newaxis] * self.pTRMmax / self.NRM0
                + minPTRM
                + Rs[c][:, np.newaxis] * np.cos(thetas) * self.pTRMmax / self.NRM0
            )
            ys = y_cs[c][:, np.newaxis] + minNRM + Rs[c][:, np.newaxis] * np.sin(thetas)

            # Plot Circles
            ax.plot(
                xs.T,
                ys.T,
                "-",
                color="lightgreen",
                alpha=0.2,
                linewidth=linewidth,
                zorder=-1,
            )
            ax.plot(100, 100, "-", color="lightgreen", label="Circle Fits")

            # Find tangents to the circle:
            if tangent == True:
                phis = extract_values(fit, "phi")[i]
                dists = extract_values(fit, "dist_to_edge")[i]
                slope_ideal = -1 / np.tan(np.median(phis)) / self.pTRMmax * self.NRM0
                x_i = (
                    np.median(dists)
                    * np.cos(np.median(phis))
                    * self.pTRMmax
                    / self.NRM0
                    + minPTRM
                )
                y_i = np.median(dists) * np.sin(np.median(phis)) + minNRM

                ax.plot(x_i, y_i, "ko")
                c = y_i - slope_ideal * x_i
                d = -c / slope_ideal
                ax.plot([0, d], [c, 0], "k", linestyle="--")

            # Add legend and title to plot
            if legend == True:
                ax.legend(fontsize=10)
            if title != None:
                ax.set_title(title, fontsize=20, loc="left")
