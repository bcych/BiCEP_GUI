import nest_asyncio

nest_asyncio.apply()
from ipyfilechooser import FileChooser
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# import asyncio
from BiCEP.datamodel import ThellierData, __version__
from BiCEP.extract import extract_values
from BiCEP.criteria import auto_interpret
from scipy import stats
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import arviz as az
from BiCEP import model_circle_fast, model_circle_slow, __version__


class GUI:
    def __init__(self):
        plt.ioff()

        row_layout_buttons = widgets.Layout(
            width="60%",
            display="flex",
            flex_flow="row",
            justify_content="space-around",
            margin="1000px left",
        )

        row_layout = widgets.Layout(
            width="100%",
            display="flex",
            flex_flow="row",
            justify_content="space-around",
        )
        self.site_wid = widgets.Dropdown(description="Site:")

        self.specimen_wid = widgets.Dropdown(options=[], description="Specimen:")

        self.lower_temp_wid = widgets.Dropdown(
            options=[],
            description="Temperatures (Low):",
            style={"description_width": "initial"},
        )

        self.upper_temp_wid = widgets.Dropdown(
            options=[], description="(High):", style={"description_width": "initial"}
        )

        self.save_wid = widgets.Button(description="Save Temperatures", disabled=True)

        self.newfile_wid = FileChooser(
            ".",
            description="Choose File",
            style={"description_width": "initial"},
            filter_pattern="*.csv",
        )

        self.run_wid = widgets.Button(
            description="Start", style={"description_width": "initial"}, disabled=True
        )

        self.newfile_wid.register_callback(self.enablerun)

        self.figsave = widgets.Button(description="Save Figures", disabled=True)

        self.figchoice = widgets.Dropdown(
            options=["Specimen Plot", "Site Plot"], layout=widgets.Layout(width="20%")
        )

        self.figformats = widgets.Dropdown(
            description="Format:", options=["pdf", "png", "jpg", "svg", "tiff"]
        )

        self.figsave.on_click(self.save_figures)
        self.run_wid.on_click(self.run)

        self.madbox = widgets.Button(description="MAD:", disabled=True)
        self.dangbox = widgets.Button(description="DANG:", disabled=True)
        self.dratbox = widgets.Button(description="DRAT:", disabled=True)
        self.rhatbox = widgets.Button(description="R_hat:", disabled=True)
        self.filebox = widgets.HBox(
            [self.newfile_wid, self.run_wid], grid_area="filebox"
        )
        self.tempbox = widgets.VBox(
            [self.lower_temp_wid, self.upper_temp_wid], grid_area="tempbox"
        )
        self.specbox = widgets.VBox(
            [self.site_wid, self.specimen_wid], grid_area="specbox"
        )
        self.checkbox = widgets.Checkbox(
            description="Exclude Specimen",
            disabled=True,
            indent=False,
            layout=widgets.Layout(width="15%"),
        )
        self.savebox = widgets.HBox(
            [
                self.save_wid,
                self.checkbox,
                self.figsave,
                self.figchoice,
                self.figformats,
            ],
            grid_area="savebox",
            layout=row_layout,
        )
        self.dirbox = widgets.HBox([self.madbox, self.dangbox])
        self.dratrhatbox = widgets.HBox([self.dratbox, self.rhatbox])
        self.critbox = widgets.VBox([self.dirbox, self.dratrhatbox])
        self.output = widgets.Output(layout={"border": "1px solid black"})

        self.specplots = widgets.Output(grid_area="specplots")
        self.dropdowns = widgets.HBox(
            [self.specbox, self.tempbox, self.critbox], grid_area="dropdowns"
        )
        self.fig, self.ax = plt.subplots(1, 2, figsize=(9, 3))
        self.fig.canvas.header_visible = False
        plt.tight_layout()

        # fullbox gives the entire specimen processing box
        self.fullbox1 = widgets.Box(
            children=[self.filebox, self.dropdowns, self.fig.canvas, self.savebox],
            title="Specimen Processing",
            layout=widgets.Layout(
                width="100%",
                flex_flow="column",
                align_content="space-around",
                align_items="flex-start",
            ),
        )

        self.madcrit = widgets.IntSlider(
            min=1, max=90, value=5, step=1, description="MAD: "
        )
        self.dangcrit = widgets.IntSlider(
            min=1, max=90, value=10, step=1, description="DANG: "
        )
        self.dratcrit = widgets.IntSlider(
            min=1, max=50, value=10, step=1, description="DRAT: "
        )
        self.madtype = widgets.Dropdown(
            options=["mad_free", "mad_coe"], description="MAD type"
        )
        self.autoint = widgets.Button(description="Auto Interpret", disabled=True)
        self.intsettings = widgets.VBox([self.madtype, self.autoint])
        self.critsliders = widgets.VBox([self.madcrit, self.dangcrit, self.dratcrit])
        self.critpage = widgets.HBox([self.critsliders, self.intsettings])
        self.fullbox3 = widgets.Box(
            children=[self.critpage],
            title="Auto-Interpreter",
            layout=widgets.Layout(
                width="100%",
                flex_flow="column",
                align_content="space-around",
                align_items="flex-start",
            ),
        )

        self.fullbox = widgets.Tab(
            children=[self.fullbox1, self.fullbox3],
            titles=["Specimen Processing", "Auto Interpreter"],
        )
        self.fullbox.set_title(0, "Specimen Processing")
        self.fullbox.set_title(1, "Auto Interpreter")
        # Make a plot in the specplots box

        self.dangbox.button_style = "info"
        self.madbox.button_style = "info"
        self.dratbox.button_style = "info"
        self.rhatbox.button_style = "info"
        # GUI widgets for the site processing box
        self.n_samples_wid = widgets.IntSlider(
            min=3000, max=100000, value=30000, step=1000, description="n samples"
        )
        self.method_wid = widgets.Dropdown(
            options=["Fast", "Slow"],
            description="Sampler:",
        )
        self.process_wid = widgets.Button(
            description="Process Site Data", disabled=True
        )
        self.process_wid.on_click(self.get_site_dist)
        self.rhatlabel = widgets.Button(
            description="R_hat:", disabled=True, layout=widgets.Layout(width="60%")
        )
        self.nefflabel = widgets.Button(description="n_eff:", disabled=True)
        self.banclabel = widgets.Button(
            description="B_anc:", disabled=True, layout=widgets.Layout(width="60%")
        )

        self.gradelabel = widgets.Button(description="Category:", disabled=True)
        self.sampler_diag = widgets.HBox([self.rhatlabel, self.nefflabel])
        self.sampler_results = widgets.HBox([self.banclabel, self.gradelabel])
        self.sampler_buttons = widgets.VBox([self.sampler_diag, self.sampler_results])
        self.sampler_pars = widgets.VBox([self.n_samples_wid, self.method_wid])
        self.sampler_line = widgets.HBox([self.sampler_pars, self.sampler_buttons])
        self.banclabel.button_style = "info"
        self.nefflabel.button_style = "info"
        self.rhatlabel.button_style = "info"
        self.gradelabel.button_style = "info"
        # siteplots=widgets.Output()

        self.fig_2, self.ax_2 = plt.subplots(1, 2, figsize=(6.4, 3), sharey=True)
        self.fig_2.canvas.header_visible = False
        plt.tight_layout()

        self.savetables = widgets.Button(
            description="Save to MagIC tables", disabled=True
        )
        self.savetables.on_click(self.save_magic_tables)
        self.savenetcdf = widgets.Button(description="Save to netCDF", disabled=True)
        self.sitesave = widgets.HBox([self.savetables, self.savenetcdf])
        self.savenetcdf.on_click(self.save_to_netcdf)

        self.fullbox2 = widgets.VBox(
            [self.process_wid, self.sampler_line, self.fig_2.canvas, self.sitesave],
            title="Site Processing",
        )
        self.fullbox4 = widgets.Tab(children=[self.fullbox2, self.output])
        self.fullbox4.set_title(0, "Site Processing")
        self.fullbox4.set_title(1, "Terminal Output")
        self.specpage = widgets.Accordion([self.fullbox])
        self.sitepage = widgets.Accordion([self.fullbox4])
        self.specpage.set_title(0, "Specimen Processing")
        self.sitepage.set_title(0, "Site Processing")

        self.gui = widgets.VBox([self.specpage, self.sitepage])

        display(self.gui)
        plt.ion()

    def run(self, a):
        """
        Runs the GUI after selecting a file.

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        global thellierData
        with self.output:
            self.run_wid.description = "Converting Data..."
            thellierData = ThellierData(
                self.newfile_wid.selected_path
                + "/"
                + self.newfile_wid.selected_filename
            )
            self.run_wid.description = "Preparing GUI..."
            self.site_wid.options = np.sort(list(thellierData.collections.keys()))
            self.site_wid.value = np.sort(list(thellierData.collections.keys()))[0]
            self.specimen_wid.options = np.sort(
                list(thellierData[self.site_wid.value].specimens.keys())
            )
            self.specimen_wid.value = np.sort(
                list(thellierData[self.site_wid.value].specimens.keys())
            )[0]

            self.lower_temp_wid.options = (
                thellierData[self.site_wid.value][self.specimen_wid.value].temps - 273
            )
            self.upper_temp_wid.options = (
                thellierData[self.site_wid.value][self.specimen_wid.value].temps - 273
            )
            self.lower_temp_wid.value = (
                thellierData[self.site_wid.value][self.specimen_wid.value].lowerTemp
                - 273
            )
            self.upper_temp_wid.value = (
                thellierData[self.site_wid.value][self.specimen_wid.value].upperTemp
                - 273
            )

            self.site_wid.observe(self.on_change)
            self.specimen_wid.observe(self.on_change)
            self.lower_temp_wid.observe(self.on_change)
            self.upper_temp_wid.observe(self.on_change)
            self.save_wid.on_click(self.save_temps)
            self.checkbox.observe(self.activate_deactivate)
            self.autoint.on_click(self.full_auto_interpreter)

            self.display_specimen_plots()
            self.process_wid.disabled = False
            self.save_wid.disabled = False
            self.savetables.disabled = False
            self.figsave.disabled = False
            self.newfile_wid.disabled = True
            self.checkbox.disabled = False
            self.run_wid.description = "Running"
            self.run_wid.disabled = True
            self.savenetcdf.disabled = False
            self.autoint.disabled = False

    def display_specimen_ring(self):
        """
        Displays a red circle around the currently selected
        specimen in the site plot of BiCEP GUI

        Inputs:
        -------
        None

        Returns
        -------
        None
        """
        with self.output:
            try:
                fit = thellierData[self.site_wid.value].fit
                if thellierData[self.site_wid.value].artist != None:
                    thellierData[self.site_wid.value].artist[0].set_marker("None")
                currspec = self.specimen_wid.value
                speclist = np.array(
                    [
                        specimen
                        for specimen in thellierData[self.site_wid.value].specimens
                        if thellierData[self.site_wid.value][specimen].active == True
                    ]
                )
                specindex = np.where(speclist == currspec)
                try:
                    specindex = specindex[0][0]
                    ks = extract_values(fit, "k")[specindex]
                    int_reals = extract_values(fit, "int_real")[specindex]
                    thellierData[self.site_wid.value].artist = self.ax_2[0].plot(
                        np.median(ks),
                        np.median(int_reals),
                        "o",
                        markeredgecolor="r",
                        markerfacecolor="None",
                    )
                except IndexError:
                    pass
            except AttributeError:
                pass

    def display_specimen_plots(self):
        """
        Displays specimen level plots on the BiCEP GUI

        Inputs:
        -------
        None

        Returns:
        --------
        None
        """
        with self.output:
            self.ax[0].cla()
            self.ax[1].cla()
            thellierData[self.site_wid.value][self.specimen_wid.value].change_temps(
                self.lower_temp_wid.value + 273, self.upper_temp_wid.value + 273
            )
            thellierData[self.site_wid.value][self.specimen_wid.value].plot_circ(
                self.ax[0]
            )
            thellierData[self.site_wid.value][self.specimen_wid.value].plot_arai(
                self.ax[0]
            )
            thellierData[self.site_wid.value][self.specimen_wid.value].plot_zijd(
                self.ax[1]
            )
            self.madbox.description = (
                "MAD: %1.2f"
                % thellierData[self.site_wid.value][self.specimen_wid.value].mad
            )
            self.dangbox.description = (
                "DANG: %1.2f"
                % thellierData[self.site_wid.value][self.specimen_wid.value].dang
            )
            self.dratbox.description = (
                "DRAT: %1.2f"
                % thellierData[self.site_wid.value][self.specimen_wid.value].drat
            )
            rhat = thellierData[self.site_wid.value][self.specimen_wid.value].rhat
            self.rhatbox.description = (
                "R_hat: %1.2f"
                % thellierData[self.site_wid.value][self.specimen_wid.value].rhat
            )
            if (rhat == None) | (0.9 < rhat < 1.1):
                self.rhatbox.button_style = "info"
            else:
                self.rhatbox.button_style = "danger"
            self.fig.tight_layout()
            self.ax[1].relim()

    def on_change(self, change):
        """
        Update GUI on changing one of our site, specimen, temperature dropdowns.

        Inputs:
        -------
        change: Dropdown change object
        Gives us information about which object was changed (owner),
        the type of change (name, either value for a value change,
        or options for all options changed),and the new value (new).
        Note that these attributes are very important to avoid repeating
        many operations, as the changing the site widget's value changes
        the specimen widgets options, which then changes it's options.
        This is the reason for the numerous if statements in this function.

        Returns:
        --------
        None
        """
        # If we're changing the site dropdown, we need to replot the site plots and change the specimen options
        with self.output:
            if (change.owner == self.site_wid) & (change.name == "value"):
                self.specimen_wid.options = np.sort(
                    list(thellierData[self.site_wid.value].specimens.keys())
                )
                fit = thellierData[self.site_wid.value].fit
                try:
                    self.display_sampler_diags(fit)
                except:
                    pass
                self.display_site_plot(fit)

            # If we're changing the specimen dropdown, we need to update the temperature steps.
            if (change.owner == self.specimen_wid) & (change.name == "value"):
                self.lower_temp_wid.options = (
                    thellierData[self.site_wid.value][change.new].temps - 273
                )
                self.upper_temp_wid.options = (
                    thellierData[self.site_wid.value][change.new].temps - 273
                )
                self.checkbox.value = not thellierData[self.site_wid.value][
                    change.new
                ].active
                # We need to change the plot to account for saved temperature steps if there are any.
                if (
                    self.lower_temp_wid.value
                    != thellierData[self.site_wid.value][change.new].savedLowerTemp
                    - 273
                ) | (
                    self.upper_temp_wid.value
                    != thellierData[self.site_wid.value][change.new].savedUpperTemp
                    - 273
                ):
                    # This is fiddly, but it prevents event loop from moving on after changing value
                    self.lower_temp_wid.unobserve(self.on_change)
                    self.upper_temp_wid.unobserve(self.on_change)
                    self.lower_temp_wid.value = (
                        thellierData[self.site_wid.value][change.new].savedLowerTemp
                        - 273
                    )
                    self.upper_temp_wid.value = (
                        thellierData[self.site_wid.value][change.new].savedUpperTemp
                        - 273
                    )
                    self.upper_temp_wid.observe(self.on_change)
                    self.lower_temp_wid.observe(self.on_change)
                    self.display_specimen_plots()
                # Additionally, we need to make sure the plot changes if the temperature steps were the exact same as last time.
                else:
                    self.display_specimen_plots()
                # Finally, we need to display a ring around the specimen for the site level plot
                try:
                    self.display_specimen_ring()
                except:
                    pass

            # If we're changing the specimen plot, we display a red circle around the currently selected specimen on the site plot
            # if (change.owner==specimen_wid):
            # display_specimen_ring()

            if (change.name == "value") & (
                (change.owner == self.lower_temp_wid)
                | (change.owner == self.upper_temp_wid)
            ):
                self.display_specimen_plots()

    def save_temps(self, a):
        """
        Saves changes to specimen temperatures

        Inputs:
        ------
        a: Button pressed object
        has no practical use.

        Returns:
        -------
        None
        """
        with self.output:
            thellierData[self.site_wid.value][self.specimen_wid.value].save_changes()

    def get_sampler_diags(self, site):
        """
        Returns useful sampler diagnostics for a particular MCMC fit with pystan

        Inputs
        ------
        fit: StanFit object
        model fit to site/sample

        Returns
        -------
        rhat_worst: float
        worst rhat of all parameters

        n_eff_int_site: float
        Effective number of pseudosamples of B_anc
        """
        with self.output:
            try:
                rhat_worst = thellierData[site].get_specimen_rhats()
                n_eff_int_site = int(
                    az.ess(thellierData[site].fit.posterior)["int_site"].values * 1
                )
                return rhat_worst, n_eff_int_site
            except AttributeError:
                return None

    def display_sampler_diags(self, fit):
        """
        Displays the worst R_hat and n_eff, B_anc
        and Category or Grade for the BiCEP fit

        Inputs
        ------
        fit: StanFit object
        model fit to site/sample

        Returns:
        --------
        None
        """
        with self.output:
            try:
                rhat_worst, n_eff_int_site = self.get_sampler_diags(self.site_wid.value)
                if (rhat_worst > 1.1) | (rhat_worst < 0.9):
                    self.rhatlabel.button_style = "danger"
                else:
                    self.rhatlabel.button_style = "success"
                if n_eff_int_site < 1000:
                    self.nefflabel.button_style = "warning"
                else:
                    self.nefflabel.button_style = "success"

                self.rhatlabel.description = "R_hat: %1.2f" % rhat_worst
                self.nefflabel.description = "n_eff:" + str(n_eff_int_site)
                int_sites = extract_values(fit, "int_site")
                cs = extract_values(fit, "c")
                minB, medB, maxB = np.percentile(int_sites, (2.5, 50, 97.5), axis=0)
                self.banclabel.description = (
                    "B_anc: %3.1f" % medB + " (%3.1f" % minB + "- %3.1f" % maxB + ") μT"
                )
                cdiff = np.diff(np.percentile(cs, (2.5, 97.5), axis=0)) / np.percentile(
                    int_sites, 50
                )
                Bdiff = np.diff([minB, maxB]) / np.percentile(int_sites, 50)
                B_absdiff = np.diff([minB, maxB])
                skew = np.abs(stats.skew(int_sites))

                if (cdiff >= 1) & (Bdiff >= 0.4) & (B_absdiff >= 16):
                    self.gradelabel.description = "Category: D"
                    if (extract_values(fit, "k").shape[1]) < 5:
                        self.gradelabel.button_style = "warning"
                    else:
                        self.gradelabel.button_style = "danger"
                elif (cdiff < 1) & (Bdiff >= 0.4) & (B_absdiff >= 16):
                    self.gradelabel.description = "Category: C"
                    self.gradelabel.button_style = "warning"
                elif (cdiff >= 1) & ((Bdiff < 0.4) | (B_absdiff < 16)):
                    if skew <= 1:
                        self.gradelabel.description = "Category: B"
                        self.gradelabel.button_style = "success"
                    else:
                        self.gradelabel.description = "Category: B-"
                        self.gradelabel.button_style = "warning"
                elif (cdiff < 1) & ((Bdiff < 0.4) | (B_absdiff < 16)):
                    if skew <= 1:
                        self.gradelabel.description = "Category: A"
                        self.gradelabel.button_style = "success"
                    else:
                        self.gradelabel.description = "Category: A-"
                        self.gradelabel.button_style = "warning"
            except TypeError:
                pass

    def get_site_dist(self, a):
        """
        Runs the MCMC sampler and updates the GUI

        Inputs:
        ------
        a: Button pressed object
        has no practical use.

        Returns:
        -------
        None
        """
        with self.output:
            self.process_wid.description = "Processing.."

            if self.method_wid.value == "Slow":
                model = model_circle_slow
            elif self.method_wid.value == "Fast":
                model = model_circle_fast

            thellierData[self.site_wid.value].BiCEP_fit(
                model=model, n_samples=self.n_samples_wid.value
            )
            fit = thellierData[self.site_wid.value].fit
            self.display_sampler_diags(fit)

            # display_specimen_ring()
            self.display_site_plot(fit)

            self.process_wid.description = "Process Site Data"
            self.ax[0].cla()
            thellierData[self.site_wid.value][self.specimen_wid.value].plot_circ(
                self.ax[0]
            )
            thellierData[self.site_wid.value][self.specimen_wid.value].plot_arai(
                self.ax[0]
            )
            thellierData[self.site_wid.value].get_specimen_rhats()

    def full_auto_interpreter(self, a):
        self.autoint.description = "Interpreting..."
        auto_interpret(
            thellierData[self.site_wid.value],
            self.madcrit.value,
            self.dangcrit.value,
            self.dratcrit.value,
            self.madtype.value,
        )
        self.autoint.description = "Auto Interpret"

    def display_site_plot(self, fit):
        """
        Displays the site plots for BiCEP GUI

        Inputs
        ------
        fit: StanFit object
        BiCEP fit for that site/sample

        Returns
        -------
        None
        """
        with self.output:
            self.ax_2[0].cla()
            self.ax_2[1].cla()
            try:
                thellierData[self.site_wid.value].regplot(self.ax_2[0])
                int_sites = extract_values(fit, "int_site")
                int_reals = extract_values(fit, "int_real")
                ks = extract_values(fit, "k")
                self.ax_2[0].axhline(np.median(int_sites), color="k")
                self.ax_2[1].axhline(np.median(int_sites), color="k")
                self.ax_2[1].hist(
                    int_sites,
                    color="skyblue",
                    bins=100,
                    density=True,
                    orientation="horizontal",
                )
                self.ax_2[0].set_ylim(
                    min(np.percentile(int_reals, 2.5, axis=0)) * 0.9,
                    max(np.percentile(int_reals, 97.5, axis=0)) * 1.1,
                )
                self.ax_2[0].set_xlim(
                    min(
                        min(np.percentile(ks, 2.5, axis=0)) * 1.1,
                        min(np.percentile(ks, 2.5, axis=0)) * 0.9,
                    ),
                    max(
                        max(np.percentile(ks, 97.5, axis=0)) * 1.1,
                        max(np.percentile(ks, 97.5, axis=0)) * 0.9,
                    ),
                )
                self.ax_2[1].set_ylabel("$B_{anc}$")
                self.ax_2[1].set_xlabel("Probability Density")
                try:
                    self.display_specimen_ring()
                except:
                    pass
            except:
                self.rhatlabel.description = "R_hat:"
                self.nefflabel.description = "n_eff:"
                self.banclabel.description = "B_anc:"
                self.gradelabel.description = "Category: "
                self.banclabel.button_style = "info"
                self.nefflabel.button_style = "info"
                self.rhatlabel.button_style = "info"
                self.gradelabel.button_style = "info"
            self.fig_2.tight_layout()

    def save_magic_tables(self, a):
        """
        Saves data from the currently displayed site to the GUI

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        with self.output:
            try:
                thellierData[self.site_wid.value].save_magic_tables()
            except:
                print("Error! Something went wrong saving to MagIC tables.")
                print("Have you calculated an intensity estimate for this site?")

    def save_figures(self, a):
        """
        Saves figures from GUI to file

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        with self.output:
            objdict = {"Specimen Plot": self.fig, "Site Plot": self.fig_2}
            value = {
                "Specimen Plot": self.specimen_wid.value,
                "Site Plot": self.site_wid.value,
            }
            objdict[self.figchoice.value].savefig(
                thellierData.wd
                + value[self.figchoice.value]
                + "_BiCEP_fit."
                + self.figformats.value
            )

    def enablerun(self, a):
        """
        Enables running the GUI after choosing a file

        Inputs:
        ------
        a: Button pressed object
        Has no practical use

        Returns
        -------
        None
        """
        with self.output:
            self.run_wid.disabled = False

    def activate_deactivate(self, a):
        """
        Function that excludes/includes a specimen depending on activation/deactivation

        Inputs:
        ------
        a: interact object
        Has no practical use

        Returns
        -------
        None
        """
        with self.output:
            thellierData[self.site_wid.value][
                self.specimen_wid.value
            ].active = not self.checkbox.value
            if (
                thellierData[self.site_wid.value][self.specimen_wid.value].active
                == False
            ):
                thellierData[self.site_wid.value][self.specimen_wid.value].change_temps(
                    min(
                        thellierData[self.site_wid.value][self.specimen_wid.value].temps
                    ),
                    min(
                        thellierData[self.site_wid.value][self.specimen_wid.value].temps
                    ),
                )
                thellierData[self.site_wid.value][
                    self.specimen_wid.value
                ].save_changes()

    def save_to_netcdf(self, a):
        """
        Function that saves site fit to netCDF

        Inputs:
        ------
        a: interact object
        Has no practical use

        Returns
        -------
        None
        """
        with self.output:
            try:
                thellierData[self.site_wid.value].fit.to_netcdf(
                    thellierData.wd + self.site_wid.value + ".nc"
                )
            except:
                pass
