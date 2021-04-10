# BiCEP GUI Readme

## Information

- BiCEP GUI is a graphical user interface for the BiCEP method of estimation paleointensity (Cych et al, in prep) using jupyter, ipywidgets, and voila. On the backend, BiCEP uses the python version of stan (pystan).

- Planned features include: Changing orientation of Zijderveld plot, ability to save figures separately.

## Setup - jupyterhub

- To use BiCEP GUI, we recommend using the Earthref jupyterhub at http://jupyterhub.earthref.org. To run the GUI from this site, first run the Bicep-GUI-Setup notebook by clicking on this and pressing the run button until you reach the end of the notebook. Note that this setup may take several minutes.

![Run button in jupyterhub](https://raw.githubusercontent.com/bcych/BiCEP_GUI/master/readme-image/jupyterhub-run.png)

- You will then have a directory called BiCEP_GUI in your jupyterhub. Navigate to this.

- Before using BiCEP GUI on your own data, you will need to upload MagIC formatted files containing your paleointensity data. You create these files using pmag_gui, part of the PmagPy package. For help with this, see the tutorial at https://github.com/ltauxe/PmagPy_tutorials

- Upload your measurements.txt, specimens.txt, samples.txt and sites.txt files to the BiCEP_GUI directory using the upload button in jupyterhub.

![Upload button in jupyterhub](https://raw.githubusercontent.com/bcych/BiCEP_GUI/master/readme-image/jupyterhub-upload.png)

- If you encounter any problems using the jupyterhub site, try pressing "Control Panel" in the top right and "Stop My Server". You will then be offered the opportunity to restart your jupyterhub server.

## Setup - local machine

- To use BiCEP GUI on a local machine, you will need Anaconda python. Follow the install instructions at https://docs.anaconda.com/anaconda/install/

- You will need several packages installed to use BiCEP GUI. At the command line, run `pip install pmagpy pystan sklearn ipympl voila tornado --upgrade` to install the required packages for BiCEP GUI.

- Using the command line, navigate to the directory you want your BiCEP GUI folder to be in and clone using the command `git clone https://github.com/bcych/BiCEP_GUI`

- Navigate to the newly created BiCEP_GUI directory, and run the "compile_models.py" python script. This compiles the pystan models as C++ code on your system. Note that if you are on Mac OS Catalina, this may not work if you are using bash as your shell, either switch to zsh or use the jupyterhub method.

- Before using BiCEP GUI on your own data, you will need to upload MagIC formatted files containing your paleointensity data. You create these files using pmag_gui, part of the PmagPy package. For help with this, see the tutorial at https://github.com/ltauxe/PmagPy_tutorials

- Copy and paste your measurements.txt, specimens.txt, samples.txt and sites.txt files into the BiCEP_GUI directory.

## Using BiCEP GUI

- Open the BiCEP GUI notebook in your folder. Press the "Appmode" button (or "Voila" button, located in the same place, if running on your own machine) to launch the GUI.

![Upload button in jupyterhub](https://raw.githubusercontent.com/bcych/BiCEP_GUI/master/readme-image/jupyterhub-appmode.png)

- On launch you should have a GUI with the following layout:

![GUI layout](https://raw.githubusercontent.com/bcych/BiCEP_GUI/master/readme-image/GUI_layout.png)

1. The Convert MagIC data button converts your measurements, sites, specimens files into a csv format that BiCEP_GUI uses, and calls it arai_data.csv. Paleointensity experiments where more than 25% of the NRM was remaining at the highest temperature step are not included in the new file, as it assumed that these are still being worked on. Anisotropy (TRM or ARM), Cooling Rate and Non Linear TRM corrections should be correctly implemented.

2. File selection button. "Use new file" reads in your converted arai_data.csv file from step one. "Use example file" opens the example dataset used in Cych et al (in prep.). The example dataset is stored as arai_data_example.csv

3. Site and specimen dropdowns. These dropdown menus allow you choose a particular paleointensity experiment.

4. Minimum and maximum temperature steps (in Celcius) to use for the paleointensity experiment. We recommend using the Zijderveld plot (7.) to choose which set of temperatures to use. By default, we use all temperature steps to make a paleointensity estimate. Currently it is required to make an estimate for all specimens.

5. Statistics about the direction and alteration of the ChRM used for paleointensity estimation. These may help with choosing which set of temperature steps to use. See the standard paleointensity definitons (Paterson et al, 2014, https://earthref.org/PmagPy/SPD/DL/SPD_v1.1.pdf).

6. Arai plot with zero field first steps plotted as red circles, in field first steps plotted as blue circles, pTRM checks plotted as triangles, and pTRM tail checks plotted as squares. Additivity checks are not currently plotted. Circle fits from the BiCEP method will be plotted as green lines under the Arai plot after the site fit (9) has been performed. All plots can be rescaled using the "move" button (3rd symbol from the bottom on left side of plot) and right clicking and dragging, or the "zoom" button (2nd symbol from the bottom) and left clicking and dragging to zoom in, or right clicking and dragging to zoom out. The "home" button (second symbol from the top) resets the plot axis, as does changing the temperatures.

7. Zijderveld plot of the data, with x,y plotted as black circles and x,z plotted as red squares.

8. "Save Temperatures" button saves the temperatures used for that specimen to RAM and also to file. This must be done for each specimen individually before switching to another one. By default, all temperature steps are used for every specimen.

9. The "Process Site Data" button performs the BiCEP method on that site and calculates the site level paleointensity. Depending on the speed of your computer and the sampler parameters used (10), this may take a while to run, especially for sites with many specimens. Please be patient.

10. Parameters for the MCMC sampler for the BiCEP method. The "n samples" slider increases the number of samples used in the MCMC sampler. Smaller numbers will take less time to run but result in lower accuracy in the resulting probability distribution. The "Sampler" button changes the parameterization of the MCMC sampler slightly (mathematically, the model is the same, but the parameters being sampled from are specified slightly differently). The "Slow, more accurate" sampler is much slower than the "Fast, less accurate" sampler, but generally (though not always) results in better sampler diagnostics than the "Fast, less accurate" sampler, particularly for sites with small numbers of specimens.

11. Plot of the estimated paleointensity for each specimen against Arai plot curvature. The currently displayed specimen in the Arai and Zijderveld plots has a red circle around it in this plot. The blue lines are samples from the posterior distribution for the relationship between specimen level paleointensity and curvature. The y intercept is the estimated site level paleointensity.

12. Histogram of the site level paleointensities sampled from the posterior distribution. This corresponds to the distribution of intercepts of the blue lines in (12.).

13. Diagnostics for the MCMC sampler (see Cych et al, in prep. or the Stan Documentation at https://mc-stan.org/docs/2_26/reference-manual/notation-for-samples-chains-and-draws.html, https://mc-stan.org/docs/2_26/reference-manual/effective-sample-size-section.html). 0.9<R_hat<1.1 and n_eff>1000 is desired, with R_hat=1.00 and n_eff>10000 being ideal. Tweak the sampler parameters (10.) or measure more specimens if these parameters give poor results (indicated by an amber color for n_eff<1000 or a red color for bad R_hat). Also displayed here is the 95% credible interval for the site.

14. Saves figures to file. Currently the Zijderveld plot and Arai plot have to be saved together (as do both site plots).

15. Saves the results from the BiCEP method to the MagIC tables (site and specimen tables are altered).

## Attributions

Paterson, G. A., L. Tauxe, A. J. Biggin, R. Shaar, and L. C. Jonestrask (2014), On improving the selection of Thellier-type paleointensity data, Geochem. Geophys. Geosyst., doi: 10.1002/2013GC005135

Stan Development Team. 2021. Stan Modeling Language Users Guide and Reference Manual, 2.26. https://mc-stan.org

Tauxe, L., R. Shaar, L. Jonestrask, N. L. Swanson-Hysell, R. Minnett, A. A. P. Koppers, C. G. Constable, N. Jarboe, K. Gaastra, and L. Fairchild (2016), PmagPy: Software package for paleomagnetic data analysis and a bridge to the Magnetics Information Consortium (MagIC) Database, Geochem. Geophys. Geosyst., 17, doi:10.1002/2016GC006307

## Licensing
PmagPy and Stan are licensed under a 3-clause BSD license. See (https://opensource.org/licenses/BSD-3-Clause)

## Contact
If you have any issues with this software, feature requests or want to collaborate, feel free to correspond with me at bcych@ucsd.edu or leave an issue or feature request on the github at http://github.com/bcych/BiCEP_GUI
