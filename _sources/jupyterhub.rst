JupyterHub Setup
============
To use BiCEP GUI, we recommend using the Earthref jupyterhub at http://jupyterhub.earthref.org. To run the GUI from thissite, first run the Bicep-GUI-Setup notebook by clicking on this and pressing the run button until you reach the end of the notebook. Note that this setup may take several minutes.

.. image:: readme-image/jupyterhub-run.png
  :width: 649
  :alt: JupyterHub run bar

You will have a directory called BiCEP_GUI in your jupyterhub. Navigate to this.

Before using BiCEP_GUI on your own data, you will need to upload MagIC formatted files containing your paleointensity data. You can create these files using pmag_gui (part of the PmagPy package, see https://github.com/ltauxe/PmagPy_tutorials) or at http://paleointensity.org

Upload your own measurements.txt, specimens.txt, samples.txt and sites.txt files to the BiCEP_GUI directory using the upload button in JupyterHub.

.. image:: readme-image/jupyterhub-upload.png 
  :width: 1157
  :alt: Button to upload on JupyterHub 

If you encounter any problems in the JupyterHub site, try pressing "Control Panel" in the top right and "Stop My Server". You will then be offered the opportunity to restart your JupyterHub server.
