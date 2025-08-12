Installation
============

====================================
Conda Environment Setup (Recommended)
====================================

Anaconda is a python environment manager that can manage different python packages in separate environments.For windows users in particular, we recommend using anaconda to manage the python environment. Instructions on how to install anaconda can be found at https://www.anaconda.com/docs/getting-started/anaconda/install

If you already have anaconda installed, we recommend updating your conda version to ensure compatibility with your new environment. This can be done using the command:

.. code-block:: bash
   conda update -n base conda

If anaconda talks about having a broken environment, you may need to uninstall and reinstall anaconda, but try the following steps first.

To create a new environment open a terminal window. In windows, after installing anaconda, you will have an application called "anaconda prompt" installed, which acts like a mac/linux terminal. Into your terminal, type:

.. code-block:: bash
   conda create -f environment.yml

This will install prerequisite packages for BiCEP_GUI, and may take a few minutes. This is highly recommended for windows users in particular, as it installs everything needed for the cmdstanpy package, which BiCEP_GUI relies on. However, the environment can be created manually if needed. If anaconda throws up any errors about the installation of particular packages, they can be deleted from the cmdstanpy file. 

Activate the conda environment using:

.. code-block:: bash
   conda activate BiCEP

This will have to be done every time you want to use BiCEP_GUI

================================
Pip Environment Setup (Optional)
================================

NOTE: The steps in this section are optional if you installed the anaconda environment.

If you don't want to deal with anaconda, and are not on windows, a simple pip virtual environment may be a simpler way to deal with using BiCEP_GUI. You can create a pip virtual environment in the following way, renaming VIRTUAL_ENVIRONMENT_PATH to the name of a directory where you want all of your python packages to be installed:

.. code-block:: bash
   
   python3 -m venv VIRTUAL_ENVIRONMENT_PATH

To enter this environment, use the command (again replacing VIRTUAL_ENVIRONMENT_PATH with the folder your environment is located in):

.. code-block:: bash

   source VIRTUAL_ENVIRONMENT_PATH/bin/activate

This command has to be executed every time you want to use BiCEP GUI, so it may make sense to create an alias for it.

On windows you need to use the command: 

.. code-block:: bash

   py -m venv VIRTUAL_ENVIRONMENT_PATH
   VIRTUAL_ENVIRONMENT_PATH\Scripts\activate

However, we highly recommend using conda on Windows. 

====================
Installing BiCEP GUI
====================

BiCEP GUI is installed through pip. If you set up the environment correctly, pip should be on your system already.

BiCEP GUI can either be downloaded using git, or as a zip file from the github page. To clone the repository using git, use the following command.

.. code-block:: bash
   
   git clone https://github.com/bcych/BiCEP_GUI.git

Navigate to the BiCEP_GUI folder, and run the following commands:

.. code-block:: bash
   pip install --upgrade setuptools
   pip install ./

Finally, BiCEP_GUI is also dependent on some code from pmagpy. If you do not have an existing developer install of PmagPy

.. code-block:: bash
   pip install --upgrade pmagpy --no-deps

With this, BiCEP_GUI should be installed

=====================================
First time launch and Troubleshooting
=====================================

In your terminal window, type:

.. code-block:: bash
   python

This will launch a python prompt. Inside this prompt, type:

.. code-block:: python3
   import BiCEP

If BiCEP has been installed correctly, you will get a series of messages about "compiling stan models". This means that BiCEP GUI is creating the underlying C++ models that define the BiCEP method. If you installed through pip you may see messages about downloading and installing cmdstan. If this fails, see the `cmdstanpy`_. documentation for more info.

.. _cmdstanpy: https://mc-stan.org/docs/cmdstan-guide/installation.html#cpp-toolchain

If everything works, you can begin to use the GUI. To launch, navigate to the BiCEP_GUI folder run the command:

.. code-block:: bash
   jupyter notebook

Now open the BiCEP_GUI notebook and follow the instructions on the page :doc:`gui_usage_guide`. You can also learn more about the deeper workings of the python package at :doc:`code_usage_guide`.
