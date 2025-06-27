Installation
============
First, install setuptools.

.. code-block:: bash
   
   pip install --upgrade setuptools


Then, clone the repository:

.. code-block:: bash
   
   git clone https://github.com/bcych/BiCEP_GUI.git

It may be best to create a new anaconda environment or pip virtual environment to install BiCEP_GUI into, to avoid conflicts with existing installations.

For a new anaconda environment (in Windows use the anaconda shell for this step).

.. code-block:: bash

   conda create -n BiCEP 
   conda activate BiCEP

For a new pip virtual environment (replace VIRTUAL_ENVIRONMENT_PATH with the path where you want to put your virtual environment)

.. code-block:: bash
   
   python3 -m venv VIRTUAL_ENVIRONMENT_PATH
   source VIRTUAL_ENVIRONMENT_PATH/bin/activate

Or on Windows:

.. code-block:: bash

   py -m venv VIRTUAL_ENVIRONMENT_PATH
   VIRTUAL_ENVIRONMENT_PATH\Scripts\activate

Navigate to the directory and install the `build` module with pip.

.. code-block:: bash
   
   python3 -m pip install --upgrade build

Build the project

.. code-block:: bash
  
   python3 -m build

Install the package:

.. code-block:: bash
   
   python3 -m pip install ./
