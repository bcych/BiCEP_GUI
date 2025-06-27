import os
import pkg_resources
import pickle
from cmdstanpy import install_cmdstan, cmdstan_path, CmdStanModel

pkg = __name__

try:
    cmdstan_path()
except ValueError:
    if platform.system == "Windows":
        install_cmdstan(compiler=True)
    else:
        install_cmdstan()

fast_path = "models/model_circle_fast"
slow_path = "models/model_circle_slow"

__version__ = "1.0.0"

print(pkg_resources.resource_filename(pkg, fast_path))
if not pkg_resources.resource_exists(
    pkg, fast_path
) or not pkg_resources.resource_exists(pkg, slow_path):
    fast_subpath = pkg_resources.resource_filename(pkg, fast_path) + ".stan"
    slow_subpath = pkg_resources.resource_filename(pkg, slow_path) + ".stan"
    model_circle_fast = CmdStanModel(
        model_name="model_circle_fast", stan_file=fast_subpath
    )
    model_circle_slow = CmdStanModel(
        model_name="model_circle_slow", stan_file=slow_subpath
    )
else:
    fast_subpath = pkg_resources.resource_filename(pkg, fast_path)
    slow_subpath = pkg_resources.resource_filename(pkg, slow_path)
    model_circle_fast = CmdStanModel(exe_file=fast_subpath)
    model_circle_slow = CmdStanModel(exe_file=slow_subpath)
