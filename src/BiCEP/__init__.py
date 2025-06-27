from importlib import resources
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
fast_ref = resources.files(pkg) / fast_path
slow_ref = resources.files(pkg) / slow_path

fast_path_2 = fast_path + ".stan"
slow_path_2 = slow_path + ".stan"
fast_ref_2 = resources.files(pkg) / fast_path_2
slow_ref_2 = resources.files(pkg) / slow_path_2

__version__ = "1.0.0"

if not fast_ref.exists() or slow_ref.exists():
    with resources.as_file(fast_ref_2) as fast_subpath:
        model_circle_fast = CmdStanModel(stan_file=fast_subpath)
    with resources.as_file(slow_ref_2) as slow_subpath:
        model_circle_slow = CmdStanModel(stan_file=slow_subpath)
else:
    with resources.as_file(fast_ref_2) as fast_subpath:
        model_circle_fast = CmdStanModel(exe_file=fast_subpath)
    with resources.as_file(slow_ref_2) as slow_subpath:
        model_circle_slow = CmdStanModel(exe_file=slow_subpath)
