#Runing this will install Numpy, Matplotlib, and optionally Scipy
#They can be installed manually, instead

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
print("Numpy installed")

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
print("Matplotlib installed")

if input("Do you want to install scipy too (y/n)? ").strip() in ('y', 'Y', 'yes', 'Yes'):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    print("Scipy installed")
