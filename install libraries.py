#Runing this will install Numpy and Matplotlib
#They can be installed manually, instead

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
print("Numpy installed")

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
print("Matplotlib installed")
