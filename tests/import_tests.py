#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-LAM

Testing installation
"""

print("\n --- THIRD PARTIES ---")
import numpy
print(numpy, f"version={numpy.__version__}")
import pandas
print(pandas, f"version={pandas.__version__}")
import torch
print(torch, f"version={torch.__version__}")
import torch_geometric
print(torch_geometric, f"version={torch_geometric.__version__}")


print("\n --- FIRST PARTIES ---")
import neural_lam
print(neural_lam, f"version={neural_lam.__version__}")
import mera_explorer
print(mera_explorer, f"version={mera_explorer.__version__}")
import metplotlib
print(metplotlib, f"version={metplotlib.__version__}")
