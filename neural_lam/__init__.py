#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-LAM: Graph-based neural weather prediction models for Limited Area Modeling

https://github.com/ThomasRieutord/neural-lam
"""
import os

PACKAGE_ROOTDIR = os.path.dirname(os.path.realpath(__path__[0]))

with open(os.path.join(PACKAGE_ROOTDIR, "pyproject.toml"), "r") as f:
    for l in f.readlines():
        if "version =" in l:
            __version__ = l.split('"')[1]
            break

del f, l, os
