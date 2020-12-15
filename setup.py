# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import sys
import os

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                requirements.append(r)
    return requirements


requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements.txt"))



setup(
    name='uroc',
    version='0.1',
    author='Eva-Maria Walz',
    author_email='evamaria.walz@gmx.de',
    description='Generalization of ROC curves and AUC to real-valued prediction problems',
    long_description=open('README.md').read(),
    license='MIT',
    install_requires=requirements,
    url='https://github.com/evwalz/isodisreg',
    packages=find_packages('src'),
    package_dir={'':'src'},
)

