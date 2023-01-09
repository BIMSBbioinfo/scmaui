#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="scmaui",
    version="0.0.6",
    license="LGPL-3.0-or-later",
    description="Negative multinomial variational auto-encoder",
    long_description=open("README.rst").read(),
    author="Wolfgang Kopp",
    author_email="wolfgang.kopp@mdc-berlin.de",
    url="https://github.com/wkopp/scmaui",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"scmaui": ["resources/*.h5ad", "resources/*.csv"]},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Documentation": "https://scmaui.readthedocs.io/",
        "Changelog": "https://scmaui.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": "https://github.com/wkopp/scmaui/issues",
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    install_requires=[
        "scikit-learn",
        "tensorflow==2.9.0",
        "numpy<1.24",
        "protobuf==3.19.*",
        "pandas",
        "scipy",
        "scanpy",
        "anndata",
        "louvain",
        "igraph",
    ],
    extras_require={},
    entry_points={
        "console_scripts": [
            "scmaui = scmaui.cli:main",
        ]
    },
)
