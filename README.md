armetrics
=========

Ongoing work...

Python implementation of some metrics proposed in:

J. A. Ward, P. Lukowicz, and H. W. Gellersen, “Performance Metrics for Activity Recognition,” ACM Trans. Intell. Syst. Technol., vol. 2, no. 1, p. 6:1–6:23, Jan. 2011.

Installation
------------

This system has been developed for a Linux environment.

System-wide installation

    sudo pip3 install armetrics

Single-user installation

    pip3 install --user armetrics

Virtualenv installation (recommended)

    pipenv install armetrics

Uninstallation

    [sudo] pip3 uninstall armetrics

Update
------

System-wide update

    sudo pip3 install -U armetrics

Single-user update (recommended)

    pip3 install --user -U armetrics

Usage
-----

There are three examples in the repo (that should be improved)

Development
-----------

Install locally from source (source directory will immediately affect the installed package
without needing to re-install):

    pip3 install --user --editable .

Update version at `setup.py` and then create a source distribution

    python3 setup.py sdist bdist_wheel

Upload to PyPI

    twine upload dist/*
