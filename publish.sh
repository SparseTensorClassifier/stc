#!/bin/bash

# remove previous releases
rm -rf build/ dist/ stc.egg-info/ __pycache__/
# compile
python setup.py sdist bdist_wheel
# publish
python -m twine upload dist/*
