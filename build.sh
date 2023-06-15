#!/bin/bash

pip uninstall -y pgmc
rm -rf dist
python3 -m build
python3 -m twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ pgmc