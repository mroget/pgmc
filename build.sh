#!/bin/bash

pip uninstall -y pgmc
rm -rf dist
python3 -m build
pip install dist/*.whl