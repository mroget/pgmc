#!/bin/bash

./build
python3 -m twine upload --repository testpypi dist/*
sleep 30
pip install -i https://test.pypi.org/simple/ pgmc