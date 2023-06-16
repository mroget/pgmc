# pgmc
This package implements the K-PGMC algorithm. (ref paper ?)

## Install
### Build from soource (recommended)
It is recommended to build the package from source since I do not promise that the pip version will be up to date.
You can run the `build.sh` script. Alternatively you can remove the `dist` folder and run the following commands:
```
pip uninstall -y pgmc
python3 -m build
pip install dist/*.whl
```

### PIP Testing installation
For now, the package is on pip test (an instance of pip used to test packages). To install it from pip run 
```
pip install -i https://test.pypi.org/simple/ pgmc
```

## Use
Check if you can import the package correctly : `import pgmc`.
The package as two modules: 
	+ `embeddings` : contains the three implemnted embeddings.
	+ `kpgmc` : contains an implementation of the kpgmc algorithm which is complient with sklearn estimators. It means that it can be sent to the sklearn fonctions like the pipeline, the model selection or the cross validation.

For an example of use, see the notebook `demo.ipynb`.