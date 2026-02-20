This model creates its graph indexing on the fly using a cpp neighbor distance function. Before using this, it must be compiled 

Temporary manual solution: to compile the cpp extension for fast nearest neighbors
run the following from this directory:

'python setup.py build_ext --inplace'

it might take a few minutes to finish 

TODO: set up compiling on pip install