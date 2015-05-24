# LSPI Python

[![Build Status](https://travis-ci.org/rhololkeolke/lspi-python.svg?branch=master)](https://travis-ci.org/rhololkeolke/lspi-python)

This is a Python implementation of the Least Squares Policy Iteration (LSPI) reinforcement learning algorithm.
For more information on the algorithm please refer to the paper

“Least-Squares Policy Iteration.”  
Lagoudakis, Michail G., and Ronald Parr.   
Journal of Machine Learning Research 4, 2003.   
[https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf](https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf)  

You can also visit their website where more information and a Matlab version is provided.

[http://www.cs.duke.edu/research/AI/LSPI/](http://www.cs.duke.edu/research/AI/LSPI/)

## Requirements

The requirements.txt file contains the python module requirements to use this
library, run the tests, and generate the docs. To install all of the listed
requirements automatically you can use the command

```
pip install -r requirements.txt
```

## Testing

If you have nosetests you can run the tests with `nosetests --config=setup.cfg lspi_testsuite`.
If you have virtual environment installed you can run `make test` which will automatically create a virtual environment
with all of the dependencies and then run the tests.

## Docs

To generate the docs you will need sphinx. If you have virtual environment installed you can run
`make html-docs`. This will automatically create a virtual environment with all of the dependencies
and then run sphinx. The output will exist in `docs/build/html/`.
