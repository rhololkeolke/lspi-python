"""A setuptools based setup module."""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lspi-python',
    version='1.0.1',
    description='LSPI algorithm in Python',
    long_description=long_description,
    url='https://github.com/rhololkeolke/lspi-python',
    author='Devin Schwab',
    author_email='digidevin@gmail.com',
    license='BSD-3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='machinelearning ai',
    packages=find_packages(exclude=['docs', '*testsuite', 'test*']),
    install_requires=['numpy', 'scipy'],
    extras_require={
        'test': ['nosetests', 'coverage']
    }
)
