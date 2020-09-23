import io
import os
from setuptools import setup

version_txt = os.path.join(os.path.dirname(__file__), 'CPFcluster', 'version.txt')
with open(version_txt, 'r') as f:
    version = f.read().strip()

setup(
    author='Joshua Tobin, Mimi Zhang',
    author_email='tobinjo@tcd.ie',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    description='An implementation of Component-wise Peak Finding Clustering Method',
    install_requires=['numpy', 'scipy', 'aghasher', 'itertools', 'multiprocessing'],
    keywords=['density-peak-clustering', 'clustering', 'mixed-attribute-data', 'machine-learning'],
    license='MIT',
    long_description=io.open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    name='CPFcluster',
    package_data={'CPFcluster': ['version.txt']},
    packages=['CPFcluster'],
    url='https://github.com/tobinjo96/CPFcluster',
    version=version,
)
