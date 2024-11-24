from setuptools import setup, find_packages

setup(
    name="eeg_ml_pipeline",
    version="0.1",  
    description="A Python package for EEG machine learning analysis, classification, and visualization.",
    author="Yossef Emara",
    author_email="emara004@umn.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",            
        "scikit-learn",
        'pandas>=2.0.0',               
        "scipy",
        "mne",
        "matplotlib",
        "seaborn",
        "antropy",
        "featurewiz",
        "imblearn",
        "tensorflow",
        'dask[dataframe]>=2021.9.1',
        'matplotlib',
        'pyts',
        'librosa',
        'torch',
        'PyWavelets',
        're'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
