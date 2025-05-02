from setuptools import setup, find_packages

setup(
    name="pseudotime_corr",
    version="0.1.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="Compute non-linear gene–pseudotime correlations",
    url="https://github.com/yourusername/pseudotime_corr",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
