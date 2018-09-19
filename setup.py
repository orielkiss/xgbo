from setuptools import setup, find_packages

setup(
    name='xgbo',
    version='0.1',
    description='Xgboost with Bayesian optimization',
    long_description='Train xgboost calssifiers and regressors with Bayesian optimized hyperparameters.',
    url='http://github.com/guitargeek/xgbo',
    author='Jonas Rembser',
    author_email='jonas.rembser@cern.ch',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
   'numpy',
   'pandas',
   'xgboost',
   'sklearn',
    ]
)
