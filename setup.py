from setuptools import setup, find_packages

setup(
    name='k_of_n_mdp_policy_opt',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=[
        'setuptools >= 20.2.2',
        # tensorflow or tensorflow-gpu >= v1.8
        'fire',
        'numpy',
        'amii_tf_nn',
        'scipy'  # For environments/inventory.
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
