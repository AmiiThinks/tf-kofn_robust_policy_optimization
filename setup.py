from setuptools import setup, find_packages

setup(
    name='tf-kofn_robust_policy_optimization',
    version='1.0.3',
    license='',
    packages=find_packages(),
    install_requires=[
        'setuptools >= 20.2.2',
        'tensorflow >= 2.0',
        'fire',
        'numpy',
        'scipy',  # For environments/inventory.
        'deprecation',
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
