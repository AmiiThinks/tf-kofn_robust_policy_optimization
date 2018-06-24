from setuptools import setup, find_packages

setup(
    name='tf-kofn_robust_policy_optimization',
    version='0.0.2',
    license='',
    packages=find_packages(),
    install_requires=[
        'setuptools >= 20.2.2',
        # tensorflow or tensorflow-gpu >= v1.8
        'fire',
        'numpy',
        'scipy'  # For environments/inventory.
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
