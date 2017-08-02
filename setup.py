from setuptools import setup, find_packages

setup(
    name='amii_tf_mdp',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=[
        'setuptools >= 20.2.2',
        # tensorflow or tensorflow-gpu v1.2
        'fire'
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
