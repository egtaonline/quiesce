import setuptools


setuptools.setup(
    name='egta',
    version='0.0.6',
    description='Scripts to perform EGTA',
    url='https://github.com/egtaonline/quiesce.git',
    author='Strategic Reasoning Group',
    author_email='strategic.reasoning.group@umich.edu',
    license='Apache 2.0',
    entry_points=dict(console_scripts=['egta=egta.__main__:main']),
    install_requires=[
        'gameanalysis~=3.2',
        'egtaonlineapi~=0.2',
        'numpy~=1.13',
    ],
    packages=['egta', 'egta.script'],
)
