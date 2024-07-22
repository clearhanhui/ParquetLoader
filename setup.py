from setuptools import setup

setup(
    name='parquet_loader',
    version='0.1',
    author='hanhui',
    author_email='clearhanhui@gmail.com',
    description='A distributed parquet dataloader',
    packages=['parquet_loader'],
    install_requires=[
        'torch', 'pandas', 'pyarrow'
    ],
)