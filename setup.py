from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'sklearn',
    'tqdm',
    'stribor==0.2.0',
    'torch>=1.11.0',
    'torchvision',
    'torchdiffeq==0.2.2',
    'torchcde',
    'torchsde',
    'pytorch_lightning'
]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='nfsde',
      version='0.1.0',
      description='Neural SDE',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=install_requires,
      packages=find_packages('.'),
      python_requires='>=3.9',
      zip_safe=False)