from setuptools import setup, find_packages

# Setup for Google Cloud ML Engine
setup(name='trainer',
      version='0.2',
      packages=find_packages(),
      description='Training Framework',
      author='Marek Ryn',
      author_email='marek@nautisys.pl',
      license='MIT',
      install_requires=[
          'h5py',
      ],
      zip_safe=False)
