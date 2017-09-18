from setuptools import setup

setup(name='armetrics',
      version='0.1.2',
      description='Continuous Activity Recognition Metrics based on (Ward et al, 2011)',
      url='http://github.com/srvanrell/armetrics',
      author='srvanrell',
      author_email='srvanrell@gmail.com',
      license='MIT',
      packages=['armetrics'],
      zip_safe=False,
      install_requires=['matplotlib',
                        'numpy',
                        'pandas'])
