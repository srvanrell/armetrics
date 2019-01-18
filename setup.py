from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='armetrics',
      version='0.1.6b2',
      description='Continuous Activity Recognition Metrics based on (Ward et al, 2011)',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='http://github.com/srvanrell/armetrics',
      author='srvanrell',
      author_email='srvanrell@gmail.com',
      license='MIT',
      packages=['armetrics'],
      zip_safe=False,
      install_requires=['matplotlib>=3.0.2',
                        'numpy>=1.13.1',
                        'pandas>=0.20.1'])
