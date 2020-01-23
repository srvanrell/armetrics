from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='armetrics',
      version='0.1.10',
      description='Continuous Activity Recognition Metrics based on (Ward et al, 2011)',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url='http://github.com/srvanrell/armetrics',
      author='srvanrell',
      classifiers=[
          "Programming Language :: Python :: 3",
          'License :: OSI Approved :: MIT License'
      ],
      author_email='srvanrell@gmail.com',
      license='MIT',
      packages=['armetrics'],
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.5',
      install_requires=['matplotlib>=3.0.2',
                        'numpy>=1.13.1',
                        'pandas>=0.20.1'])
