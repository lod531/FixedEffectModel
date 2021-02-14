from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


dependencies = [
    'pandas>=0.16.0',
    'numpy>=1.9.2',
    'scipy',
    'statsmodels',
    'networkx',
    'pyhdfe',
]

setup(name='FixedEffectModelPyHDFE',
      version='0.0.2',
      description='Solutions to linear model with high dimensional fixed effects.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      author='ksecology',
      author_email='da_ecology@kuaishou.com',
      url='https://github.com/lod531/FixedEffectModel',
      packages=find_packages(),
      install_requires=dependencies,
      zip_safe=False,
      license='MIT',
      python_requires='>=3.6',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Sociology',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Operating System :: OS Independent',
      ]
      )
