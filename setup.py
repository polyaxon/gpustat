from setuptools import setup


def read_readme():
    with open('README.md') as f:
        return f.read()


install_requires = [
    'nvidia-ml-py3==7.352.0',
    'psutil==5.4.7',
    'blessings==1.6.1',
]

tests_requires = [
    'mock==2.0.0',
    'pytest==3.7.0',
]

setup(
    name='polyaxon-gpustat',
    version='0.2.8',
    license='MIT',
    description='An utility to monitor NVIDIA GPU status and usage',
    url='https://github.com/polyaxon/polyaxon-gpustat',
    author='Mourad Mourafiq',
    author_email='mourad.mourafiq@gmail.com',
    keywords='nvidia-smi gpu cuda monitoring gpustat',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: System :: Monitoring',
    ],
    py_modules=['polyaxon_gpustat'],
    install_requires=install_requires,
    extras_require={'test': tests_requires},
    setup_requires=['pytest-runner'],
    tests_require=tests_requires,
    include_package_data=True,
    zip_safe=False,
)
