import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = 'dtcwtfusion',
    version = '0.1.0dev1',
    author = 'Rich Wareham',
    author_email = 'rjw57@cam.ac.uk',
    description = 'Experimental toolchain for registration of image sequences',
    license = 'To be determined',
    keywords = 'numpy, wavelet, complex wavelet, DT-CWT, image, registration, alignment, vision',
    url = '', # No URL as yet
    packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    long_description=read('README.rst'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],

    entry_points = {
        'console_scripts': [
            'fuseimages = dtcwtfusion.fuseimages:main',
            'fusevideo = dtcwtfusion.fusevideo:main',
            'hdf5toraw = dtcwtfusion.hdf5toraw:main',
            'hdf5toimg = dtcwtfusion.hdf5toimg:main',
        ],
    },

    package_data = {
        'dtcwtfusion': [],
    },

    setup_requires=[ 'nose>=1.0', ],

    install_requires=[
        'numpy', 'six', 'scipy', 'pillow', 'dtcwt>0.8.0', 'docopt',
        'h5py',
    ],

    extras_require={
        'docs': [ 'sphinx', 'docutils', 'matplotlib', 'ipython', ],
    },

    tests_require=[ 'coverage', ],
)

# vim:sw=4:sts=4:et
