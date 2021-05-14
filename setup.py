from setuptools import setup

with open('requirements.txt') as file:
    install_requires = file.read().splitlines()

with open("README.md", "r", encoding="UTF-8") as file:
    long_description = file.read()

setup(
    name='stc',
    version='0.1.0',
    packages=['stc'],
    python_requires='>=3.7',
    install_requires=install_requires,
    license='GPLv3',
    author='Emanuele Guidotti',
    author_email='emanuele.guidotti@unine.ch',
    description='Sparse Tensor Classifier',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://sparsetensorclassifier.com',
    project_urls={
        'Documentation': 'https://sparsetensorclassifier.com/docs/',
        'Source': 'https://github.com/sparsetensorclassifier',
        'Tracker': 'https://github.com/sparsetensorclassifier/stc/issues',
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Database',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)