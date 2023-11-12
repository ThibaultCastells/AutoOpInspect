from setuptools import setup, find_packages

setup(
    name='auto_op_inspect',
    version='1.0',
    author='Thibault Castells',
    description='AutoOpInspect is a Python tool designed to facilitate the inspection and profiling of operators within PyTorch models, aiming to assist machine learning developers, researchers, and enthusiasts in debugging, optimizing, and understanding PyTorch models more efficiently and effectively.',
    url='https://github.com/ThibaultCastells/AutoOpInspect', 
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(),
    install_requires=[
        'torch',
        'tqdm'
    ],
    python_requires='>=3.6',
)
