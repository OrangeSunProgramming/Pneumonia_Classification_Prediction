from setuptools import setup, find_packages

setup(
    name='pneumonia_classification_prediction',
    version='0.1.0',
    description='A deep learning project for classifying chest X-rays as pneumonia or normal.',
    author='Marcos Masip',
    author_email='marcosmasipcompany@gmail.com',
    url='https://github.com/OrangeSunProgramming/Pneumonia_Classification_Prediction',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'matplotlib',
        'Pillow',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
