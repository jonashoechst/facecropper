from setuptools import setup, find_packages


with open('Readme.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='facecropper',
    version='0.1',
    description='Detect and crop faces from images',
    long_description_content_type="text/markdown",
    long_description=readme,
    author='Jonas Höchst',
    author_email='hoechst@mathematik.uni-marburg.de',
    url="https://github.com/hoechst/facecropper",
    license="GPL-3.0",
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="face detection crop image processing facecropper head crop-image circular circle",
    install_requires=requirements,
    zip_safe=True,
    project_urls={
        "Bug Reports": "https://github.com/jonashoechst/facecropper/issues",
        "Source": "https://github.com/jonashoechst/facecropper/",
    },
    entry_points={
        'console_scripts': [
            'facecropper=facecropper.core:main']
    },
)
