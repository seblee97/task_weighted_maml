import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="maml-pq",
    version="0.0.1",
    author="Sebastian Lee",
    author_email="sebalexlee@gmail.com",
    description="Package with MAML implementation. Research into affect of priority queue in inner loop sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seblee97/maml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
