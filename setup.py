#!/usr/bin/env python
# -*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: mage
# Mail: mage@woodcol.com
# Created Time:  2018-1-23 19:17:34
#############################################


from setuptools import setup, find_packages

setup(
    name="AAdeepLearning",
    version="1.0.8",
    keywords=("AAdeepLearning", "AA", "deepLearning frame"),
    description="AAdeepLearning is a deep learning frame",
    long_description="AAdeepLearning is a deep learning frame",
    license="MIT Licence",

    url="https://github.com/luojiangtao/aadeeplearning",
    author="luojiangtao",
    author_email="1368761119@qq.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=['numpy']
)
