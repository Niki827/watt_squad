from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
    requirements = [x.strip() for x in content if "git+" not in x]

setup(name='wattsquad',
      version="0.0.1",
      description="Watt_Squad Model (api_pred)",
    #   license="MIT",
    #   author="Le Wagon",
    #   author_email="contact@lewagon.org",
      #url="https://github.com/Niki827/watt_squad/",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
