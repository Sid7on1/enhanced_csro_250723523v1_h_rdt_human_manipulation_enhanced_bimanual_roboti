import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from typing import List, Dict

# Constants
PROJECT_NAME = "enhanced_cs"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
DESCRIPTION = "Enhanced AI project based on cs.RO_2507.23523v1_H-RDT-Human-Manipulation-Enhanced-Bimanual-Roboti"
URL = "https://github.com/your-username/your-repo-name"
REQUIRES_PYTHON = ">=3.8.0"
REQUIRED_PACKAGES: List[str] = [
    "torch",
    "numpy",
    "pandas",
]

# Configuration
class Configuration:
    def __init__(self):
        self.project_name = PROJECT_NAME
        self.version = VERSION
        self.author = AUTHOR
        self.author_email = AUTHOR_EMAIL
        self.description = DESCRIPTION
        self.url = URL
        self.requires_python = REQUIRES_PYTHON
        self.required_packages = REQUIRED_PACKAGES

    def get_project_name(self) -> str:
        return self.project_name

    def get_version(self) -> str:
        return self.version

    def get_author(self) -> str:
        return self.author

    def get_author_email(self) -> str:
        return self.author_email

    def get_description(self) -> str:
        return self.description

    def get_url(self) -> str:
        return self.url

    def get_requires_python(self) -> str:
        return self.requires_python

    def get_required_packages(self) -> List[str]:
        return self.required_packages


class CustomInstallCommand(install):
    def run(self):
        try:
            install.run(self)
        except Exception as e:
            print(f"An error occurred during installation: {e}")


class CustomDevelopCommand(develop):
    def run(self):
        try:
            develop.run(self)
        except Exception as e:
            print(f"An error occurred during development installation: {e}")


class CustomEggInfoCommand(egg_info):
    def run(self):
        try:
            egg_info.run(self)
        except Exception as e:
            print(f"An error occurred during egg info generation: {e}")


def main():
    configuration = Configuration()
    setup(
        name=configuration.get_project_name(),
        version=configuration.get_version(),
        author=configuration.get_author(),
        author_email=configuration.get_author_email(),
        description=configuration.get_description(),
        url=configuration.get_url(),
        python_requires=configuration.get_requires_python(),
        packages=find_packages(exclude=["tests", "tests.*"]),
        install_requires=configuration.get_required_packages(),
        include_package_data=True,
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
            "egg_info": CustomEggInfoCommand,
        },
    )


if __name__ == "__main__":
    main()