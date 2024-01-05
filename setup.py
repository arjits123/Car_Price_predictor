from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='carpricepredictor',
version='0.0.1',
author='Arjit Sharma',
author_email='arjitpkt96@gmail.com',
packages = find_packages(), # this will find __init__.py in every folder and that will acts as a package
install_requires = get_requirements('requirement.txt')

)