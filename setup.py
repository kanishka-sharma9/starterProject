from setuptools import find_packages,setup
from typing import List



def get_reqs(path:str)->List[str]:
    HYPHEN_E_DOT='-e .'
    '''
    this func will return the list of requirements
    '''
    requirements=[]
    with open(path) as file:
        requirements=file.readlines()
        requirements=[req.replace('\n','') for req in requirements]
    
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements




setup(
    name="starterProj",
    version='0.0.1',
    author="kans",
    author_email="kanishka.sharma891@gmail.com",
    packages=find_packages(),
    install_requires=get_reqs("requirements.txt"),
    )
