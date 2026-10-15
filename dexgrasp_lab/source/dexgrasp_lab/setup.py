from setuptools import setup,find_packages
setup(name='dexgrasp_lab',version='0.1.0',packages=find_packages(),python_requires='>=3.11',
      description='DexGrasp-Zero shared GNN training in Isaac Lab DirectRLEnv',
      package_data={'dexgrasp_lab':['configs/*.yaml'],'dexgrasp_lab.algorithm':['LICENSE','provenance.json']},
      install_requires=['numpy','scipy','trimesh','PyYAML','urdf-parser-py','torch-geometric'],
      license='BSD-3-Clause; vendored DexGrasp-Zero components under MIT',zip_safe=False)
