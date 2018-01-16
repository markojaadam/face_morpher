from setuptools import setup, find_packages
from setuptools.command.install import install
import os

# To test locally: python setup.py install
# To upload to pypi: python setup.py sdist upload
class OverrideInstall(install):
  def run(self):
    uid, gid = 0, 0
    mode = '0700'
    install.run(self) # install everything as per usual
    for filepath in self.get_outputs():
      if 'bin/stasm_util' in filepath:
        # make binaries executable
        os.chmod(filepath, 0o755)

setup(
  name='facemorpher_memlab',
  version='0.1.0',
  description=('Warp, morph and average human faces!'),
  keywords='face morphing, averaging, warping',
  packages=find_packages(),
  package_data={'facemorpher_memlab': [
    'data/*.xml',
    'bin/stasm_util_osx_cv3.2',
    'bin/stasm_util_osx_cv3.4',
    'bin/stasm_util_linux_cv3.2',
    'bin/stasm_util_linux_cv3.4'
  ]},
  install_requires=[
    'docopt',
    'numpy',
    'scipy',
    'matplotlib',
    'Pillow'
  ],
  cmdclass={'install': OverrideInstall},
  entry_points={'console_scripts': [
      'facemorpher=facemorpher_memlab.morpher:main',
      'faceaverager=facemorpher_memlab.averager:main'
    ]
  },
  data_files=[('readme', ['README.rst'])],
  long_description=open('README.rst').read(),
)
