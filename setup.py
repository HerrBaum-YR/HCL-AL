import sys
import platform
from setuptools import setup, find_packages


required_packages_base = ['numpy', 'easydict', 'future']
required_packages_ext = []

infer_tag = ''
infer_key = '--infer'
if infer_key in sys.argv:
   required_packages_ext = []
   sys.argv.remove(infer_key)
   infer_tag = '+infer'

setup(name='md_clip3d',
      version='1.0.0'+infer_tag,
      description='UII 3D Clip3d Engine',
      url='',
      author='UII 3D Clip Team',
      author_email='yiran.shu@uii-ai.com',
      license='',
      packages=find_packages(),
      install_requires=required_packages_base + required_packages_ext,
      include_package_data=False,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts':
              ['clip_coarse_train=md_clip3d.clip_coarse_train:main',
               'clip_fine_train=md_clip3d.clip_fine_train:main',
               'clip_apply=md_clip3d.inference.clip_apply:main',
               'clip_evaluate=md_clip3d.evaluate.evaluate:clip_evaluate']
      }
)
