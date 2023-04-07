import io
from setuptools import setup, find_packages

def readme():
    with io.open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='tensormodel',
      version='0.1.1',
      install_requires=[
          'linora>=1.4.0', 
#           'opencv-python>=4.5.0',
#           'tensorflow>=2.7.0',
#           'paddleocr>=2.6.1.3'
      ],
      description='Deep learning application collection.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/tensormodel',
      author='JinQing Lee',
      author_email='hourout@163.com',
      keywords=['machine-learning', 'image', 'text', 'data-science', 
                'deep-learning', 'model'],
      license='Apache License Version 2.0',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      packages=find_packages(),
      zip_safe=False)
