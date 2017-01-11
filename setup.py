#   PySOR - solve Poisson's equation with successive over-relaxation.
#   Copyright (C) 2017  Christoph Wehmeyer
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup
from setuptools import Extension
import versioneer
import sys
import os

class lazy_cythonize(list):
    """evaluates extension list lazyly.
    pattern taken from http://tinyurl.com/qb8478q"""
    def __init__(self, callback):
        self._list, self.callback = None, callback
    def c_list(self):
        if self._list is None: self._list = self.callback()
        return self._list
    def __iter__(self):
        for e in self.c_list(): yield e
    def __getitem__(self, ii): return self.c_list()[ii]
    def __len__(self): return len(self.c_list())

def extensions():
    from numpy import get_include
    from Cython.Build import cythonize
    ext_fast_sor = Extension(
        "pysor.fast_sor",
        sources=["ext/fast_sor.pyx", "ext/src_fast_sor.c"],
        include_dirs=[get_include()],
        extra_compile_args=["-O3", "-std=c99"])
    exts = [ext_fast_sor]
    return cythonize(exts)

def get_cmdclass():
    versioneer_cmds = versioneer.get_cmdclass()
    class sdist(versioneer_cmds['sdist']):
        """ensure cython files are compiled to c, when distributing"""
        def run(self):
            # only run if .git is present
            if not os.path.exists('.git'):
                print("Not on git, can not create source distribution")
                return
            try:
                from Cython.Build import cythonize
                print("cythonizing sources")
                cythonize(extensions())
            except ImportError:
                warnings.warn('sdist cythonize failed')
            return versioneer_cmds['sdist'].run(self)
    versioneer_cmds['sdist'] = sdist
    from setuptools.command.test import test as TestCommand
    class PyTest(TestCommand):
        user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]
        def initialize_options(self):
            TestCommand.initialize_options(self)
            self.pytest_args = ['pysor']
        def run_tests(self):
            # import here, cause outside the eggs aren't loaded
            import pytest
            errno = pytest.main(self.pytest_args)
            sys.exit(errno)
    versioneer_cmds['test'] = PyTest
    return versioneer_cmds

setup(
    cmdclass=get_cmdclass(),
    ext_modules=lazy_cythonize(extensions),
    name='pysor',
    version=versioneer.get_version(),
    description="Solve Poisson's equation with successive over-relaxation",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'],
    keywords=[
        'poisson equation',
        'sor',
        'successive over-relaxation'],
    url='https://github.com/cwehmeyer/pysor',
    author='Christoph Wehmeyer',
    author_email='christoph.wehmeyer@fu-berlin.de',
    license='GPLv3+',
    packages=['pysor'],
    install_requires=['numpy>=1.7.0', 'cython>=0.22'],
    tests_require=['pytest'])
