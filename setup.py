from distutils.core import setup, Extension
import numpy

neblina_module = Extension('neblina',
                        define_macros = [('MAJOR_VERSION', '0'),
                                         ('MINOR_VERSION', '4')],
                        include_dirs = ['../neblina-core/include',numpy.get_include()],
                        libraries = ['neblina-core'],
                        library_dirs = ['/usr/local/lib64'],
                        sources = ['neblina_wrapper.c'])
setup(name = 'NeblinaExtension',version='0.4',description = 'This is the neblina math package',ext_modules = [neblina_module],
      include_dirs=[numpy.get_include()])
