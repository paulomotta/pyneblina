from distutils.core import setup, Extension
neblina_module = Extension('neblina',
                        define_macros = [('MAJOR_VERSION', '0'),
                                         ('MINOR_VERSION', '2')],
                        include_dirs = ['../neblina-core/include'],
                        libraries = ['neblina-core'],
                        library_dirs = ['../neblina-core'],
                        sources = ['neblina_wrapper.c'])
setup(name = 'NeblinaExtension',version='0.2',description = 'This is the neblina math package',ext_modules = [neblina_module])
