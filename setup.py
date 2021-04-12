from distutils.core import setup, Extension
factorial_module = Extension('cmath',
                        define_macros = [('MAJOR_VERSION', '0'),
                                         ('MINOR_VERSION', '1')],
                        include_dirs = ['/home/paulo/Documentos/pesquisa/quantum/neblina-core/include'],
                        libraries = ['neblina-core'],
                        library_dirs = ['/home/paulo/Documentos/pesquisa/quantum/neblina-core'],
                        sources = ['neblina_wrapper.c'])
setup(name = 'MathExtension',version='1.0',description = 'This is a math package',ext_modules = [factorial_module])
