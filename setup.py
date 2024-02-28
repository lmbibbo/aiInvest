from setuptools import setup, find_packages

setup(
    name='AiInvest',
    version='1.0.0',
    description='Una descripción corta del proyecto',
    packages=find_packages(),  # Busca automáticamente todos los paquetes del proyecto
    install_requires=[  # Lista de dependencias requeridas para el proyecto
        'scikit-learn',
    ],
    author='Luis Mariano Bibbo',
    author_email='lmbibbo@gmail.com',
    url='https://github.com/lmbibbo/aiinvest',
)
