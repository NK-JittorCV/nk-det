from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    with open('jittordet/version.py', 'r') as f:
        exec(compile(f.read(), 'jittordet/version.py', 'exec'))
    return locals()['__version__']


def get_requires():
    with open('requirements.txt', encoding='utf-8') as f:
        requires = [line for line in f]
    return requires


if __name__ == '__main__':
    setup(
        name='jittordet',
        version=get_version(),
        description='An object detection codebase based on Jittor.',
        long_description=readme(),
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=('configs', 'tools')),
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        install_requires=get_requires())
