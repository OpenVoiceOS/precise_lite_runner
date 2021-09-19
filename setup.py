from setuptools import setup

setup(
    name='precise_lite_runner',
    version='0.3.3',
    packages=['precise_lite_runner'],
    url='https://github.com/OpenVoiceOS/precise_lite_runner',
    license='',
    install_requires=["tflit",
                      "sonopy==0.1.2",
                      "pyaudio"],
    author='jarbas',
    author_email='jarbasai@mailfence.com',
    description=''
)
