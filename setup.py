from setuptools import setup

setup(
    name='precise_lite_runner',
    version='0.0.1',
    packages=['precise_lite_runner'],
    url='https://github.com/OpenVoiceOS/precise_lite_runner',
    license='',
    install_requires=["tflit",
                      "sonopy==0.1.2",
                      "wavio==0.0.4",
                      "pyaudio"],
    author='jarbas',
    author_email='jarbasai@mailfence.com',
    description=''
)
