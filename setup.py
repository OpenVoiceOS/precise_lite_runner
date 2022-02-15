from setuptools import setup

setup(
    name='precise_lite_runner',
    version='0.4.0',
    packages=['precise_lite_runner'],
    url='https://github.com/OpenVoiceOS/precise_lite_runner',
    license='Apache-2.0',
    install_requires=["tflite-runtime",
                      "sonopy==0.1.2",
                      "pyaudio"],
    author='jarbas',
    author_email='jarbasai@mailfence.com',
    description=''
)
