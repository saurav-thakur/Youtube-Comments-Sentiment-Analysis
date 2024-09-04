import setuptools

with open("README.md","r",encoding="utf-8") as f:
    long_desc = f.read()

setuptools.setup(
    name="youtube_sentiment",
    version="0.1.0",
    author="Saurav",
    author_email="thakursaurav1341@gmail.com",
    description="youtube comments sentiment classification",
    long_description=long_desc,
    packages=setuptools.find_packages()
)