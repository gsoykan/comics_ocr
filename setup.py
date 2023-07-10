from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(long_description=long_description,
          long_description_content_type='text/markdown', )
