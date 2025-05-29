from setuptools import setup, find_packages

setup(
    name="synthetic_data_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-experimental>=0.0.49",
        "langchain-openai>=0.0.5",
        "langchain-core>=0.1.17",
        "langchain-community>=0.0.19",
        "openai>=1.12.0",
        "pydantic>=2.6.1",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.1",
        "plotly>=5.18.0"
    ],
    python_requires=">=3.8",
) 