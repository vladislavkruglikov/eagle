import setuptools


setuptools.setup(
    name="eagle",
    version="0.0.1",
    description="Pretty and simple to use implementation of extrapolation algorithm for greater language model efficiency 🦅",
    author="Vladislav Kruglikov",
    author_email="vladislavkruglikov@icloud.com",
    url="https://github.com/vladislavkruglikov/eagle",
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "accelerate==1.7.0",
        "datasets==3.6.0",
        "torch==2.5.1",
        "transformers==4.52.3",
        "wandb==0.19.11",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
