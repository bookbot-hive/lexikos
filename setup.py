from setuptools import find_packages, setup
from pathlib import Path

this_path = Path(__file__).parent

readme_path = this_path / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    setup(
        name="lexikos",
        description="A collection of pronunciation dictionaries and neural grapheme-to-phoneme models.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="w11wo",
        author_email="wilson@bookbotkids.com",
        url="https://github.com/bookbot-hive/lexikos",
        license="Apache License",
        packages=find_packages(),
        include_package_data=True,
        platforms=["linux", "unix", "windows"],
        python_requires=">=3.7",
    )
