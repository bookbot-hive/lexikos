from setuptools import find_packages, setup
from pathlib import Path

this_path = Path(__file__).parent
module_dir = this_path / "lexikos"

readme_path = this_path / "README.md"
requirements_path = this_path / "requirements.txt"

long_description = readme_path.read_text(encoding="utf-8")

data_dir = module_dir / "dict"
data_files = [
    str(f.relative_to(module_dir)) for f in data_dir.rglob("*.tsv") if f.is_file()
]

with open(requirements_path, "r", encoding="utf-8") as requirements_file:
    requirements = requirements_file.read().splitlines()

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
        install_requires=requirements,
        package_data={"lexikos": data_files},
        include_package_data=True,
        platforms=["linux", "unix", "windows"],
    )
