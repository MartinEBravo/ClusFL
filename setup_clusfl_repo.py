import os

# Define the directory structure
repo_structure = {
    "clusfl": [
        "clients",
        "server",
        "aggregations",
        "models",
        "utils",
    ],
    "examples": [],
    "tests": [],
    "docs": [],
}

# Define base files to create
base_files = {
    "README.md": "# ClusFL",
    "requirements.txt": "",
    "setup.py": """from setuptools import setup, find_packages

setup(
    name='clusfl',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    description='A Modular Framework for Clustered Federated Learning',
    author='MartinEBravo',
    license='MIT'
)
""",
    "LICENSE": "MIT License",
    ".gitignore": "__pycache__/\n*.pyc\n*.pyo\n",
    "CONTRIBUTING.md": "# Contributing Guidelines\nThank you for considering contributing to ClusFL!",
    "CODE_OF_CONDUCT.md": "# Code of Conduct\nAll contributors must adhere to the code of conduct.",
}


# Helper function to create .gitkeep files
def create_gitkeep_files(dir_path):
    with open(os.path.join(dir_path, ".gitkeep"), "w") as f:
        f.write("")


# Function to create directories
def create_directories():
    for parent, subdirs in repo_structure.items():
        os.makedirs(parent, exist_ok=True)
        if subdirs:
            for subdir in subdirs:
                os.makedirs(os.path.join(parent, subdir), exist_ok=True)
                create_gitkeep_files(os.path.join(parent, subdir))
        else:
            create_gitkeep_files(parent)


# Function to create base files
def create_base_files():
    for filename, content in base_files.items():
        with open(filename, "w") as f:
            f.write(content)


# Run setup
if __name__ == "__main__":
    create_directories()
    create_base_files()
    print("✅ ClusFL repository structure created successfully!")
