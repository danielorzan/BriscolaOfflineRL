import subprocess

def install_package(package):
    subprocess.check_call(["pip", "install", package])

def check_and_install(package):
    try:
        import importlib
        importlib.import_module(package)
    except ImportError:
        print(f"{package} is not installed. Installing now...")
        install_package(package)
        print(f"{package} has been installed.")
