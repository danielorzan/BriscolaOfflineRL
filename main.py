from install_modules import check_and_install

# List of required packages
required_packages = ["pygame", "time", "torch", "os", "copy", "pillow", "IPython", "math", "numpy", "csv"]

# Check and install each package
for package in required_packages:
    check_and_install(package)


from play_pygame import game_play

game_play()