# build_exe.py
# Python script to compile the application into a standalone executable using PyInstaller.
# Automatically detects and bundles CustomTkinter asset resources to prevent runtime loading issues.

import os
import sys
import customtkinter
import PyInstaller.__main__

def compile_executable():
    # Retrieve absolute path of customtkinter library to bundle its asset theme files
    ctk_path = os.path.dirname(customtkinter.__file__)
    print(f"CustomTkinter package location detected at: {ctk_path}")
    
    # Formulate PyInstaller build arguments
    # Includes --onefile to produce a single EXE, and --noconsole to suppress command line windows.
    args = [
        'main.py',
        '--onefile',
        '--name=StockAnalyzer',
        '--noconsole',
        f'--add-data={ctk_path}{os.pathsep}customtkinter',
        '--clean'
    ]
    
    print(f"Executing PyInstaller build commands: pyinstaller {' '.join(args)}")
    PyInstaller.__main__.run(args)
    print("Build processes complete. Executable can be found in the 'dist' folder.")

if __name__ == "__main__":
    compile_executable()
