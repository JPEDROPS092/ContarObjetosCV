#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip
pip install wheel setuptools

# Install PyTorch first
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt

echo "Installation complete! Activate the virtual environment with: source venv/bin/activate" 