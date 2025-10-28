#!/bin/bash
# Setup script for DQN Crafter on cluster

echo "=========================================="
echo "Setting up DQN Crafter Environment"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version - change if GPU available)
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install requirements
echo "Installing requirements..."
pip install gym==0.21.0
pip install stable-baselines3==2.0.0
pip install numpy
pip install matplotlib

# Install Crafter
echo "Installing Crafter environment..."
pip install git+https://github.com/danijar/crafter.git

# Optional: Install tensorboard for monitoring
pip install tensorboard

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py --timesteps 500000 --outdir ./results"
echo ""
