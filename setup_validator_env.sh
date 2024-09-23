#!/bin/bash

# Update system and install required packages
sudo apt update -y
sudo apt install python3-pip -y
sudo apt install nano -y
sudo apt install libgl1 -y
sudo apt install npm -y
sudo npm install pm2@latest -g

# Install Python dependencies
pip install -e .
pip install -r requirements-validator.txt

# Setup environment variables for validator configuration
echo "NETUID=34
SUBTENSOR_NETWORK=finney
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
WALLET_NAME=default
WALLET_HOTKEY=default
VALIDATOR_AXON_PORT=8092" > validator.env

# Run Python script to download and test each diffuser model
python3 verify_diffusion_models.py