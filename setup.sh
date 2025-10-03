
#!/bin/bash

echo "Setting up PawCare AI Application..."

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python -m venv env
fi

# Activate virtual environment
source env/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p models
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

echo "Setup complete!"
echo "Run './start.sh' to start the application"
