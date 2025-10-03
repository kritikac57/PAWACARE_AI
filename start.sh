
#!/bin/bash

echo "Starting PawCare AI Application..."

# Activate virtual environment
source env/bin/activate

# Start the application
echo "Starting server on http://localhost:5000"
python app.py
