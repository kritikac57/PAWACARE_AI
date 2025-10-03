@echo off
echo Starting PawCare AI Application...
echo.

REM Activate virtual environment
call env\Scripts\activate

REM Check if required packages are installed
echo Checking dependencies...
pip list | findstr Flask >nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "models" mkdir models
if not exist "static\css" mkdir static\css
if not exist "static\js" mkdir static\js

REM Start the application
echo.
echo Starting PawCare AI server...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause