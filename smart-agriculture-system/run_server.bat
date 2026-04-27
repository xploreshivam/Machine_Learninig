@echo off
echo Starting Smart Agriculture System...
cd /d "d:\collage\semester4\machine learning\smart agricultur analytic system\smart-agriculture-system"

:: Activate virtual environment if it exists
if exist venv\Scripts\activate (
    echo Activating Virtual Environment...
    call venv\Scripts\activate
)

:: Run the application
echo Launching Flask Server on http://127.0.0.1:5001
python app.py

pause
