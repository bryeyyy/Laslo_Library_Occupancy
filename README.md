# Laslo_Library_Occupancy
A Flask Web application, enabled by YOLOv5 computer vision that can determine the occupancy of Manila City Library.
NOTE: supplementary arduino code and devices must be included in this setup. 

<mark> Unzip "Arduino INO Files.zip" and run them using your Arduino IDE </mark>

For full guide refer to this document:

Virtual Environment Setup
```
python -m venv (name of your virtual environment)
```
Dependencies Setup 
```
pip install -r requirements.txt
```
Flask Setup for Windows
```
set FLASK_APP=app.py
set FLASK_ENV=development
```
Run Ipconfig in your local CMD and replace the SERVER_NAME in app.py with your local IP before running the flask app.

Flask Run
```
flask run --host=0.0.0.0
```
