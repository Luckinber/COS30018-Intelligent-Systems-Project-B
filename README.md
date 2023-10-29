# COS30018-Intelligent-Systems-Project-B

## 1. Prepare Environment
Clone the repo and open a cli in the directory, ensure [Python](https://pip.pypa.io/en/stable/installation/) and [Pip](https://pip.pypa.io/en/stable/installation/) are installed then create a python virtual environment:
```bash
python -m venv venv
```
To use the venv, it must be sourced
```bash
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate
```
Then install the requirement libraries with:
```bash
pip install -r requirements.txt
```

## 2. Running the Model
Running the model in a CLI can be done but without a graphical interface, the final graph will not be displayed. To display the graph, run it in an enviroment with graphics drivers installed and a GUI. To run the model, run the following command:
```
python stock_prediction.py
```
The model will then run and output various information to the console. The final graph will be displayed in a new window.
