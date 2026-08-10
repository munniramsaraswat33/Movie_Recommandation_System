@echo off
call .venv\Scripts\activate
python train_model.py
streamlit run app.py
pause
