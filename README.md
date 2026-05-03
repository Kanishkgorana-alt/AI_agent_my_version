# Run Commands

## Reset

```powershell
.\.venv312\Scripts\python.exe reset_state.py
```

Clear generated diagnostics too:

```powershell
.\.venv312\Scripts\python.exe reset_state.py --clear-reports
```

## Report

```powershell
.\.venv312\Scripts\python.exe report_visuals.py
```

## App

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py --server.headless true
```
