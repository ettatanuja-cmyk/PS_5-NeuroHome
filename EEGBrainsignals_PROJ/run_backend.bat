@echo off
cd /d "%~dp0backend"
echo Checking scipy native extensions (App Control often blocks Store-Python DLLs)...
python -c "import scipy.io, scipy.signal; print('scipy OK')" 2>nul
if errorlevel 1 (
  echo.
  echo WARNING: scipy failed to import. If you see "Application Control policy
  echo has blocked this file" ^(e.g. _lsap, _mio_utils, pyduccfft^), do this:
  echo   1. Install Python from python.org ^(NOT the Microsoft Store build^)
  echo   2. py -3.13 -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r requirements.txt
  echo   3. Or: turn off Smart App Control / add a Defender exclusion for site-packages
  echo.
)
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
