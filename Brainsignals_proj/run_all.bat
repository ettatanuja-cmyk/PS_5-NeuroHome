@echo off
REM One-click start: backend (port 8000) + frontend (port 5173) in two windows.
start "BCI Backend :8000" cmd /k "%~dp0run_backend.bat"
timeout /t 5 /nobreak >nul
start "BCI Frontend :5173" cmd /k "%~dp0run_frontend.bat"
echo.
echo Both servers starting...
echo Dashboard: http://localhost:5173
echo API docs:  http://127.0.0.1:8000/docs
echo.
echo Keep both black windows OPEN while using the app. Close them to stop.
pause
