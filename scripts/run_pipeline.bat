@echo off
setlocal
REM A adapter si l'environnement virtuel porte un autre nom.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo ERREUR : environnement .venv absent. Creez-le et installez requirements.txt.
  exit /b 2
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m src.pipeline --auto-promote
exit /b %ERRORLEVEL%
