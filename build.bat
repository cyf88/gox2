@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   Go2X build (PyInstaller, onedir)
echo ==========================================

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  echo         Run: python -m venv .venv  then  pip install -r requirements.txt
  goto :fail
)

tasklist /FI "IMAGENAME eq Go2X.exe" 2>nul | find /I "Go2X.exe" >nul
if not errorlevel 1 (
  echo [ERROR] Go2X.exe is running. Close it before building.
  goto :fail
)

if /I "%~1"=="clean" (
  echo [1/3] Full clean: removing build\ and dist\ ...
  if exist "build" rmdir /s /q "build"
  if exist "dist" rmdir /s /q "dist"
) else (
  echo [1/3] Incremental build ^(use "build.bat clean" for a full rebuild^)
)

echo [2/3] Packaging, please wait ...
".venv\Scripts\python.exe" -m PyInstaller Go2X.spec --noconfirm
if errorlevel 1 (
  echo [ERROR] PyInstaller failed, see the log above.
  goto :fail
)

echo [3/3] Done
if exist "dist\Go2X\Go2X.exe" (
  echo       Output: %CD%\dist\Go2X\Go2X.exe
  echo       Runtime data dir: %CD%\dist\Go2X\faces
  echo       Copy faces\ and config.json there if you need the existing face library.
) else (
  echo [WARN] dist\Go2X\Go2X.exe not found, check the log above.
)
goto :end

:fail
echo Build FAILED.

:end
pause
