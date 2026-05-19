@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%adb_e2e_leafseverity.ps1"

if not exist "%PS1%" (
    echo ERROR: No se encuentra %PS1%
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo La prueba funcional termino con error. Codigo: %EXIT_CODE%
)

exit /b %EXIT_CODE%
