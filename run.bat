@echo off
setlocal

echo ==============================
echo          TinyGPT
echo ==============================
echo.

where gcc >nul 2>nul

if %errorlevel% neq 0 (
    echo ERROR: GCC was not found in PATH.
    echo.
    pause
    exit /b 1
)

echo Building...
echo.

gcc -O2 -Wall -Wextra -std=c11 src\*.c -o tinygpt.exe -lm

echo.
pause