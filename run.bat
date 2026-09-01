@echo off
setlocal
where gcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: GCC was not found in PATH.
    echo.
    pause
    exit /b 1
)
gcc -O2 -Wall -Wextra -std=c11 src\*.c -o tinygpt.exe -lm