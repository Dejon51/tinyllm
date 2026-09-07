@echo off
setlocal
where gcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: GCC was not found in PATH.
    echo.
    pause
    exit /b 1
)
gcc -Wall -Wextra -std=c11 src\*.c -o tinygpt.exe -flto -O3 -march=native -lm