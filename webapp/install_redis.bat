@echo off
echo ============================================
echo Redis Installation Helper for Windows
echo ============================================
echo.
echo Choose an option:
echo.
echo [1] Install via winget (Windows Package Manager)
echo [2] Download Memurai manually
echo [3] Use WSL2 (if available)
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Installing Memurai via winget...
    winget install Memurai.MemuraiDeveloper
    echo.
    echo After installation, start Memurai from Start Menu
    echo or run: memurai-server
)

if "%choice%"=="2" (
    echo.
    echo Opening Memurai download page...
    start https://www.memurai.com/get-memurai
    echo.
    echo Download and install Memurai Developer edition (free)
)

if "%choice%"=="3" (
    echo.
    echo Run these commands in WSL2:
    echo   sudo apt update
    echo   sudo apt install redis-server
    echo   redis-server
    echo.
    echo Then Redis will be available at localhost:6379
)

echo.
pause
