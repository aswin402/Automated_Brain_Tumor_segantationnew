@echo off
echo.
echo Brain Tumor Classification Dashboard - Quick Start
echo ======================================================

:: Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo X Node.js not found. Please install Node.js v16+
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo Y Node.js %NODE_VERSION%

:: Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo X Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Y Python %PYTHON_VERSION%

:: Setup Backend
echo.
echo Setting up backend...
cd backend

if not exist "node_modules" (
    echo Installing backend dependencies...
    call npm install
)

if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
    echo Y .env created (using defaults)
)
echo Y Backend ready

:: Setup Frontend
echo.
echo Setting up frontend...
cd ..\frontend

if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
    echo Y .env created (using defaults)
)
echo Y Frontend ready

cd ..

echo.
echo ======================================================
echo Z Setup complete!
echo.
echo To start the dashboard:
echo.
echo Terminal 1 - Backend:
echo   cd backend
echo   npm start
echo.
echo Terminal 2 - Frontend:
echo   cd frontend
echo   npm run dev
echo.
echo Then open: http://localhost:5173
echo.
echo Make sure MongoDB is running!
echo ======================================================
pause
