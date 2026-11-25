@echo off
setlocal
wsl.exe --cd "%~dp0." --exec bash ./build_all.sh %*
exit /b %errorlevel%
