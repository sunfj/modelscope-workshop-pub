@echo off
chcp 65001 >nul
echo ============================================
echo   ??? OpenVINO Workshop
echo ============================================
echo.

:: ??????????
call ov_workshop\Scripts\activate.bat

:: ??? JupyterLab
echo ?????? JupyterLab ...
jupyter lab .

:: ????
call deactivate
pause
