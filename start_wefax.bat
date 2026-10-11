@echo off
title WEFAX Decoder
rem Starts the WEFAX decoder: audio from VAC Line 8 (default in wefax_decoder.py),
rem SDR Console CAT on COM4 (SDR Console uses COM3), auto-scheduler on.
rem Remove --auto below to start with the scheduler off (AUTO button on the page).
cd /d "%~dp0"
python wefax_decoder.py --cat COM4 --auto %*
echo.
echo Decoder stopped.
pause
