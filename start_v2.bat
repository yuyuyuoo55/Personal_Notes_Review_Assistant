@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 未找到项目 .venv，请先按 v2/frontend/README.md 配置依赖。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B -X utf8 "v2\start_local.py"
pause
