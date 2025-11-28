@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 智能笔记 Agent - 通用启动器

echo ========================================================
echo        正在初始化智能笔记 Agent...
echo ========================================================
echo.

:: ---------------------------------------------------------
:: 第一阶段：寻找并锁定 Python 环境
:: ---------------------------------------------------------

:: 1. 优先检查是否存在项目自带的虚拟环境 (venv)
if exist "venv\Scripts\python.exe" (
    echo [环境] 检测到已创建的虚拟环境，正在加载...
    set "TARGET_PYTHON=venv\Scripts\python.exe"
    goto :RunApp
)

:: 2. 如果没有虚拟环境，我们需要创建一个。
::    这将解决“不同电脑路径不同”的问题，因为我们会创建一个本地副本。
echo [初次运行] 正在为您配置运行环境...

:: 尝试方法 A: 使用 Windows 官方 Python 启动器 (推荐，通常能避开 msys64 等干扰)
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [安装] 正在使用 Python Launcher 创建虚拟环境...
    py -3 -m venv venv
    goto :CheckVenv
)

:: 尝试方法 B: 使用系统默认的 python 命令
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [安装] 正在使用系统 Python 创建虚拟环境...
    python -m venv venv
    goto :CheckVenv
)

:: 如果都失败了
echo.
echo [错误] 您的电脑似乎没有安装 Python，或者环境变量未配置。
echo.
echo 请访问 https://www.python.org/downloads/ 下载并安装 Python。
echo *注意：安装时请务必勾选 "Add Python to PATH"*
echo.
pause
exit /b

:CheckVenv
:: 检查虚拟环境是否创建成功
if not exist "venv\Scripts\python.exe" (
    echo [错误] 虚拟环境创建失败。您的 Python 可能损坏或版本不兼容。
    pause
    exit /b
)
set "TARGET_PYTHON=venv\Scripts\python.exe"
echo [成功] 环境配置完成！

:: ---------------------------------------------------------
:: 第二阶段：安装依赖 (只在第一次或依赖更新时运行)
:: ---------------------------------------------------------
:InstallDeps
if exist "requirements.txt" (
    echo [配置] 正在检查并安装必要的库...
    "%TARGET_PYTHON%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple >nul
)

:: ---------------------------------------------------------
:: 第三阶段：配置 FFmpeg
:: ---------------------------------------------------------
if exist "bin" (
    set "PATH=%CD%\bin;%PATH%"
    echo [配置] 已加载本地视频处理工具
)

:: ---------------------------------------------------------
:: 第四阶段：启动程序
:: ---------------------------------------------------------
:RunApp
echo.
echo [启动] 正在启动应用程序...
echo 浏览器即将自动弹出...
echo --------------------------------------------------------

:: 使用我们要么找到、要么刚刚创建的那个 python 来运行
"%TARGET_PYTHON%" -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序意外退出。
    pause
)