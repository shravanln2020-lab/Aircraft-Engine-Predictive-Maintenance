@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  run_tests.bat — AEPM Automated Test Suite Runner (Windows)
REM  Aircraft Engine Predictive Maintenance using Machine Learning
REM
REM  Runs the complete 14-point unit and integration test suite.
REM  Exit code: 0 = all tests passed | 1 = one or more failures.
REM ─────────────────────────────────────────────────────────────────────────────

echo ======================================================
echo   AEPM Automated Test Suite
echo   Aircraft Engine Predictive Maintenance
echo ======================================================

REM ── Activate virtual environment ──────────────────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] No .venv found. Using system Python.
)

REM ── Run test suite ────────────────────────────────────────────────────────────
echo [INFO] Running AEPM test suite via run_tests.py ...
.venv\Scripts\python.exe run_tests.py

REM ── Capture and relay exit code ───────────────────────────────────────────────
set EXIT_CODE=%ERRORLEVEL%

echo ======================================================
if %EXIT_CODE% EQU 0 (
    echo   [SUCCESS] All tests passed.
) else (
    echo   [FAILED]  One or more tests failed. Exit code: %EXIT_CODE%
)
echo ======================================================

pause
exit /b %EXIT_CODE%
