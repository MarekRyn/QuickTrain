@echo off

set DATA_FILE=fashion-mnist.npz
set MODULE=trainer.trainer

FOR /F "tokens=*" %%g IN ('powershell get-date -format "{yyyyMMddHHmmss}"') do (SET STAMP=%%g)
set JOB_NAME=training_%STAMP%
set JOB_DIR=.\models\%JOB_NAME%
set LOG_DIR=.\logs\%JOB_NAME%
set DATA_SOURCE=.\data\%DATA_FILE%

md %JOB_DIR%
md %LOG_DIR%

python -m %MODULE%^
 --job-dir %JOB_DIR%^
 --data-source %DATA_SOURCE%^
 --log-dir %LOG_DIR%^
 --verbose 1

