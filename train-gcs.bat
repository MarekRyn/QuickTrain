@echo off

set BUCKET=marekml-data
set PROJECT=fashion
set DATA_FILE=fashion-mnist.npz
set CLOUD_CONFIG=cloudml-gpu.yaml
set REGION=us-east1
set RUNTIME=1.6


set MODULE=trainer.trainer
set PACKAGE_PATH=./trainer

FOR /F "tokens=*" %%g IN ('powershell get-date -format "{yyyyMMddHHmmss}"') do (SET STAMP=%%g)
set JOB_NAME=training_%STAMP%

set JOB_DIR=gs://%BUCKET%/%PROJECT%/models/%JOB_NAME%
set LOG_DIR=gs://%BUCKET%/%PROJECT%/logs/%JOB_NAME%
set DATA_SOURCE=gs://%BUCKET%/%PROJECT%/data/%DATA_FILE%

gcloud ml-engine jobs submit training %JOB_NAME%^
 --job-dir %JOB_DIR%^
 --runtime-version %RUNTIME%^
 --module-name %MODULE%^
 --package-path %PACKAGE_PATH%^
 --region %REGION%^
 --config=%CLOUD_CONFIG%^
 --^
 --data-source %DATA_SOURCE%^
 --log-dir %LOG_DIR%^
 --verbose 2



