Param(
  [string]$CondaEnvPrefix = "C:\ProgramData\anaconda3\envs\agent"
)

$ErrorActionPreference = "Stop"

function Assert-Admin {
  $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $isAdmin) {
    throw "Please run this script in an elevated (Administrator) PowerShell. It must write into the conda env directory."
  }
}

function Backup-File([string]$Path) {
  if (-not (Test-Path $Path)) {
    throw "File not found: $Path"
  }
  $ts = Get-Date -Format "yyyyMMdd-HHmmss"
  $bak = "$Path.$ts.bak"
  Copy-Item -Force $Path $bak
  return $bak
}

Assert-Admin

$activateBat = Join-Path $CondaEnvPrefix "etc\conda\activate.d\khronos-opencl-icd-loader_activate.bat"
$helperBat   = Join-Path $CondaEnvPrefix "Library\etc\OpenCL\vendors\opencl-helper.bat"

Write-Host "Target env prefix: $CondaEnvPrefix"
Write-Host "Patching: $activateBat"
Write-Host "Patching: $helperBat"

$activateBak = Backup-File $activateBat
$helperBak   = Backup-File $helperBat

Write-Host "Backed up:"
Write-Host " - $activateBak"
Write-Host " - $helperBak"

# Helper: write temp output to %TEMP% instead of CONDA_PREFIX (ProgramData is often RO)
$helperContent = @'
@echo off
setlocal enabledelayedexpansion

REM khronos-opencl-icd-loader helper (patched by KnotLiEdge)
REM Writes temp output to a caller-provided path (or %TEMP%) to avoid permission issues

set "OUT_FILE=%~1"
if "%OUT_FILE%"=="" (
  set "OUT_FILE=%TEMP%\khronos-opencl-icd-loader-temp.txt"
)

set OCL_ICD_FILENAMES_NEW=
for %%f in (%CONDA_PREFIX%\Library\etc\OpenCL\vendors\*.icd) do (
  set /p dllname=< %%~f
  set "OCL_ICD_FILENAMES_NEW=!OCL_ICD_FILENAMES_NEW!;!dllname!"
)

if NOT "%OCL_ICD_FILENAMES_NEW%" == "" (
  echo %OCL_ICD_FILENAMES_NEW%>"%OUT_FILE%"
) else (
  type NUL > "%OUT_FILE%"
)
'@

# Activator: call helper with a temp file under %TEMP%, read it if present, then delete it quietly.
$activateContent = @'
@echo off

set "OCL_ICD_FILENAMES_CONDA_BACKUP=%OCL_ICD_FILENAMES%"

set "KNOTLIEDGE_OCL_TEMP=%TEMP%\khronos-opencl-icd-loader-%RANDOM%-%RANDOM%.txt"

REM Use /wait so OCL temp file is ready for reading
start /wait /b cmd.exe /c "%CONDA_PREFIX%\Library\etc\OpenCL\vendors\opencl-helper.bat "%KNOTLIEDGE_OCL_TEMP%""

if exist "%KNOTLIEDGE_OCL_TEMP%" (
  set /p OCL_ICD_FILENAMES_NEW=<"%KNOTLIEDGE_OCL_TEMP%"
  set "OCL_ICD_FILENAMES=%OCL_ICD_FILENAMES%%OCL_ICD_FILENAMES_NEW%"
  del /f /q "%KNOTLIEDGE_OCL_TEMP%" 1>nul 2>nul
) else (
  REM If temp file wasn't created (permission/path issue), do not spam output.
)

set OCL_ICD_FILENAMES_NEW=
set KNOTLIEDGE_OCL_TEMP=
'@

Set-Content -LiteralPath $helperBat -Value $helperContent -Encoding ASCII
Set-Content -LiteralPath $activateBat -Value $activateContent -Encoding ASCII

Write-Host "Done. Re-run: conda run -n agent python -c \"print('ok')\" (should be silent now)."

