param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    if (Test-Path ".\build\CellGPS") {
        Remove-Item ".\build\CellGPS" -Recurse -Force
    }

    if (Test-Path ".\dist\CellGPS.exe") {
        Remove-Item ".\dist\CellGPS.exe" -Force
    }

    if (Test-Path ".\dist\error.log") {
        Remove-Item ".\dist\error.log" -Force
    }

    & $PythonExe -m PyInstaller --clean --noconfirm ".\CellGPS.spec"
}
finally {
    Pop-Location
}
