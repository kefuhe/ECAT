<#
.SYNOPSIS
Run one ECAT MPI script on native Windows with controlled numerical threads.

.EXAMPLE
conda activate ecat
.\run_ecat_mpi_windows.ps1 -PythonScript .\test_smc.py

.EXAMPLE
.\run_ecat_mpi_windows.ps1 -PythonScript .\test_smc.py `
    -Ranks 20 -ThreadsPerRank 2 -ScriptArguments "-r"

.NOTES
PowerShell treats unique parameter prefixes as launcher parameters. In
particular, a bare -r is resolved as -Ranks and therefore cannot be used to
pass -r to the Python script. Use -ScriptArguments "-r" instead.

For multiline commands, the backtick (`) must be the final character on the
line. A backslash is not a PowerShell line-continuation character.

The launcher changes thread-related environment variables only for the
duration of this script and restores the previous process environment before
returning. It does not select an MPI vendor or change ECAT configuration.
#>

[CmdletBinding()]
param(
    [string]$PythonScript = "test_smc.py",

    [ValidateRange(1, 2147483647)]
    [int]$Ranks = 4,

    [ValidateRange(1, 2147483647)]
    [int]$ThreadsPerRank = 1,

    [string]$CaseDirectory = (Get-Location).Path,
    [string]$PythonExecutable = "python",
    [string]$MpiExecutable = "mpiexec",
    [switch]$InteractivePlots,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $CaseDirectory -PathType Container)) {
    throw "Case directory does not exist: $CaseDirectory"
}
$casePath = (Resolve-Path -LiteralPath $CaseDirectory).Path
$scriptPath = Join-Path $casePath $PythonScript
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Python script does not exist: $scriptPath"
}

if (-not (Get-Command $PythonExecutable -ErrorAction SilentlyContinue)) {
    throw "Python executable was not found: $PythonExecutable"
}
if (-not (Get-Command $MpiExecutable -ErrorAction SilentlyContinue)) {
    throw "MPI launcher was not found: $MpiExecutable"
}

$runEnvironment = [ordered]@{
    OMP_NUM_THREADS      = "$ThreadsPerRank"
    MKL_NUM_THREADS      = "$ThreadsPerRank"
    OPENBLAS_NUM_THREADS = "$ThreadsPerRank"
    MKL_DYNAMIC          = "FALSE"
    OMP_DYNAMIC          = "FALSE"
    NUMEXPR_NUM_THREADS  = "1"
    NUMBA_NUM_THREADS    = "1"
}
if (-not $InteractivePlots) {
    $runEnvironment["MPLBACKEND"] = "Agg"
}

$savedEnvironment = @{}
foreach ($name in $runEnvironment.Keys) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable(
        $name,
        "Process"
    )
    [Environment]::SetEnvironmentVariable(
        $name,
        $runEnvironment[$name],
        "Process"
    )
}

$timer = [Diagnostics.Stopwatch]::StartNew()
Push-Location -LiteralPath $casePath
try {
    Write-Host "ECAT MPI launcher (Windows)"
    Write-Host "  case directory : $casePath"
    Write-Host "  Python script  : $PythonScript"
    Write-Host "  MPI ranks      : $Ranks"
    Write-Host "  threads/rank   : $ThreadsPerRank"
    Write-Host "  thread budget  : $($Ranks * $ThreadsPerRank)"

    & $MpiExecutable -n $Ranks $PythonExecutable $scriptPath @ScriptArguments
    if ($LASTEXITCODE -ne 0) {
        throw "MPI task exited with code $LASTEXITCODE."
    }
}
finally {
    $timer.Stop()
    Pop-Location

    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable(
            $name,
            $savedEnvironment[$name],
            "Process"
        )
    }

    Write-Host "ECAT MPI launcher finished; wall time: $($timer.Elapsed)"
}
