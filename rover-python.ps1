$runtime = Join-Path $PSScriptRoot '.tools/rover-env'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PATH = "$runtime\Library\bin;$runtime;" + $env:PATH
$env:TEMP = Join-Path $PSScriptRoot '.cache/tmp'
$env:TMP = $env:TEMP
& "$runtime/python.exe" -B @args
exit $LASTEXITCODE
