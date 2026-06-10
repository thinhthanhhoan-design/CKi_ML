$ErrorActionPreference = "SilentlyContinue"
$logPath = "E:\Project2\outputs\day4_cl_v9\live_monitor.log"
while ($true) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $proc = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "Cl\\stage_01_core\.py" } | Select-Object -First 1
  $os = Get-CimInstance Win32_OperatingSystem
  if (-not $proc) {
    Add-Content -Path $logPath -Value ("[$ts] PROCESS_NOT_FOUND")
    break
  }
  $p = Get-Process -Id $proc.ProcessId
  $usedPct = (((($os.TotalVisibleMemorySize/1MB)-($os.FreePhysicalMemory/1MB))/($os.TotalVisibleMemorySize/1MB))*100)
  $line = "[{0}] PID={1} WS_GB={2:N2} PRIV_GB={3:N2} CPU_TOTAL_SEC={4:N1} THREADS={5} SYS_FREE_GB={6:N2} SYS_USED_PCT={7:N1}" -f $ts,$p.Id,($p.WorkingSet64/1GB),($p.PrivateMemorySize64/1GB),$p.CPU,$p.Threads.Count,($os.FreePhysicalMemory/1MB),$usedPct
  Add-Content -Path $logPath -Value $line
  Start-Sleep -Seconds 30
}
