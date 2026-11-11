# Keeps the GPU single-tenant while the CIFAR job runs: any OTHER python process that shows up as a GPU compute app is
# SUSPENDED (never killed) and RESUMED when results\CIFAR_DONE appears. Does not touch non-python processes.
param([string]$DoneFile = "C:\Users\rohil\Downloads\gap-projects\05-diffusion-lab\results\CIFAR_DONE")
Add-Type -Namespace Nt -Name Api -MemberDefinition @'
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
'@
$log = "C:\Users\rohil\Downloads\gap-projects\05-diffusion-lab\results\gpu_guard.log"
function Log($m) { "$(Get-Date -Format s) $m" | Out-File -Append -Encoding ascii $log }
$suspended = @{}
Log "guard started"
while (-not (Test-Path $DoneFile)) {
  $gpuPids = (nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>$null) | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+$' }
  foreach ($procId in $gpuPids) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
    if (-not $p -or $p.Name -notmatch '^python') { continue }
    if ($p.CommandLine -match 'diffusionlab|scripts\run_cifar|05-diffusion-lab') { continue }   # this job (and its child)
    if ($suspended.ContainsKey([int]$procId)) { continue }
    $h = (Get-Process -Id $procId -ErrorAction SilentlyContinue).Handle
    if ($h) { [Nt.Api]::NtSuspendProcess($h) | Out-Null; $suspended[[int]$procId] = $p.CommandLine; Log "SUSPENDED pid $procId : $($p.CommandLine.Substring(0,[Math]::Min(120,$p.CommandLine.Length)))" }
  }
  Start-Sleep 15
}
foreach ($procId in @($suspended.Keys)) {
  $h = (Get-Process -Id $procId -ErrorAction SilentlyContinue).Handle
  if ($h) { [Nt.Api]::NtResumeProcess($h) | Out-Null; Log "RESUMED pid $procId" } else { Log "pid $procId already gone" }
}
Log "guard finished"
