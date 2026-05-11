param(
  [string]$Model = "qwen2.5:14b-instruct-q4_K_M",
  [string]$HostUrl = "http://localhost:11435",
  [int]$GpuIndex = 0,
  [switch]$RequireGpuOnly = $true,
  [string]$KeepAlive = "-1m",
  [switch]$ShowGpu
)

$ErrorActionPreference = "Stop"

$Global:OllamaSystemRu = @"
Ты — полезный ассистент.
Отвечай строго на русском языке.
Запрещено использовать любые другие алфавиты/иероглифы (например: 얽힘).
Латиница (английский) разрешена ТОЛЬКО если в русском нет точного/общепринятого аналога; тогда пиши кратко: русский термин, а затем английский в скобках.
Если вопрос неясен — задай уточняющий вопрос по-русски.
"@.Trim()

function Sanitize-RuChunk {
  param([string]$s)
  if (-not $s) { return $s }
  # Keep Cyrillic + Latin letters, digits, whitespace and punctuation; drop other scripts (e.g. CJK/Korean).
  return [regex]::Replace($s, "[^\p{IsCyrillic}A-Za-z0-9\s\p{P}]", "")
}

function Resolve-NvidiaSmi {
  $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $candidates = @(
    "$env:WINDIR\System32\nvidia-smi.exe",
    "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "$env:ProgramFiles(x86)\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
  ) | Where-Object { $_ -and (Test-Path $_) }

  if ($candidates.Count -gt 0) { return $candidates[0] }
  return $null
}

function Get-GpuUtilPercent {
  param([int]$Index = 0)
  try {
    $out = [string](cmd /c "docker exec gpu-metrics nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    if (-not $out) {
      $smi = Resolve-NvidiaSmi
      if (-not $smi) { return $null }
      $out = & $smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    }
    if (-not $out) { return $null }

    $lines = @()
    foreach ($l in ($out -split "`r?`n")) {
      if ($l -and $l.Trim().Length -gt 0) { $lines += $l }
    }
    if ($Index -ge $lines.Count) { return $null }
    $v = $lines[$Index].Trim()
    if ($v -match "^\d+$") { return [int]$v }
    return $null
  } catch {
    return $null
  }
}

function Ensure-OllamaReachable {
  param([string]$Base)
  try {
    Invoke-RestMethod -Method Get -Uri "$Base/api/tags" -TimeoutSec 5 | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Get-RecommendedModels {
  return @(
    @{ name = "qwen2.5:7b-instruct-q4_K_M";       kind = "general" },
    @{ name = "qwen2.5:14b-instruct-q4_K_M";      kind = "general" },
    @{ name = "mistral:7b-instruct-q4_K_M";       kind = "general" },
    @{ name = "mistral:7b-instruct-v0.3-q4_K_M";  kind = "general" },
    @{ name = "llama3.1:8b-instruct-q4_K_M";      kind = "general" },
    @{ name = "gemma2:9b-instruct-q4_K_M";        kind = "general" },
    @{ name = "gemma3:12b-it-q4_K_M";             kind = "general" },
    @{ name = "phi4:14b-q4_K_M";                  kind = "general" },
    @{ name = "qwen2.5-coder:7b-instruct-q4_K_M"; kind = "coder"   },
    @{ name = "qwen2.5-coder:14b-instruct-q4_K_M";kind = "coder"   }
  )
}

function Get-InstalledModelMap {
  param([string]$Base)
  $resp = Invoke-RestMethod -Method Get -Uri "$Base/api/tags" -TimeoutSec 30
  $map = @{}
  foreach ($m in ($resp.models | Where-Object { $_ -and $_.name })) {
    $map[$m.name] = $m
  }
  return $map
}

function Pull-Model {
  param([string]$Base, [string]$ModelName)

  $uri = "$Base/api/pull"
  $payload = @{ model = $ModelName; stream = $true } | ConvertTo-Json -Depth 5

  try { Add-Type -AssemblyName System.Net.Http | Out-Null } catch {}
  $handler = New-Object System.Net.Http.HttpClientHandler
  $client  = New-Object System.Net.Http.HttpClient($handler)
  $client.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan

  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
  $req.Content = New-Object System.Net.Http.StringContent($payload, [System.Text.Encoding]::UTF8, "application/json")

  $res = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
  if (-not $res.IsSuccessStatusCode) {
    $body = $res.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    throw "Pull failed $([int]$res.StatusCode): $body"
  }

  $stream = $res.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
  $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

  $last = ""
  try {
    while (-not $reader.EndOfStream) {
      $line = $reader.ReadLine()
      if (-not $line) { continue }
      $obj = $null
      try { $obj = $line | ConvertFrom-Json } catch { continue }

      $status = [string]$obj.status
      if ($status) { $last = $status }

      $pct = $null
      if ($obj.completed -and $obj.total -and $obj.total -gt 0) {
        $pct = [int][Math]::Floor(100 * ($obj.completed / $obj.total))
      }

      if ($pct -ne $null) {
        Write-Progress -Id 2 -Activity "Downloading model" -Status ("{0} ({1}%)" -f $status, $pct) -PercentComplete $pct
      } else {
        Write-Progress -Id 2 -Activity "Downloading model" -Status $status -PercentComplete 0
      }

      if ($status -eq "success") { break }
    }
  } finally {
    Write-Progress -Id 2 -Activity "Downloading model" -Completed
    $reader.Dispose()
    $stream.Dispose()
    $res.Dispose()
    $client.Dispose()
  }

  if ($last -and $last -ne "success") {
    throw "Pull did not finish successfully (last status: $last)"
  }
}

function Warmup-Model {
  param(
    [string]$Base,
    [string]$ModelName,
    [string]$KeepAliveValue = "-1m",
    [int]$TimeoutSec = 420
  )

  $uri = "$Base/api/generate"
  $payload = @{
    model      = $ModelName
    prompt     = " "
    system     = $Global:OllamaSystemRu
    stream     = $false
    keep_alive = $KeepAliveValue
    options    = @{ num_predict = 1 }
  } | ConvertTo-Json -Depth 5

  try { Add-Type -AssemblyName System.Net.Http | Out-Null } catch {}
  $handler = New-Object System.Net.Http.HttpClientHandler
  $client  = New-Object System.Net.Http.HttpClient($handler)
  $client.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan

  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
  $req.Content = New-Object System.Net.Http.StringContent($payload, [System.Text.Encoding]::UTF8, "application/json")

  $cts = New-Object System.Threading.CancellationTokenSource
  $cts.CancelAfter([TimeSpan]::FromSeconds($TimeoutSec))

  $task = $client.SendAsync($req, $cts.Token)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()

  try {
    while (-not $task.IsCompleted) {
      $elapsed = [int][Math]::Floor($sw.Elapsed.TotalSeconds)
      $pct = [int][Math]::Max(0, [Math]::Min(99, [Math]::Floor(100 * ($elapsed / [Math]::Max(1, $TimeoutSec)))))
      Write-Progress -Id 3 -Activity "Warming up $ModelName" -Status ("{0}s elapsed (keep_alive={1})" -f $elapsed, $KeepAliveValue) -PercentComplete $pct
      Start-Sleep -Seconds 1
    }

    $res = $task.GetAwaiter().GetResult()
    if (-not $res.IsSuccessStatusCode) {
      $body = $res.Content.ReadAsStringAsync().GetAwaiter().GetResult()
      throw "Warmup failed $([int]$res.StatusCode): $body"
    }
  } catch {
    # Bubble up a useful message to the REPL startup rather than silently swallowing.
    throw $_
  } finally {
    Write-Progress -Id 3 -Activity "Warming up $ModelName" -Completed
    $sw.Stop()
    $cts.Dispose()
    $req.Dispose()
    $client.Dispose()
    $handler.Dispose()
  }
}

function Test-ModelGpuOnly {
  param([string]$ModelName)
  try {
    $out = [string](cmd /c "docker compose exec -T ollama ollama ps")
    if (-not $out) { return $false }
    foreach ($l in ($out -split "`r?`n")) {
      if ($l -match [regex]::Escape($ModelName)) { return ($l -match "100% GPU") }
    }
    return $false
  } catch { return $false }
}

function Ensure-ModelReady {
  param([string]$Base, [string]$ModelName, [switch]$RequireGpuOnly)
  $installed = Get-InstalledModelMap -Base $Base
  if (-not $installed.ContainsKey($ModelName)) {
    Write-Host "Model not installed. Pulling $ModelName ..."
    Pull-Model -Base $Base -ModelName $ModelName
  }
  Write-Host "Warming up $ModelName (keep_alive=$KeepAlive; may take a few minutes under load) ..."
  Warmup-Model -Base $Base -ModelName $ModelName -KeepAliveValue $KeepAlive -TimeoutSec 420
  if ($RequireGpuOnly) {
    if (-not (Test-ModelGpuOnly -ModelName $ModelName)) {
      throw "GPU-only check failed: model is not shown as '100% GPU' in 'ollama ps'. Choose a smaller model."
    }
  }
}

function Delete-Model {
  param([string]$Base, [string]$ModelName)
  $uri = "$Base/api/delete"
  $payload = @{ model = $ModelName } | ConvertTo-Json -Depth 3
  try {
    Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $payload -TimeoutSec 60 | Out-Null
  } catch {
    $out = [string](cmd /c "docker compose exec -T ollama ollama rm $ModelName")
    if ($out -match "not found") { throw $out.Trim() }
  }
}

function Select-InstalledModelForDelete {
  param([string]$Base)
  $installed = Get-InstalledModelMap -Base $Base
  $names = @($installed.Keys | Sort-Object)
  if ($names.Count -eq 0) { Write-Host "No installed models."; return $null }
  Write-Host ""
  Write-Host "Delete model (enter number):"
  for ($i = 0; $i -lt $names.Count; $i++) { Write-Host ("{0,2}) {1}" -f ($i + 1), $names[$i]) }
  Write-Host " 0) cancel"
  while ($true) {
    $raw = Read-Host "Delete #"
    if ($raw -eq "0" -or [string]::IsNullOrWhiteSpace($raw)) { return $null }
    $n = $raw -as [int]
    if ($n -and $n -ge 1 -and $n -le $names.Count) { return $names[$n - 1] }
  }
}

function Select-ModelInteractive {
  param([string]$Base, [string]$Current)
  $rec = Get-RecommendedModels
  $installed = Get-InstalledModelMap -Base $Base
  Write-Host ""
  Write-Host "Select model (enter number). Current: $Current"
  for ($i = 0; $i -lt $rec.Count; $i++) {
    $m = $rec[$i]
    $name = $m.name
    $kind = $m.kind
    $flag = if ($installed.ContainsKey($name)) { "installed" } else { "not installed" }
    Write-Host ("{0,2}) {1} [{2}, {3}]" -f ($i + 1), $name, $kind, $flag)
  }
  Write-Host " 0) keep current"
  while ($true) {
    $raw = Read-Host "Model #"
    if ($raw -eq "0" -or [string]::IsNullOrWhiteSpace($raw)) { return $Current }
    $n = $raw -as [int]
    if ($n -and $n -ge 1 -and $n -le $rec.Count) { return $rec[$n - 1].name }
  }
}

function Stream-Generate {
  param([string]$Base, [string]$Model, [string]$Prompt)

  $uri = "$Base/api/generate"
  $payload = @{ model=$Model; prompt=$Prompt; system=$Global:OllamaSystemRu; stream=$true; keep_alive=$KeepAlive; options=@{ temperature=0.2 } } | ConvertTo-Json -Depth 10

  try { Add-Type -AssemblyName System.Net.Http | Out-Null } catch {}
  $handler = New-Object System.Net.Http.HttpClientHandler
  $client  = New-Object System.Net.Http.HttpClient($handler)
  $client.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan

  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
  $req.Content = New-Object System.Net.Http.StringContent($payload, [System.Text.Encoding]::UTF8, "application/json")

  $res = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
  if (-not $res.IsSuccessStatusCode) {
    $body = $res.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    throw "Ollama error $([int]$res.StatusCode): $body"
  }

  $stream = $res.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
  $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

  try {
    $lastUiUpdate = [DateTime]::MinValue
    while (-not $reader.EndOfStream) {
      $line = $reader.ReadLine()
      if (-not $line) { continue }
      $obj = $null
      try { $obj = $line | ConvertFrom-Json } catch { continue }

      if ($ShowGpu -and (([DateTime]::UtcNow - $lastUiUpdate).TotalMilliseconds -ge 250)) {
        $gpu = Get-GpuUtilPercent -Index $GpuIndex
        if ($null -ne $gpu) {
          Write-Progress -Id 1 -Activity ("GPU {0} utilization" -f $GpuIndex) -Status ("{0}%" -f $gpu) -PercentComplete ([Math]::Max(0, [Math]::Min(100, $gpu)))
        }
        $lastUiUpdate = [DateTime]::UtcNow
      }

      if ($obj.response) { Write-Host -NoNewline (Sanitize-RuChunk ([string]$obj.response)) }
      if ($obj.done -eq $true) { break }
    }
  } finally {
    Write-Host ""
    if ($ShowGpu) { Write-Progress -Id 1 -Activity ("GPU {0} utilization" -f $GpuIndex) -Completed }
    $reader.Dispose()
    $stream.Dispose()
    $res.Dispose()
    $client.Dispose()
  }
}

Write-Host "Ollama REPL"
Write-Host "Host:  $HostUrl"
Write-Host "Model: $Model"
if ($ShowGpu) {
  $gpu0 = Get-GpuUtilPercent -Index $GpuIndex
  if ($null -eq $gpu0) { Write-Host "GPU:   n/a" } else { Write-Host ("GPU:   index {0} (util now {1}%)" -f $GpuIndex, $gpu0) }
} else {
  Write-Host ("GPU:   index {0} (monitoring disabled)" -f $GpuIndex)
}
Write-Host ""

if (-not (Ensure-OllamaReachable -Base $HostUrl)) {
  throw "Ollama not reachable at $HostUrl. Start the stack first: docker compose up -d"
}

Write-Host "Commands: /exit, /delete, /model"

$Model = Select-ModelInteractive -Base $HostUrl -Current $Model
Ensure-ModelReady -Base $HostUrl -ModelName $Model -RequireGpuOnly:$RequireGpuOnly

while ($true) {
  Write-Host ""
  $inp = Read-Host ">"
  if ($null -eq $inp) { continue }
  $inp = $inp.Trim()
  if ($inp.Length -eq 0) { continue }

  if ($inp -eq "/exit") { break }
  if ($inp -eq "/delete") {
    $toDel = Select-InstalledModelForDelete -Base $HostUrl
    if ($toDel) {
      Write-Host "Deleting $toDel ..."
      Delete-Model -Base $HostUrl -ModelName $toDel
      Write-Host "Deleted."
    }
    continue
  }

  if ($inp -eq "/model") {
    $Model = Select-ModelInteractive -Base $HostUrl -Current $Model
    Ensure-ModelReady -Base $HostUrl -ModelName $Model -RequireGpuOnly:$RequireGpuOnly
    continue
  }

  Stream-Generate -Base $HostUrl -Model $Model -Prompt $inp
}

