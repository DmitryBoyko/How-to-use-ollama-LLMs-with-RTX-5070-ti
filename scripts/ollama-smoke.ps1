param(
  [string]$Question = "",
  [string]$Model = "qwen2.5:14b-instruct-q4_K_M",
  [string]$HostUrl = "http://localhost:11435",
  [int]$GpuIndex = 0,
  [int]$TimeoutSec = 300,
  [int]$NumPredict = 2048,
  [int]$TargetWords = 0,
  [int]$MaxParts = 10,
  [string]$OutFile = "",
  [switch]$ChooseModel,
  [switch]$RequireGpuOnly = $true,
  [switch]$DeleteModel,
  [switch]$ShowGpu
)

$ErrorActionPreference = "Stop"

if (-not $Question) {
  $defaultQuestionB64 = "0JTQsNC5INC60YDQsNGC0LrQvtC1INC+0L/QuNGB0LDQvdC40LUg0LrQstCw0L3RgtC+0LLQvtC5INC80LXRhdCw0L3QuNC60Lgu"
  $Question = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($defaultQuestionB64))
}

function Ensure-OllamaReachable {
  param([string]$Base)
  try { Invoke-RestMethod -Method Get -Uri "$Base/api/tags" -TimeoutSec 5 | Out-Null; return $true } catch { return $false }
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
  foreach ($m in ($resp.models | Where-Object { $_ -and $_.name })) { $map[$m.name] = $m }
  return $map
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

  if ($last -and $last -ne "success") { throw "Pull did not finish successfully (last status: $last)" }
}

function Warmup-Model {
  param([string]$Base, [string]$ModelName)
  $uri = "$Base/api/generate"
  $payload = @{ model=$ModelName; prompt=" "; stream=$false; options=@{ num_predict=1 } } | ConvertTo-Json -Depth 5
  try { Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $payload -TimeoutSec 180 | Out-Null } catch {}
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
  Write-Host "Warming up $ModelName ..."
  Warmup-Model -Base $Base -ModelName $ModelName
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

function Get-GpuUtilPercent {
  param([int]$Index = 0)
  try {
    $out = [string](cmd /c "docker exec gpu-metrics nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    if (-not $out) { return $null }
    $lines = @()
    foreach ($l in ($out -split "`r?`n")) { if ($l -and $l.Trim().Length -gt 0) { $lines += $l } }
    if ($Index -ge $lines.Count) { return $null }
    $v = $lines[$Index].Trim()
    if ($v -match "^\d+$") { return [int]$v }
    return $null
  } catch { return $null }
}

function Get-WordCount {
  param([string]$Text)
  if (-not $Text) { return 0 }
  return [regex]::Matches($Text, "\S+").Count
}

function Stream-Generate {
  param([string]$Base, [string]$Model, [string]$Prompt)

  $uri = "$Base/api/generate"
  $payload = @{ model=$Model; prompt=$Prompt; stream=$true; options=@{ num_predict=$NumPredict } } | ConvertTo-Json -Depth 10

  try { Add-Type -AssemblyName System.Net.Http | Out-Null } catch {}
  $handler = New-Object System.Net.Http.HttpClientHandler
  $client  = New-Object System.Net.Http.HttpClient($handler)
  $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)

  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, $uri)
  $req.Content = New-Object System.Net.Http.StringContent($payload, [System.Text.Encoding]::UTF8, "application/json")

  $res = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
  if (-not $res.IsSuccessStatusCode) {
    $body = $res.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    throw "Ollama error $([int]$res.StatusCode): $body"
  }

  $stream = $res.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
  $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)

  $acc = New-Object System.Text.StringBuilder
  $doneReason = $null
  $lastProgress = [DateTime]::MinValue

  try {
    while (-not $reader.EndOfStream) {
      $line = $reader.ReadLine()
      if (-not $line) { continue }
      $obj = $null
      try { $obj = $line | ConvertFrom-Json } catch { continue }

      if ($ShowGpu -and (([DateTime]::UtcNow - $lastProgress).TotalMilliseconds -ge 500)) {
        $gpu = Get-GpuUtilPercent -Index $GpuIndex
        if ($null -ne $gpu) {
          Write-Progress -Id 1 -Activity ("GPU {0} utilization" -f $GpuIndex) -Status ("{0}%" -f $gpu) -PercentComplete ([Math]::Max(0, [Math]::Min(100, $gpu)))
        }
        $lastProgress = [DateTime]::UtcNow
      }

      if ($obj.response) {
        [void]$acc.Append($obj.response)
        if ($OutFile) { Add-Content -LiteralPath $OutFile -Value $obj.response -Encoding utf8 }
        Write-Host -NoNewline $obj.response
      }

      if ($obj.done -eq $true) {
        if ($obj.done_reason) { $doneReason = [string]$obj.done_reason }
        break
      }
    }
  } finally {
    if ($ShowGpu) { Write-Progress -Id 1 -Activity ("GPU {0} utilization" -f $GpuIndex) -Completed }
    $reader.Dispose()
    $stream.Dispose()
    $res.Dispose()
    $client.Dispose()
  }

  return @{ text=$acc.ToString(); done_reason=$doneReason }
}

if (-not (Ensure-OllamaReachable -Base $HostUrl)) {
  throw "Ollama not reachable at $HostUrl. Start the stack first: docker compose up -d"
}

if ($ChooseModel) {
  $Model = Select-ModelInteractive -Base $HostUrl -Current $Model
}

if ($DeleteModel) {
  Write-Host "Deleting model: $Model ..."
  Delete-Model -Base $HostUrl -ModelName $Model
  Write-Host "Deleted."
  exit 0
}

Ensure-ModelReady -Base $HostUrl -ModelName $Model -RequireGpuOnly:$RequireGpuOnly

Write-Host "Ollama smoke"
Write-Host "Host:  $HostUrl"
Write-Host "Model: $Model"
if ($ShowGpu) {
  $gpu0 = Get-GpuUtilPercent -Index $GpuIndex
  if ($null -ne $gpu0) { Write-Host ("GPU:   index {0} (util now {1}%)" -f $GpuIndex, $gpu0) } else { Write-Host ("GPU:   index {0}" -f $GpuIndex) }
} else {
  Write-Host ("GPU:   index {0} (monitoring disabled)" -f $GpuIndex)
}
Write-Host ""
Write-Host "Q:"
Write-Host $Question
Write-Host ""
Write-Host "A:"

$full = ""
$part = 0
$prompt = $Question

if ($OutFile) { try { Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue } catch {} }

while ($true) {
  if ($part -ge $MaxParts) { break }
  $part++

  $r = Stream-Generate -Base $HostUrl -Model $Model -Prompt $prompt
  $full += $r.text
  Write-Host ""

  $wc = Get-WordCount -Text $full
  if ($TargetWords -gt 0 -and $wc -ge $TargetWords) { break }
  if ($r.done_reason -ne "length") { break }
  $prompt = "Продолжай ровно с места остановки. Не повторяй уже сказанное. Продолжай текст далее."
}

