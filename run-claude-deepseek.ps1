# 在 Cursor 里：终端切到本文件夹，执行 .\run-claude-deepseek.ps1
# 第一次请先在同一终端设置密钥（不要把 sk- 写进脚本或提交到 git）：
#   $env:DEEPSEEK_API_KEY = "sk-你的密钥"

$ErrorActionPreference = "Stop"

$key = $env:DEEPSEEK_API_KEY
if (-not $key) { $key = $env:ANTHROPIC_API_KEY }
if (-not $key) {
    Write-Host "未找到 DEEPSEEK_API_KEY（或可用的 ANTHROPIC_API_KEY）。" -ForegroundColor Yellow
    Write-Host "请在本终端先执行：" -ForegroundColor Yellow
    Write-Host '  $env:DEEPSEEK_API_KEY = "sk-..."' -ForegroundColor Cyan
    Write-Host "然后再运行： .\run-claude-deepseek.ps1" -ForegroundColor Yellow
    exit 1
}

$opus = if ($env:DEEPSEEK_OPUS_MODEL) { $env:DEEPSEEK_OPUS_MODEL } else { "deepseek-v4-pro" }
$fast = if ($env:DEEPSEEK_CHAT_MODEL) { $env:DEEPSEEK_CHAT_MODEL } else { "deepseek-v4-flash" }

$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
$env:ANTHROPIC_API_KEY = $key

# DeepSeek V4 官方模型名；可用环境变量覆盖
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = $opus
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = $fast
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $fast

Write-Host "已配置 DeepSeek V4：opus=$opus  sonnet/haiku=$fast" -ForegroundColor Green
Write-Host "ANTHROPIC_BASE_URL=$($env:ANTHROPIC_BASE_URL)"

$claudeCmd = Join-Path $env:APPDATA "npm\claude.cmd"
if (Test-Path $claudeCmd) {
    & $claudeCmd @args
} else {
    & claude @args
}
