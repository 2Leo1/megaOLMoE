Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Starting OLMoE++ Docker Environment..." -ForegroundColor Cyan
docker compose up -d olmoe_dev