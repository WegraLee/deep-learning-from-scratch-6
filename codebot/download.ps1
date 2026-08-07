Set-Location $PSScriptRoot

function Download-IfMissing($Url, $FileName) {
    if (-not (Test-Path $FileName)) {
        Write-Host "Downloading: $FileName"
        Invoke-WebRequest -Uri $Url -OutFile $FileName
    }
}

# === model files ===
Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/codebot/model_pretrain.pt" "model_pretrain.pt"

# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/codebot/model_sft.pt" "model_sft.pt"
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/codebot/model_grpo.pt" "model_grpo.pt"