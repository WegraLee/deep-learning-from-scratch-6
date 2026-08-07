Set-Location $PSScriptRoot

function Download-IfMissing($Url, $FileName) {
    if (-not (Test-Path $FileName)) {
        Write-Host "Downloading: $FileName"
        Invoke-WebRequest -Uri $Url -OutFile $FileName
    }
}

# === dataset files ===
Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_train.txt" "owt_train.txt"
Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_valid.txt" "owt_valid.txt"

# === binary dataset files ===
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_train.bin" "owt_train.bin"
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/owt_valid.bin" "owt_valid.bin"

# === tokenizer files ===
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/webbot/merge_rules.pkl" "merge_rules.pkl"