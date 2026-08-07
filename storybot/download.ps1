Set-Location $PSScriptRoot

function Download-IfMissing($Url, $FileName) {
    if (-not (Test-Path $FileName)) {
        Write-Host "Downloading: $FileName"
        Invoke-WebRequest -Uri $Url -OutFile $FileName
    }
}

# === dataset files ===
Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.txt" "tiny_stories_train.txt"
Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_valid.txt" "tiny_stories_valid.txt"

# === binary dataset files ===
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_train.bin" "tiny_stories_train.bin"
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/tiny_stories_valid.bin" "tiny_stories_valid.bin"

# === tokenizer files ===
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/merge_rules.pkl" "merge_rules.pkl"

# === model files ===
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_pretrain.pt" "model_pretrain.pt"
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_iter_500.pt" "model_iter_500.pt"
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_iter_5000.pt" "model_iter_5000.pt"
# Download-IfMissing "https://huggingface.co/datasets/koki0702/zero-llm-data/resolve/main/storybot/model_dpo.pt" "model_dpo.pt"