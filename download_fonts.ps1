$fonts = @{
    "DejaVuSans.ttf" = "https://github.com/prawnpdf/prawn/raw/master/data/fonts/DejaVuSans.ttf"
    "DejaVuSans-Bold.ttf" = "https://github.com/prawnpdf/prawn/raw/master/data/fonts/DejaVuSans-Bold.ttf"
}

$targetDir = "backend/assets/fonts"
if (!(Test-Path $targetDir)) {
    New-Item -ItemType Directory -Force -Path $targetDir
}

foreach ($name in $fonts.Keys) {
    $url = $fonts[$name]
    Write-Host "Downloading $name..."
    try {
        Invoke-WebRequest -Uri $url -OutFile "$targetDir/$name" -ErrorAction Stop
        Write-Host "Done: $name"
    } catch {
        Write-Host "Error $name: $_"
    }
}
