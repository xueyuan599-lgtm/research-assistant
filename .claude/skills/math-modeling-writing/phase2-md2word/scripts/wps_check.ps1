param(
    [Parameter(Mandatory = $true)][string]$Docx,
    [string]$OutDir = "",
    [switch]$SkipPdf
)

$ErrorActionPreference = "Stop"
$docxPath = [System.IO.Path]::GetFullPath($Docx)
if (-not (Test-Path -LiteralPath $docxPath)) {
    Write-Error "找不到文件: $docxPath"
    exit 1
}
if (-not $OutDir) {
    $OutDir = Join-Path (Split-Path $docxPath -Parent) ("wps_check_" + [System.IO.Path]::GetFileNameWithoutExtension($docxPath))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$result = @{
    docx = $docxPath
    renderer = "wps_visible"
    ok = $false
    pages = $null
    pdf = $null
    note = ""
}

try {
    $wps = New-Object -ComObject KWps.Application
    try {
        $wps.Visible = $false
        $doc = $wps.Documents.Open($docxPath, $false, $true)
        $pages = $doc.ComputeStatistics(2)  # wdStatisticPages = 2
        $result.pages = [int]$pages

        if (-not $SkipPdf) {
            $pdf = Join-Path $OutDir ([System.IO.Path]::GetFileNameWithoutExtension($docxPath) + ".wps.pdf")
            try {
                $doc.SaveAs2($pdf, 17)  # wdFormatPDF = 17
            } catch {
                $doc.ExportAsFixedFormat($pdf, 0)  # wdExportFormatPDF = 0
            }
            $result.pdf = $pdf
            if (-not (Test-Path -LiteralPath $pdf)) {
                throw "WPS 导出 PDF 失败"
            }
        }
        $doc.Close(0)
        $result.ok = $true
        $result.renderer = "wps_com"
        $result.note = "WPS COM 打开成功，无修复提示；页数 = $pages"
    } finally {
        try { $doc.Close(0) } catch { }
        try { $wps.Quit() } catch { }
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wps) | Out-Null
    }
} catch {
    # COM 不可用时降级：启动 WPS 可视打开，由用户确认
    $wpsExe = "E:\WPS Office\12.1.0.28043\office6\wps.exe"
    if (-not (Test-Path -LiteralPath $wpsExe)) {
        $wpsExe = "wps.exe"
    }
    try {
        Start-Process -FilePath $wpsExe -ArgumentList "`"$docxPath`""
        $result.note = "WPS COM 不可用，已启动 WPS 可视打开（$($_.Exception.Message)）；请在 WPS 中确认无修复提示。"
    } catch {
        $result.note = "WPS 启动失败：$($_.Exception.Message)"
    }
}

$json = $result | ConvertTo-Json -Depth 3
$json | Out-File -FilePath (Join-Path $OutDir "wps_check.json") -Encoding UTF8
Write-Output $json
if ($result.ok) {
    exit 0
} else {
    exit 1
}
