param(
    [string]$DeviceId = "",
    [string]$PackageName = "ar.edu.unicen.faa.leafseveritycalculator",
    [string]$MainActivity = "org.beeware.android.MainActivity",
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

function Invoke-Adb {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Args
    )

    $cmd = "adb -s $DeviceId $Args"
    return Invoke-Expression $cmd
}

function Wait-UiMatch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [int]$TimeoutSec = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Invoke-Adb "shell uiautomator dump /sdcard/window_dump.xml" | Out-Null
        Invoke-Adb "pull /sdcard/window_dump.xml .\\window_dump.xml" | Out-Null
        $xml = Get-Content .\\window_dump.xml -Raw
        if ($xml -match $Pattern) {
            return $xml
        }
        Start-Sleep -Milliseconds 500
    }

    throw "Timeout esperando patron UI: $Pattern"
}

function Get-LatestSavedFile {
    $files = Invoke-Adb "shell ls -t /sdcard/Download"
    if (-not $files) {
        return $null
    }

    $match = $files | Select-String -Pattern "Severity_.*pct\\.png" | Select-Object -First 1
    if ($match) {
        return $match.ToString().Trim()
    }

    return $null
}

function Resolve-DeviceId {
    param(
        [string]$RequestedId
    )

    $devices = adb devices | Select-String -Pattern "^(.+)\s+device$"
    if (-not $devices) {
        throw "No hay dispositivos ADB conectados. Ejecuta 'adb devices' y levanta el emulador."
    }

    $ids = @()
    foreach ($line in $devices) {
        $ids += ($line.Matches[0].Groups[1].Value.Trim())
    }

    if ($RequestedId -and ($ids -contains $RequestedId)) {
        return $RequestedId
    }

    if ($RequestedId -and -not ($ids -contains $RequestedId)) {
        Write-Host "Aviso: '$RequestedId' no esta disponible; usando '$($ids[0])'."
    }

    return $ids[0]
}

Write-Host "[1/8] Verificando dispositivo ADB..."
$DeviceId = Resolve-DeviceId -RequestedId $DeviceId
Write-Host "Usando dispositivo: $DeviceId"

Write-Host "[2/8] Limpiando logs y arrancando app..."
Invoke-Adb "logcat -c" | Out-Null
Invoke-Adb "shell am force-stop $PackageName" | Out-Null
Invoke-Adb "shell am start -n $PackageName/$MainActivity" | Out-Null
Wait-UiMatch -Pattern 'text="(TAKE A PHOTO|TOMAR UNA FOTO)"' -TimeoutSec 20 | Out-Null

Write-Host "[3/8] Tomando foto en camara..."
Invoke-Adb "shell input tap 270 300" | Out-Null    # TAKE A PHOTO
Wait-UiMatch -Pattern 'package="com.android.camera2"' -TimeoutSec 20 | Out-Null
Invoke-Adb "shell input tap 540 1635" | Out-Null   # Shutter
Wait-UiMatch -Pattern 'content-desc="Done"' -TimeoutSec 20 | Out-Null
Invoke-Adb "shell input tap 540 1636" | Out-Null   # Done/confirm

Write-Host "[4/8] Esperando correccion de iluminacion..."
$xml = Wait-UiMatch -Pattern 'Illumination corrected|Iluminacion corregida|Iluminación corregida' -TimeoutSec $TimeoutSeconds
if ($xml -notmatch 'text="(CALCULATE SEVERITY|CALCULAR LA SEVERIDAD)"[^>]*enabled="true"') {
    throw "La correccion termino, pero CALCULATE SEVERITY no se habilito."
}

Write-Host "[5/8] Calculando severidad..."
Invoke-Adb "shell input tap 540 1320" | Out-Null   # CALCULATE SEVERITY
$xml = Wait-UiMatch -Pattern 'Severity calculated|Severidad calculada|Severity:|Severidad:' -TimeoutSec $TimeoutSeconds
$severityMatch = [regex]::Match($xml, '(Severity|Severidad):\s*[0-9]+([\.,][0-9]{2})?%')
if ($severityMatch.Success) {
    $severity = $severityMatch.Value
} else {
    $severity = "(detectada por estado UI, sin texto numerico visible)"
}
Write-Host "Severidad detectada: $severity"

Write-Host "[6/8] Bajando a iconos y guardando..."
Invoke-Adb "shell input swipe 540 1700 540 700 300" | Out-Null
Wait-UiMatch -Pattern 'bounds="\[284,1189\]\[536,1367\]"' -TimeoutSec 20 | Out-Null
Invoke-Adb "shell input tap 410 1278" | Out-Null   # Save icon
Wait-UiMatch -Pattern 'package="com.android.documentsui"' -TimeoutSec 20 | Out-Null
Invoke-Adb "shell input tap 928 1730" | Out-Null   # SAVE button in picker

Write-Host "[7/8] Confirmando guardado..."
$xml = Wait-UiMatch -Pattern 'Image saved successfully\\.|Imagen guardada correctamente\\.|Guardada correctamente' -TimeoutSec 30
Invoke-Adb "shell input tap 894 1060" | Out-Null   # OK dialog

Write-Host "[8/8] Verificando archivo en /sdcard/Download..."
$savedFile = Get-LatestSavedFile
if (-not $savedFile) {
    throw "No se encontro un archivo Severity_*.png en /sdcard/Download."
}

Write-Host "OK - Prueba funcional completa exitosa"
Write-Host "Archivo guardado: /sdcard/Download/$savedFile"
exit 0
