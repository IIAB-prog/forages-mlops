# À exécuter dans PowerShell sous le compte Windows qui utilise Docker Desktop.
# Crée une surveillance quotidienne à 02:00. Le compte doit pouvoir lancer Docker.
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runner = Join-Path $ProjectRoot "scripts\run_pipeline.bat"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$Runner`"" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName "Forages-MLOps-Pipeline" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description "Surveillance de dérive, réentraînement conditionnel, tests, promotion et redémarrage contrôlés de l'API forages-mlops." -Force
Write-Host "Tâche créée : Forages-MLOps-Pipeline, tous les jours à 02:00."
Write-Host "Docker Desktop doit être démarré. Consultez reports\automation_ledger.jsonl après chaque exécution."
