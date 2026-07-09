#Requires -Version 5.1
# setup-task.ps1 — run ONCE to register the daily refresh as a Windows Task Scheduler job.
# Equivalent to the systemd --user timer on the Ubuntu laptop.
#
# Usage: Right-click setup-task.ps1 → Run with PowerShell
#   or:  powershell -ExecutionPolicy Bypass -File ".\setup-task.ps1"
#
# The task runs at 08:00 every weekday (Mon-Fri) and is set to catch up immediately
# if the machine was off or asleep at that time (matches systemd Persistent=true).
# It runs under the current logged-in user account — no admin required.

$TASK_NAME   = "GoldForecastDailyRefresh"
$SCRIPT_DIR  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$REFRESH_PS1 = Join-Path $SCRIPT_DIR "refresh.ps1"

# Remove existing task if re-running setup
if (Get-ScheduledTask -TaskName $TASK_NAME -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false
    Write-Host "[setup] Removed existing task '$TASK_NAME'"
}

# Action: run refresh.ps1 via powershell.exe
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$REFRESH_PS1`"" `
    -WorkingDirectory $SCRIPT_DIR

# Trigger: 08:00 Mon-Fri, catch up if missed
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "08:00AM"

# Settings: run as soon as possible after a missed start; allow long run time (4 hrs)
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -MultipleInstances IgnoreNew

# Register under the current user (no admin, no password prompt for interactive sessions)
Register-ScheduledTask `
    -TaskName $TASK_NAME `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Limited `
    -Force | Out-Null

Write-Host "[setup] Task '$TASK_NAME' registered successfully."
Write-Host "        Runs: 08:00 Mon-Fri, catches up if machine was off."
Write-Host "        Script: $REFRESH_PS1"
Write-Host ""
Write-Host "To test it now:  Start-ScheduledTask -TaskName '$TASK_NAME'"
Write-Host "To remove it:    Unregister-ScheduledTask -TaskName '$TASK_NAME' -Confirm:`$false"
