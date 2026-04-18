# Test SSH automation: cd into PeekIR_Pi, activate venv, run a small move.
# Usage (from repo root):
#   .\scripts\ssh_test_move.ps1
#   .\scripts\ssh_test_move.ps1 -PiUser sean
#
# Requires: OpenSSH client (ssh) on PATH, SSH key or password auth to the Pi.

param(
    [string]$PiHost = "10.244.13.117",
    [string]$PiUser = "peekir",
    [string]$RemoteProjectDir = "PeekIR_Pi"
)

# Build remote shell command; $HOME is expanded on the Pi by bash (not by PowerShell).
$remoteCmd = 'cd $HOME/' + $RemoteProjectDir + ' && source venv/bin/activate && python move.py --right 100'
ssh "${PiUser}@${PiHost}" "bash -lc `"$remoteCmd`""
