# Repairing D: (2026-10-21)

**What happened.** At 11:57 Windows logged *Delayed Write Failed* on
`D:\nfl-ml-data\ask\traces\pilot\results.jsonl` (event 50, exfat), and the volume now reports
HealthStatus *Warning*, OperationalStatus *Full Repair Needed*. The physical disk (Seagate One
Touch HDD, 4 TB, USB) reports *Healthy*: the filesystem (exFAT) is inconsistent, most likely because
the drive dropped off the USB bus for a moment (cable, hub, or USB power saving), not because the
disk is failing. About 10 trace answers were lost; everything else of the trainer was copied to
F:\nfl-ask-trainer-data and byte-checked.

**What else lives on D:** the NFL data root (`D:\nfl-ml-data`), WSL Ubuntu's disk
(`D:\WSL\Ubuntu\ext4.vhdx`: the training environment) and Docker Desktop's disk
(`D:\Docker-Image\...`: all containers, the NFL Neo4j graphs included). WSL Ubuntu already returns
I/O errors, and `docker stop` no longer takes effect, so training and the Cypher checks wait for the
repair. Trace generation runs on Windows from F: and keeps going.

## Steps (Rishi, at the PC; ~30-90 min, mostly waiting)

1. Quit Docker Desktop (tray icon -> Quit Docker Desktop).
2. In any terminal: `wsl --shutdown` (stops Ubuntu, docker-desktop and EXL3-Runtime; their disks are on D: and F:).
3. Close Explorer windows or programs that have D: open.
4. In an **admin** terminal (Win+X -> Terminal (Admin)): `chkdsk D: /f /x`
   (`/f` fixes the exFAT structures, `/x` takes D: offline first). Keep its last lines: how many
   files it fixed or truncated.
5. Check: `Get-Volume -DriveLetter D | Format-List HealthStatus,OperationalStatus` should say
   Healthy / OK.
6. Then tell Claude: it checks WSL Ubuntu's own ext4 filesystem (inside the vhdx), with a read-only
   `e2fsck -n` first; a fix needs `wsl --mount --vhd D:\WSL\Ubuntu\ext4.vhdx --bare` (admin) and
   `e2fsck -f` from the EXL3-Runtime distro, then `wsl --unmount`. Then Docker Desktop is started
   again and the containers checked.
7. Worth doing once: plug the drive straight into a rear USB port with its own cable; the usual
   cause of a *Delayed Write Failed* on a USB disk is the connection, not the disk.

The trainer's data stays on F: (`configs/paths.yaml`) until D: is healthy again; then it is copied
back and the paths reverted.
