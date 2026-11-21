#!/bin/sh
# Read-only audio diagnostics for the QNX Pi. Run as root: sudo sh pi_audio_diag.sh
echo "=== audio/usb processes"; pidin -f a | grep -i -E 'audio|usb'
echo "=== slog (audio)";       slog2info | grep -i -E 'audio|snd|deva' | tail -25
echo "=== startup files that mention io-audio"; grep -rl io-audio /system/etc /etc 2>/dev/null
echo "=== /dev/snd";           ls -l /dev/snd
echo DIAG_DONE
