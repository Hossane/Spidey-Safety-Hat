#!/bin/sh
# Root-only, reversible experiment: start a temporary second io-snd bound to the USB mic
# (bus 0, dev 1) with different options, try a 1 s 16 kHz mono recording, then stop that
# io-snd only. Uses its own conf files in /tmp; the system io_snd.conf is not touched.
# Usage: sudo sh pi_mic_experiment.sh
OUT=/tmp/mic_exp; mkdir -p $OUT

snd_pids() { pidin -f an 2>/dev/null | awk '$2 ~ /io-snd$/ {print $1}' | sort; }

try() {
  name=$1; shift
  conf=$OUT/$name.conf
  { echo "[global]"; echo "verbosity=0x3"; echo "[ctrl]"; echo "name=usb"
    echo "busno=0"; echo "devno=1"; for o in "$@"; do echo "$o"; done; } > $conf
  echo; echo "########## $name: $*"
  before=$(snd_pids)
  io-snd -c $conf &
  sleep 3
  new=$(snd_pids | grep -v -x -F "$before")
  echo "new io-snd pid(s): ${new:-none}"
  slog2info | grep io_snd | tail -6 | cut -c 1-200
  ls /dev/snd
  rm -f $OUT/$name.wav
  waverec -c 1 -r 16000 -t 1 $OUT/$name.wav 2>&1 | tail -3
  [ -s $OUT/$name.wav ] && echo "RESULT $name: recorded $(ls -l $OUT/$name.wav | awk '{print $5}') bytes" || echo "RESULT $name: no recording"
  for p in $new; do kill $p; done
  sleep 1
}

try A_no_splitter input_splitter_enable=0
try B_no_splitter_frag64 input_splitter_enable=0 max_frag_ms=64
try C_no_swmixer input_splitter_enable=0 disable_sw_mixer=1
try D_frag8 max_frag_ms=8
echo; echo "remaining io-snd: $(snd_pids | tr '\n' ' ')"
chmod 644 $OUT/*.wav 2>/dev/null
echo EXP_DONE
