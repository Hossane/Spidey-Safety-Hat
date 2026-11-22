# Safety features: camera, mic and voice

This page covers every alert the hat can raise: what triggers it, how it's worked out, how
to test it and which settings control it. All detection runs on the Pi. The laptop lends
its mic and speaker (the JBL) and handles voice commands. Settings live in
`hat_config.json`, which the Pi reads at start-up.

Every alert has a **severity**, which sets how urgent the spoken warning sounds and how
much it adds to a worker's risk score on the dashboard:

| Severity | Alerts |
|---|---|
| High | `COMBINED`, `INJURY`, `FALLING_OBJECT`, `PERSON_DOWN`, `SOS`, `EMERGENCY_CALL`, `ALARM`, `SIREN`, `IMPACT`, `FIRE`, `GUNSHOT`, `SKID` |
| Medium | `HORN`, `VOICE`, `GLASS`, `VEHICLE`, `SHARP`, `PERSON_CLOSE`, `MOTION` |
| Low | `PERSON`, `MACHINERY` |

Each kind of alert has a 3 s cooldown (`alerts.cooldown_seconds`), so one event doesn't
produce a stream of warnings. `MACHINERY` uses 60 s instead (`alerts.cooldown_overrides`).

## Camera (on the Pi)

The webcam is read through the QNX Camera service. Each frame goes through
EfficientDet-Lite0, which finds people, vehicles (car, truck, bus, motorcycle, bicycle,
train) and sharp objects (knife, scissors), with a score threshold of 0.5
(`video.threshold`). The camera runs at up to 5 frames per second (`video.fps`).

| Feature | What it does | Alert |
|---|---|---|
| Tracking (`tracking.py`) | Each person or vehicle alerts once. Someone who leaves and comes back is recognised by their clothing colours. | `PERSON`, `VEHICLE` |
| Speed and direction | Every track is labelled approaching, moving away, crossing, still or steady. For anything approaching, the hat estimates the time to reach the wearer from how fast its box grows. Vehicles moving away are never announced. | `motion` and `ttc` on events |
| Closing in | A box grows fast, or the time to reach is under 3 s (`ttc_alert_s`). | `PERSON_CLOSE` / `VEHICLE` |
| Fall seen | Someone seen standing turns wide, low in the frame, and stays that way. | `PERSON_DOWN` |
| Lying still | A person lying on the ground and not moving for 5 s (`lying_still_s`), even if the fall wasn't seen. | `PERSON_DOWN` |
| Sudden movement (`motion.py`) | A burst of movement the object model can't label, such as a falling load or a swinging hook. | `MOTION` |
| Sharp object | A knife or scissors in view. | `SHARP` |

### Tracking: why a familiar person doesn't keep beeping

Without tracking, anyone standing behind the wearer would set off an alert every few
seconds. The tracker gives each person or vehicle an ID and only announces it when it's
first seen.

- **Following someone from frame to frame:** boxes are matched by how much they overlap,
  and then by how close their centres are.
- **Recognising someone who comes back:** each track keeps a colour signature of its
  clothing (a 64-bin colour histogram), with up to 12 different looks. A person who steps
  out of view and comes back within 2 minutes (`memory_s`) is matched by colour and stays
  quiet. After that they're announced again.
- **Alerting again:** a known person only causes another alert if something happens to
  them (person down) or they close in fast.

### Speed, direction and time to reach

For each track, the hat fits a line through the last 1.5 s of its position and size.
- **Approaching:** the box is growing. How fast it grows gives the seconds until it reaches
  the wearer, and under 3 s raises a "closing in" alert.
- **Moving away:** the box is shrinking. Vehicles moving away are never announced.
- **Crossing:** the box moves sideways across the view.
- **Still:** it isn't moving.

This is a rough estimate from box size, not a measured distance.

### Person down

"Person down" fires in either of two cases:
- **Fall seen:** someone seen standing in the last 10 s (`fall_window_s`) is then wide and
  low in the frame for 2 s (`down_hold_s`). The hold means a quick crouch doesn't count.
- **Lying still:** someone wide and low in the frame, with very little movement, for 5 s
  (`lying_still_s`). This catches a person who was already on the floor when they came
  into view.

Rules that keep it reliable:
- **Glitches don't reset it:** a single odd frame, up to 1 s long (`down_grace_s`), doesn't
  reset the spell.
- **It must clearly be lying down:** most of the spell (70%, `floor_share`) must look like
  lying on the floor, and the box must be clearly wide (median height-to-width at most 0.7,
  `lying_aspect`). Someone sitting or leaning in close to the camera gives boxes that are
  only a little wide and flicker, so they don't count.
- **Re-arming:** after an alert, it can alert again once the person has been out of view
  for 10 s, or is seen standing at least 10 s later (`down_rearm_s`). That stops one person
  causing three alerts in a few seconds.

The camera is close, so a lying body may run off one side or the bottom of the view. These
don't count:
- a box touching the top edge (a head too close to judge);
- a box touching both sides (a torso filling the view);
- a thin strip along the bottom edge (head and shoulders);
- a box in the upper half of the frame, or a tiny far-away box.

**Testing:** stand fully in view first, then lie down across the frame and hold it for at
least 2 s. With the camera this close, a person on the floor sometimes looks like someone
sitting near it, so not every lie-down alerts.

### Sudden movement

`motion.py` compares each frame with the previous one at 80×60 in greyscale.
- **Ignored:** movement inside the boxes of tracked people and vehicles, since the tracker
  covers those, and frames where most of the view changes at once, which is the wearer
  turning their head.
- **Alert:** a burst of change covering at least 6% of the view (`motion.burst`), after a
  quiet second.
- **Wording:** a burst high in the frame is called "movement overhead".

Small objects are below its limit, so throwing something small doesn't trigger it. A
falling-object alert also needs a crash sound; see below.

## Mic (YAMNet on the Pi)

The laptop streams its mic (16 kHz mono) to the Pi over SSH. YAMNet scores each 0.975 s
window against 521 sound classes, every 0.5 s (`audio.hop_seconds`). A class scoring above
0.3 (`audio.threshold`) raises its group's alert:

| Group | YAMNet classes |
|---|---|
| `ALARM` | Alarm, fire alarm, smoke alarm, buzzer, car alarm, alarm clock, reversing beeps, beep |
| `SIREN` | Siren, civil defense siren, police, ambulance and fire engine sirens |
| `HORN` | Car horn, air horn / truck horn, foghorn |
| `VOICE` | Shout, yell, screaming, children shouting, crying, groan |
| `IMPACT` | Smash / crash, breaking, explosion, bang, boom, slam, thump, crack |
| `GLASS` | Glass, shatter |
| `GUNSHOT` | Gunshot, machine gun, fusillade, artillery fire |
| `SKID` | Skidding, tire squeal |
| `FIRE` | Fire, crackle |
| `MACHINERY` | Chainsaw, jackhammer, power tool, drill, sawing, sanding, lawn mower (at most once a minute) |

Ordinary construction-site noise, such as speech, hammering and engines, is recorded but
never raises an alert on its own.

**Known limits from testing:** a synthesised siren was heard as a theremin and missed, so
test with a real siren recording. Whistling can register as an alarm.

## Sound and sight together

Some alerts only fire when the mic and the camera agree within a short window. Each makes a
stronger, high-severity alert:

| Heard | Seen | Within | Becomes |
|---|---|---|---|
| Horn, siren or tires screeching | Vehicle | 2 s | `COMBINED` (high) |
| Reversing beeps | Vehicle | 2 s | `COMBINED` (high) |
| Crash or shout | Person down | 8 s | `INJURY` (high) |
| Crash or breaking glass | Sudden movement | 3 s | `FALLING_OBJECT` (high) |

A smoke alarm plus a car is not combined, because only reversing beeps count for vehicles.

**Testing:** hold up a phone showing a truck in front of the camera while playing a horn
sound near the laptop mic.

## Voice commands (laptop, offline)

`scripts/voice_commands.py` uses Windows' built-in speech recogniser (System.Speech),
which works offline, with a fixed list of phrases. The recogniser always picks the closest
phrase on its list, so the list also has about 60 decoys that sound like the commands
("hello", "kelp", "helmet", "conference", "call me"). Everyday speech matches a decoy,
which is ignored, instead of being forced onto a command.

| Say | What happens | Confidence needed |
|---|---|---|
| **"Help help"** or **"I need help"** | `SOS` event; the speaker says help was requested and a red banner shows on the dashboard | 0.85 |
| **"Call nine one one"** | The hat replies "Say confirm to call nine one one, or say cancel" | 0.8 |
| **"Confirm"** within 10 s | A **simulated** 911 call: the terminal prints the message dispatch would get (worker, hat, zone, recent alerts), an `EMERGENCY_CALL` event shows as a red banner, and the speaker says it was a simulation | 0.7 |
| **"Cancel"** | Cancels a pending call | 0.7 |

- **Plain "help" or "help me" don't count:** the recogniser scored "hello", "yellow" and
  "kelp" as "help" with over 0.9 confidence.
- **"Mayday" and "emergency" were dropped** because they misfired.
- **Tested with synthesised speech:** 28 everyday phrases trigger nothing, and every
  command is recognised (0.87–0.94). A real voice through a real mic may score
  differently.

**Nothing is dialled.** This is a proof of concept with no phone or telephony integration.
While the laptop is speaking, the recogniser ignores what it hears, so the hat can't confirm
its own prompt.

Test the recogniser without a mic: `python scripts/voice_commands.py --selftest`.

## Spoken alerts

Each alert plays a warning tone and is then spoken on the laptop's default speaker. Phrases
are relative to the wearer: they say where the hazard is, how many people there are, which
way it's moving and how soon it will arrive. By default the camera faces backwards
(`--camera-facing back`), so the image's left is the wearer's right. Examples:

- "Person behind you on your left"
- "Two people right behind you"
- "Warning! Truck closing in behind you on your left, about 2 seconds away"
- "Warning! Person lying still right behind you"
- "Watch out! Something falling right behind you"
- "Possible gunshot, take cover"

Use `--camera-facing front` if the camera looks forward. On the Pi itself, every alert also
pulses a buzzer, LED or vibration motor on GPIO pin BCM 18 (`gpio` in `hat_config.json`).

## Dashboard

- **Incident timeline** (Worker tab): alerts less than 20 s apart are grouped into one
  incident, showing what the camera and mic each caught, in order.
- **Emergency banner** (both tabs): any SOS or simulated 911 call from the last 15 minutes
  shows in red at the top.

See [TIGER_DATA.md](TIGER_DATA.md) for the rest of the dashboard.

## Tuning tools

- **Record tracker input:** `scripts/capture_tracks.py` runs on the Pi and records the
  detections the tracker sees.
- **Replay it:** `scripts/replay_tracks.py` replays a recording on the laptop with
  different settings (`--set KEY=VALUE`), so you can tune the tracker without the camera.
- **Test sounds:** the files in `samples/` can be run through YAMNet on the Pi with
  `python3 audio_detect.py samples/<file>.wav`.
