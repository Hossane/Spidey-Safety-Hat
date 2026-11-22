# Safety features: camera, mic and voice

Everything here uses only the hat's webcam (on the Pi), the laptop mic and the laptop's
speaker (the JBL). Settings are in `hat_config.json`.

## Camera (on the Pi)

| Feature | What it does | Alert |
|---|---|---|
| Tracking (`tracking.py`) | Each person or vehicle alerts once. Someone who leaves and comes back is recognised by their clothing colours. | `PERSON`, `VEHICLE` |
| Speed and direction | Every track is labelled approaching, moving away, crossing, still or steady. For anything approaching, the hat estimates the time to reach the wearer from how fast its box grows. Vehicles moving away are never announced. | `motion` and `ttc` on events |
| Closing in | A box grows fast, or the time to reach is under 3 s (`ttc_alert_s`). | `PERSON_CLOSE` / `VEHICLE` |
| Fall seen | An upright shape turns wide, low in the frame, and stays that way. | `PERSON_DOWN` |
| Lying still | A person lying on the ground and not moving for 5 s (`lying_still_s`), even if the fall wasn't seen. | `PERSON_DOWN` |
| Sudden movement (`motion.py`) | A burst of movement the object model can't label, such as a falling load or a swinging hook. Movement of tracked people and vehicles is ignored, and so is the whole view moving (the wearer turning their head). | `MOTION` |

"Person down" needs a wide box low in the frame, held for 8 frames (`down_frames`, about
2–3 s at the Pi's frame rate) so a quick crouch doesn't count. The camera is close, so a lying
body may run off one side or the bottom of the view. These don't count: a box touching the
top edge (a head too close to judge), a box touching both sides (a torso filling the view),
and a thin strip along the bottom edge (head and shoulders).

## Mic (YAMNet on the Pi)

New sound groups in `audio_detect.py`: `GLASS` (breaking glass), `GUNSHOT`, `SKID` (tires
screeching) and `MACHINERY` (chainsaw, jackhammer, power tools). `MACHINERY` alerts at most
once a minute (`cooldown_overrides`).

## Sound and sight together

| Heard | Seen | Within | Becomes |
|---|---|---|---|
| Horn, siren or tires screeching | Vehicle | 2 s | `COMBINED` (high) |
| Reversing beeps | Vehicle | 2 s | `COMBINED` (high) |
| Crash or shout | Person down | 8 s | `INJURY` (high) |
| Crash or breaking glass | Sudden movement | 3 s | `FALLING_OBJECT` (high) |

A smoke alarm plus a car is not combined, because only reversing beeps count for vehicles.

## Voice commands (laptop, offline)

`scripts/voice_commands.py` uses Windows' built-in speech recogniser with a fixed list of
phrases. The list also has decoys that sound like the commands ("hello", "kelp", "conference"),
so everyday speech matches a decoy, which is ignored, instead of being forced onto a command.

- **"Help help"** or **"I need help"**: `SOS` event, red banner on the dashboard. A plain
  "help" or "help me" doesn't count: the recogniser confuses it with "hello". "Mayday" and
  "emergency" were dropped because they misfired.
- **"Call nine one one"**: the hat replies "Say confirm to call nine one one, or say cancel".
  - **"Confirm"** within 10 s: a **simulated** 911 call. The terminal prints the message that
    would go to dispatch (worker, hat, zone, recent alerts). An `EMERGENCY_CALL` event shows
    as a red banner on the dashboard, and the speaker says it was a simulation.
  - **"Cancel"** cancels it.

**Nothing is dialled.** This is a proof of concept with no phone or telephony integration.
While the laptop is speaking, the recogniser ignores what it hears, so the hat can't confirm
its own prompt.

Test the recogniser without a mic: `python scripts/voice_commands.py --selftest`.

## Spoken alerts

Phrases are relative to the wearer. By default the camera faces backwards
(`--camera-facing back`), so the image's left is the wearer's right. Examples:

- "Two people right behind you"
- "Warning! Truck closing in behind you on your left, about 2 seconds away"
- "Warning! Person lying still right behind you"
- "Watch out! Something falling right behind you"

Use `--camera-facing front` if the camera looks forward.

## Dashboard

- **Incident timeline** (Worker tab): alerts less than 20 s apart are grouped into one
  incident, showing what the camera and mic each caught, in order.
- **Emergency banner** (both tabs): any SOS or simulated 911 call from the last 15 minutes
  shows in red at the top.
