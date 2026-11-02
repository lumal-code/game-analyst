# Event labels for the first pilot

Start with `media/videos/test.mp4` (30.03 seconds). Watch it with:

```sh
ffplay -autoexit -seek_interval 1 media/videos/test.mp4
```

In FFplay, `Space` pauses, `s` advances one frame while paused, left/right
seek one second with the command above, and `q` quits. A normal video player
with a visible time counter is fine too.

Edit `annotations/pilot_events.csv` in a spreadsheet or text editor. Add one
row for each time interval you want to mark. Times are seconds from the start
of the video; decimals are allowed. Do not guess an exact frame if it is hard
to tell—record an approximate interval and describe the uncertainty in `notes`.

Columns:

- `video`: `test.mp4` for the pilot.
- `target`: a stable human-readable description of the *same* chosen player,
  for example `red jersey 14`. If the number is unreadable, describe the
  player's starting location and uniform. Do not switch players mid-clip.
- `kind`: `close` when the visible ball comes close to that player,
  `not_close` for a clear negative interval, or `unknown` when the ball or
  target cannot be located well enough to decide.
- `start_s`, `end_s`: interval start and end in seconds, with `0 <= start_s <
  end_s <= 30.03` for `test.mp4`.
- `notes`: why the interval was marked; mention occlusion, camera movement,
  or another player nearby when relevant.

For the pilot, "close" includes defending and does **not** require a touch.
Mark when the ball is visibly near the player's body or immediate playing
space. The exact automatic distance threshold will be tuned to these human
labels later. If the ball is hidden and its position is uncertain, use
`unknown` rather than inferring a position.

The first pass is complete when the CSV contains at least one `close` and one
`not_close` interval for one selected player. If the 30-second pilot does not
contain both, choose another 30-second segment and record its source video and
start time in `notes`. Keep this CSV as a record of what the app should find.
