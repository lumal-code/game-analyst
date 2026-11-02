# Game Analyst: work queue and release plan

## The app we are building

Input: a recorded soccer game and a player selected by clicking them in the
video. Output: a reviewable timeline and downloadable clips for every period
when the ball comes close to that player, including defending. A touch is not
required. The first release runs locally and processes video after recording
on a CPU. It may ask the user to reselect the player if tracking loses them.

## How to use this document

Work on the **NOW** card below. When its pass check is true, mark its checkbox
`[x]`, put a short note in `Progress log`, and promote the next numbered card
to **NOW**. Do not skip a pass check to move to a later feature. "You" means
the human who watches/labels video; "Codex" means implementation work to ask
the coding agent to perform. A blocked task should name the exact missing
input in the progress log; continue with an independent task if possible.

**NOW: T01 — Label one close-ball moment in `test.mp4`.**

1. From the project root, run
   `ffplay -autoexit -seek_interval 1 media/videos/test.mp4`. This is a
   30-second video. Watch it once without pausing. `Space` pauses; `s` steps
   one frame; left/right seek one second; `q` quits.
2. Choose one player who has a visible ball-near-player moment. Describe that
   player by jersey/team and starting position. Keep the same player for the
   whole pilot. If no player has both close and not-close moments, use a
   30-second segment from another source video and note its source/start time.
3. Watch again, pausing as needed. Fill
   [`annotations/pilot_events.csv`](annotations/pilot_events.csv) with `close`,
   `not_close`, and, where needed, `unknown` intervals. The exact column rules
   are in [`annotations/README.md`](annotations/README.md).
4. Open the CSV and check that every row refers to the same player, start is
   before end, and at least one interval is `close` and one is `not_close`.

When done, tell Codex **"T01 done; implement T02"**. It can check the CSV and
start the next card without you having to restate the plan.

**Owner:** You. **Input:** `media/videos/test.mp4`. **Output:**
`annotations/pilot_events.csv`. **Pass check:** one selected player, at least
one positive and one negative interval, with usable timestamps. **Next:** T02.

## What exists already

- [x] Two games with 50 hand-labeled, widely spaced frames.
- [x] A YOLOv7-tiny CPU train/evaluate pipeline in
  `training/player_detector/train.py` and `src/training/testing.py`.
- [x] On Game 2 at 960 input: 300/352 players and 2/15 balls found at
  confidence 0.01; 0/14 referees found.
- [x] A notebook with an initial ByteTrack player experiment.

These frame labels measure object detection only. They cannot measure tracking
or close-ball events. Game 2 has been used for model selection and is not an
untouched final test game.

## Milestone A — Prove the clip-making logic with human labels

- [ ] **T01 (You): Mark pilot event intervals.** Complete the **NOW** card.
- [ ] **T02 (Codex, then You): Make pilot frames and position labels.** Ask
  Codex to add a repeatable frame extractor and timestamp manifest. A direct
  first-pass command is:

  ```sh
  mkdir -p media/pilot/test/frames
  ffmpeg -y -i media/videos/test.mp4 -vf fps=5 -q:v 2 -start_number 0 media/pilot/test/frames/frame_%06d.jpg
  ```

  This gives about 150 frames at 0.2-second spacing. In the same annotation
  tool used for Games 1 and 2, mark only the selected player's box and the
  visible ball's box on each sampled frame. Leave the ball absent when not
  visible; do not invent its location. Export labels to `media/pilot/test/`.
  **Pass check:** every sampled frame has a timestamp and target-player box
  or an explicit `target_missing` mark; visible balls have a box. Codex can
  inspect the export and convert it to a standard track table.
- [ ] **T03 (Codex): Validate the pilot labels.** Add an importer/validator for
  the annotation export. It must check timestamp order, image dimensions,
  missing target/ball states, duplicate boxes, and out-of-frame coordinates.
  Save a simple overlay video for human review. **Pass check:** You can watch
  the overlay and confirm the target and ball positions match the video.
- [ ] **T04 (Codex): Implement proximity on hand-marked tracks.** Input is the
  validated target and ball positions; output is time intervals plus a score.
  Start with distance from ball center to target box normalized by target
  height, a short persistence window, and different enter/exit thresholds.
  Keep these values configurable. `unknown` must remain unknown when the ball
  is hidden. **Pass check:** on the 30-second pilot, predicted intervals overlap
  the hand-labeled `close` intervals and avoid its clear negatives. Review
  misses and false alarms before changing thresholds.
- [ ] **T05 (Codex): Export clips from hand-marked events.** Add a command
  that reads a video plus intervals and writes MP4 clips and a CSV/JSON
  manifest. Start with 3 seconds before and 4 seconds after each interval,
  merge overlaps, and clamp to video bounds. **Pass check:** You can play each
  clip with audio and verify start/end times; rerunning does not create
  ambiguous duplicate names. FFmpeg can do the media cutting.

**Milestone A is done when a selected player and hand-marked ball positions
produce useful clips from a real 30-second video.** This proves the app's
purpose before model errors are introduced.

## Milestone B — Make the ball detectable

- [ ] **T06 (Codex): Establish a video ball baseline.** Run the current 960
  checkpoint on the pilot frames. Save every ball candidate, score, box,
  timestamp, and an overlay video. Compute visible-ball frame recall and
  false candidates per video minute against T02 labels. **Pass check:** a
  reproducible report shows exactly which frames fail.
- [ ] **T07 (Codex): Test tiled inference.** Split the 1920x1080 frame into
  overlapping high-resolution regions, infer on each, remap boxes to full
  coordinates, and merge duplicates. Compare with full-frame 960 and 1280 on
  the same frames, recording ball recall, false candidates, and CPU seconds
  per video minute. Use development clips, leaving the final test game
  untouched. **Pass check:** choose the best measured method; do not
  assume tiling wins.
- [ ] **T08 (You): Add varied labeled clips.** Choose 10-20 short continuous
  plays from different game situations: close and far balls, passes, defending,
  aerial balls, clutter, camera motion, and visible-ball negatives. Label ball
  positions at the sampling rate used for development, and mark close
  intervals for a selected target. Aim for a few hundred *distinct* visible
  ball examples, not adjacent copies of the same frame. Keep source video and
  timestamps. **Pass check:** counts and coverage by game/condition are
  recorded; one whole new game is reserved for final testing if available.
- [ ] **T09 (Codex): Train and compare ball models.** Prepare the expanded
  labels, train with the winning image strategy from T07, and compare the
  current three-class model with a ball-focused model if needed. Use the same
  development clips and threshold procedure for both. **Pass check:** a table
  of ball recall, false candidates, and CPU time on unseen development clips;
  save the chosen weights and exact training settings.

**Milestone B is done when ball observations recur often enough to support a
track.** A provisional gate is at least 80% recall on visible-ball frames
with a manageable false-candidate rate; adjust the gate based on real clip
review, not a single mAP number. If recall remains low, collect/label more
diverse balls before adding more epochs.

## Milestone C — Track and recognize close-ball moments

- [ ] **T10 (Codex): Turn player tracking into a module.** Move the notebook's
  tracker experiment into code that accepts timestamped detections and returns
  track IDs. Add a test clip with overlaps and players off the field. **Pass
  check:** an overlay video shows IDs and known ID switches.
- [ ] **T11 (Codex + You): Select and follow the target.** Start with a click
  on a video frame; map it to a player track. Permit reselection after a loss
  or camera cut. Add an optional manually drawn field polygon to suppress
  off-field players. **Pass check:** the selected target remains correct in
  the pilot; losses are surfaced rather than silently switching players.
- [ ] **T12 (Codex): Build a ball track.** Associate ball candidates over
  time, reject implausible jumps, and bridge only short supported gaps. Record
  `unknown` for long occlusion or no detections. **Pass check:** an overlay and
  gap report agree with labeled ball trajectories on development clips.
- [ ] **T13 (Codex): Connect live tracks to T04/T05.** Feed predicted tracks
  to the same proximity and clip code used with human labels. Keep target
  selection, distance threshold, persistence, and clip padding configurable.
  **Pass check:** one command processes a continuous clip end to end and
  writes a playable clip manifest.
- [ ] **T14 (Codex + You): Measure event quality.** Compare predicted clips
  with human `close` intervals. Report event recall, false clips per 10 video
  minutes, start-time error, and unknown time. Review every miss and false
  clip; fix the largest error source first. Run a final check on a previously
  untouched game. **Pass check:** performance is acceptable to you on unseen
  video. Provisional targets are at least 90% event recall and no more than
  one false clip per 10 minutes; revise them with actual review costs.

**Milestone C is done when a video, click, and processing command produce
reviewable close-ball clips without hand-marked tracks.**

## Milestone D — Make it a usable local app

- [ ] **T15 (Codex): App skeleton.** Add a local interface and job API around
  the working command. Choose an existing video, scrub, click the target, and
  start processing. **Pass check:** a new user can start a pilot without a
  terminal or notebook.
- [ ] **T16 (Codex): Long-job controls.** Show stage, processed time, elapsed
  time, and a realistic estimate; allow cancel and restart. Persist partial
  work and errors. **Pass check:** interrupt a long run and recover without
  corrupting prior clips.
- [ ] **T17 (Codex): Review timeline.** Show candidate moments with thumbnail,
  time, score, and playback. Allow keep/reject, manual time adjustment, and
  target reselection. Clearly distinguish ball-unknown periods. **Pass check:**
  a reviewer can correct results without editing files.
- [ ] **T18 (Codex): Export.** Download selected MP4 clips, a combined reel
  if useful, and a timestamp manifest. Preserve original audio and source
  references. **Pass check:** exports play in a standard video player and
  match the approved timeline.

**Milestone D is done when the full workflow is usable through the interface.**

## Milestone E — Polish and release

- [ ] **T19 (Codex): Reliability.** Test no ball, no chosen player, camera
  cuts, missing/corrupt video, variable frame rate, different aspect ratios,
  empty results, duplicate events, and canceled jobs. Add tests for event
  boundaries and clip timing. **Pass check:** failures have clear messages and
  do not destroy saved work.
- [ ] **T20 (Codex): Full-game benchmark.** Process a complete half and then a
  full game on the target CPU. Record wall time, peak memory, disk use, and
  output counts. **Pass check:** the app shows a realistic runtime estimate
  and does not exhaust memory or fill disk unexpectedly.
- [ ] **T21 (Codex): Reproducible install.** Pin model/code versions, automate
  dependency and weight setup, document local data locations and cleanup,
  and add a one-command launch path. **Pass check:** a clean machine can run
  the pilot from the instructions.
- [ ] **T22 (You + Codex): Usability trial.** Ask someone unfamiliar with the
  code to select a player, process a video, review clips, and export them.
  Record every confusion and failed action; fix the blockers and repeat.
  **Pass check:** they finish without developer help.
- [ ] **T23 (You + Codex): Release check.** Run the held-out game once, compare
  against the agreed event targets, verify clips, document limitations, and
  package the local app. **Pass check:** you would trust it to find useful
  candidate moments in a new game, with a review step for uncertainty.

## Progress log

| Date | Task | Evidence / decision | Next |
| --- | --- | --- | --- |
| 2026-10-30 | Plan created | `test.mp4` is 30.03 s, 1920x1080, 30 fps; first pilot source. | T01 |
