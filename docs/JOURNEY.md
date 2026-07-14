# From One Mario to a Wall of Them: How This Project Works

This is the story of the project in the order it was built, plus the reinforcement learning concepts behind it. If you read this top to bottom you should be able to explain every piece of the system to someone else.

## Part 1: One Mario, one level

Everything starts with a real NES emulator running inside Python (`nes-py`), wrapped by `gym-super-mario-bros` so that World 1-1 behaves like a game the code can play: it can press buttons, read the screen, and ask questions like "where is Mario" and "is he dead".

The AI does not see the game the way we do. Before every decision, the screen gets simplified:

- Color is dropped and the image is shrunk to 84x84 grayscale. A goomba is still a goomba in low-res gray, and the network trains much faster on small inputs.
- The AI sees the last 4 frames stacked together, not one. A single frame can't show movement; four frames show velocity and direction.
- It only decides every 4th frame and holds the button in between (frame skip). No human makes 60 decisions per second either.
- Instead of all 256 possible button combos, it picks from 7 useful ones (run right, jump right, and so on).

So the "player" is a small neural network: input is a 4x84x84 stack of gray pixels, output is one of 7 button choices. That's it.

## Part 2: The wall

One emulator is cheap, so we run 10 of them in parallel (one per CPU core), each in its own process. All 10 Marios share the same single brain: the network makes 10 decisions at once, one per screen. The tiled display just collects each emulator's frame and arranges them in a grid.

Why 10 and not 30? Emulators are CPU-bound. Ten cores means ten emulators running truly in parallel; a 30th emulator doesn't add speed, it just makes the other 29 take turns. More instances = more spectacle, not more learning.

## Part 3: The ghost

TrackMania-style improvement needs a benchmark you can see. Ours is a translucent Mario running at world-record pace inside every tile, with a green/red border showing who's ahead.

The ghost is a recording: for every frame of a near-perfect run, the x and y position Mario had. First lesson learned: we tried replaying the actual world-record TAS input file through our emulator, and it desyncs. Frame-perfect inputs recorded on one emulator drift on another because they handle lag frames slightly differently. So instead, the ghost was **grown in our own emulator**: hold max run speed the whole level, and let a search algorithm find the fastest surviving jump timing for every pipe, pit, and staircase, backtracking whenever a jump choice leads into a dead end. Result: a 19.60 second run with 17 jumps that behaves like a world record and replays perfectly.

The AI never sees the ghost. It exists only in the display, for us. The AI's motivation comes from the reward.

## Part 4: How the AI actually learns

This is the part everyone gets wrong at first, because it is nothing like training on a dataset.

### There is no dataset. The data is a river, not a lake.

In supervised learning you store a lake of examples and revisit it for weeks. Reinforcement learning here works like a river: experience flows through the network, changes it a little, and is gone. The AI has already processed millions of frames of gameplay; it has stored none of them. What persists is the change those frames caused in the network's weights, the same way you don't keep video recordings of every basketball shot you've practiced, your muscle memory just changed.

That's why the trained "brain" is a single ~20MB file forever, no matter how long it trains. The file is the muscle memory, not the practice footage.

### The learning loop

The cycle repeats endlessly:

1. **Play**: all 10 Marios play for 512 decisions each, about 34 seconds of game time. That's one batch: ~5,120 moves spread over roughly 20 to 45 runs depending on how fast they're dying.
2. **Study**: training pauses (the "STUDYING LAST N RUNS..." banner) and the network reads through that batch 10 times. Every run gets studied, including the bad ones: actions from runs that scored well become slightly more probable, actions from runs that died become slightly less probable.
3. **Discard**: the batch is deleted and play resumes with the slightly-updated brain.

Slow runs are not skipped; they are the negative examples. But once studied, a run is useless: it describes what the *old* brain would have done, and the algorithm (PPO) is only mathematically valid on fresh behavior from the *current* brain. Throwing data away isn't a memory-saving trick, it's a correctness requirement. This is what "on-policy" means.

### What the reward pays for

The AI never gets told "jump over the pipe". It gets points, every frame, for exactly three things (plus shaping):

- moving right (speed),
- the clock ticking down (a constant penalty, so wasting time hurts),
- dying (a big penalty), reaching the flag (a big bonus), and a small bonus for score pickups along the way.

Everything you see it do is downstream of those numbers.

### How it beats an obstacle it has never seen

It doesn't know what a pipe is. It runs into the wall a few hundred times. But its choices always include randomness (we even pay it a small bonus, the entropy term, for keeping its options open), and eventually a random jump clears the pipe. That run travels further right, earns more reward, and the study step makes every action from that run more likely, including that jump, in that visual situation. A few hundred repetitions later, "green blob approaching" reliably triggers jump. Multiply by every obstacle in the level.

This is also why the progress chart looks like stairs, not a ramp: long flat stretches while it polishes what it knows, then a sudden step up when luck cracks a new obstacle and gets reinforced.

## Part 5: Engineering lessons that made it work

- **Watchable vs fast is a real trade-off.** Unthrottled, the game runs at many times real speed, great for learning, unwatchable for humans. Realtime mode paces it to true NES speed for spectating and roughly halves the learning rate. Same checkpoints either way, switch freely.
- **The study pause is fundamental, but its length isn't.** PPO must stop playing to learn, and the 10 Marios never finish their runs at the same moment, so the pause can't wait for "all runs complete". Moving the network onto the Apple GPU (MPS) cut the pause from ~25s to ~4s.
- **Checkpoints are the only storage that matters.** One ~20MB file per snapshot. We keep the newest 5 plus million-step milestones, auto-delete the rest, and a hard 5GB fuse protects the disk no matter what. Trained models are published as GitHub Releases so nobody has to retrain from scratch.
- **Test the pipeline before trusting it.** The first training run silently lost progress because the display code was writing a screenshot to disk 40 times per second and the first checkpoint hadn't landed yet. Cheap sanity checks first, long runs second.

## Part 6: The scoreboard so far

| Milestone | Time |
|---|---|
| Ghost (auto-tuned WR-style run, 17 jumps) | **19.60s** |
| First ever clear (fresh model) | 76.67s |
| After resume + speed pressure | 32.33s |
| Same session, later | 24.93s |
| Current best | 22.13s |

The gap left to the ghost is hesitation: late jumps, momentum lost on landings. That's exactly what repetition polishes.

## Part 7: What's next

Once the ghost falls: the whole game, the speedrunner way. Real world records don't play all 32 levels, they use the warp zones: 1-1, 1-2, 4-1, 4-2, then all of world 8. Eight levels, eight small brains (one per level, ~160MB total), transfer learning from each level to the next, and the nastiest challenge in the project: teaching an AI paid to "go right" that in 4-2, the fastest way forward is a warp zone hidden off the obvious path.
