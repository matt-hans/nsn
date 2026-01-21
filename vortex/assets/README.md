# ToonGen Audio Assets

This directory contains audio assets for the ToonGen pipeline.

## Directory Structure

```
assets/
├── voices/              # F5-TTS voice reference clips (5-10 seconds)
│   ├── manic_salesman.wav
│   ├── nervous_morty.wav
│   ├── monotone_bot.wav
│   └── ...
├── audio/
│   ├── bgm/            # Background music loops
│   │   ├── cheesy_elevator.wav
│   │   ├── 80s_synth.wav
│   │   └── ...
│   └── sfx/            # Sound effects
│       ├── static_glitch.wav
│       ├── explosion_short.wav
│       └── ...
```

## Voice References (F5-TTS)

Voice reference files should be:
- Format: WAV (16-bit PCM)
- Duration: 5-10 seconds of clear speech
- Quality: Clean recording, minimal background noise
- Content: Representative of the target voice style

### MVP Voice Styles (10 required)

1. `manic_salesman.wav` - Fast, loud, infomercial energy
2. `nervous_morty.wav` - Stuttering, cracking voice
3. `monotone_bot.wav` - Flat, robotic delivery
4. `whispering_creep.wav` - Quiet, breathy
5. `excited_host.wav` - Game show energy
6. `deadpan_narrator.wav` - Documentary style
7. `angry_chef.wav` - Gordon Ramsay energy
8. `surfer_dude.wav` - Laid back, California
9. `old_timey.wav` - 1920s radio announcer
10. `alien_visitor.wav` - Confused, formal

## Background Music

BGM files should be:
- Format: WAV (16-bit PCM)
- Style: Loopable (seamless loop points preferred)
- Duration: 30+ seconds (will be looped automatically)

## Sound Effects

SFX files should be:
- Format: WAV (16-bit PCM)
- Duration: 1-5 seconds
- Used for: Intro/outro stings, transitions, emphasis
