# T2V Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch from I2V (Flux keyframes + CogVideoX-5b-I2V) to pure T2V (CogVideoX-5b text-to-video) to fix video decoherence.

**Architecture:** Remove Flux dependency entirely. Extend Showrunner to generate dense 50-100 word video prompts. Switch CogVideoX from I2V to T2V pipeline. Simplify renderer by removing keyframe generation.

**Tech Stack:** Python, PyTorch, diffusers (CogVideoXPipeline), Ollama (llama3:8b), YAML configs

---

## Task 1: Update Script Dataclass with video_prompts

**Files:**
- Modify: `src/vortex/models/showrunner.py:64-84`
- Test: `tests/unit/test_showrunner.py`

**Step 1: Update Script dataclass**

In `src/vortex/models/showrunner.py`, replace lines 64-84:

```python
@dataclass
class Script:
    """Generated comedy script with 3-scene storyboard and T2V prompts.

    Attributes:
        setup: Opening premise (1 sentence, may include Bark tokens)
        punchline: Absurd conclusion (1 sentence, may include Bark tokens)
        subject_visual: Consistent visual description of the main subject
        storyboard: List of 3 brief scene descriptions (for logging)
        video_prompts: List of 3 dense T2V prompts (50-100 words each)
    """

    setup: str
    punchline: str
    subject_visual: str
    storyboard: list[str]
    video_prompts: list[str]  # Dense prompts for CogVideoX T2V

    @property
    def visual_prompt(self) -> str:
        """Return first video_prompt for backward compatibility."""
        return self.video_prompts[0] if self.video_prompts else ""
```

**Step 2: Run existing tests to see failures**

```bash
cd /home/matt/nsn/vortex && source .venv/bin/activate && python -m pytest tests/unit/test_showrunner.py -v --tb=short 2>&1 | head -50
```

Expected: Tests fail because FALLBACK_TEMPLATES don't have video_prompts yet.

---

## Task 2: Add ADULT_SWIM_STYLE Constant and Update Prompt Template

**Files:**
- Modify: `src/vortex/models/showrunner.py:40-445`

**Step 1: Add ADULT_SWIM_STYLE constant after BARK_TOKEN_INSTRUCTIONS**

Insert after line 51 (after BARK_TOKEN_INSTRUCTIONS closing `"""`):

```python
# Adult Swim / Interdimensional Cable visual style for T2V prompts
ADULT_SWIM_STYLE = (
    "2D cel-shaded cartoon, flat colors, rough expressive linework, "
    "adult swim aesthetic, exaggerated proportions, squash and stretch animation"
)
```

**Step 2: Replace SCRIPT_PROMPT_TEMPLATE**

Replace the entire SCRIPT_PROMPT_TEMPLATE (lines 415-445) with:

```python
SCRIPT_PROMPT_TEMPLATE = """You are a writer for "Interdimensional Cable" - absurdist TV from infinite dimensions.
Write a SHORT surreal TV clip about: {theme}
Tone: {tone}

This could be ANY type of interdimensional TV content:
- Bizarre commercials for impossible products
- Surreal talk shows with weird hosts
- Breaking news from absurd realities
- Movie trailers for films that shouldn't exist
- Game shows with incomprehensible rules
- Public service announcements gone wrong
- Random channel-surfing moments of pure chaos

The clip has 3 QUICK CUTS (5 seconds each):
- Scene 1: Hook - Grab attention with something weird
- Scene 2: Escalation - Things get weirder
- Scene 3: Payoff - Maximum absurdity or abrupt cut

IMPORTANT: Define a MAIN SUBJECT that appears in ALL scenes.
{bark_tokens}
Format your response ONLY as JSON (no markdown, no explanation):
{{
  "setup": "Opening line/premise with optional [laughs], [sighs], [gasps] tokens",
  "punchline": "Closing line or absurd conclusion with optional tokens",
  "subject_visual": "Detailed visual description of the MAIN SUBJECT (under 30 words)",
  "storyboard": [
    "Scene 1: Brief description",
    "Scene 2: Brief description",
    "Scene 3: Brief description"
  ],
  "video_prompts": [
    "DENSE 50-100 word prompt for scene 1. Include: the subject, action, camera movement, lighting, environment. End with: {style}",
    "DENSE 50-100 word prompt for scene 2. Include: the subject, action, camera movement, lighting, environment. End with: {style}",
    "DENSE 50-100 word prompt for scene 3. Include: the subject, action, camera movement, lighting, environment. End with: {style}"
  ]
}}

Rules for video_prompts:
- Each prompt must be 50-100 words
- Start with the subject description from subject_visual
- Describe specific actions and movements
- Include camera direction (zoom in, pan, static shot, shaky cam)
- Include lighting and atmosphere
- ALWAYS end each prompt with: "{style}"
- Focus on VISUALS only - no dialogue, no sound descriptions
"""
```

**Step 3: Update generate_script to pass style to template**

In the `generate_script` method (around line 572), update the prompt formatting:

```python
        # Build the prompt with Bark TTS token instructions and style
        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            theme=theme,
            tone=tone,
            bark_tokens=BARK_TOKEN_INSTRUCTIONS,
            style=ADULT_SWIM_STYLE,
        )
```

---

## Task 3: Update _parse_script_response for video_prompts

**Files:**
- Modify: `src/vortex/models/showrunner.py:632-736`

**Step 1: Update _parse_script_response to handle video_prompts**

Replace the method (lines 632-736) with:

```python
    def _parse_script_response(self, raw_response: str) -> Script:
        """Parse the raw LLM response into a Script object.

        Attempts to extract JSON from the response, handling common
        formatting issues like markdown code blocks or extra text.

        Args:
            raw_response: Raw text response from Ollama

        Returns:
            Parsed Script object with 3-scene storyboard and video_prompts

        Raises:
            ShowrunnerError: If response cannot be parsed as valid JSON
        """
        # Try to extract JSON from the response
        json_str = self._extract_json(raw_response)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse script JSON",
                extra={"raw_response": raw_response[:500], "error": str(e)},
            )
            raise ShowrunnerError(
                f"Invalid JSON response from Ollama: {e}"
            ) from e

        # Validate required fields
        required_string_fields = ["setup", "punchline"]
        missing_fields = [f for f in required_string_fields if f not in data]

        # Check for storyboard (new) or visual_prompt (legacy)
        has_storyboard = "storyboard" in data
        has_visual_prompt = "visual_prompt" in data
        has_video_prompts = "video_prompts" in data

        if not has_storyboard and not has_visual_prompt:
            missing_fields.append("storyboard")

        if missing_fields:
            raise ShowrunnerError(
                f"Missing required fields in response: {missing_fields}"
            )

        # Validate string field types and content
        for field in required_string_fields:
            value = data[field]
            if not isinstance(value, str):
                raise ShowrunnerError(
                    f"Field '{field}' must be a string, got {type(value).__name__}"
                )
            if not value.strip():
                raise ShowrunnerError(f"Field '{field}' cannot be empty")

        # Parse storyboard (or convert from legacy visual_prompt)
        if has_storyboard:
            storyboard = data["storyboard"]
            if not isinstance(storyboard, list):
                raise ShowrunnerError(
                    f"Field 'storyboard' must be a list, got {type(storyboard).__name__}"
                )
        else:
            # Legacy format: convert visual_prompt to single-scene storyboard
            visual_prompt = data["visual_prompt"].strip()
            storyboard = [visual_prompt, visual_prompt, visual_prompt]
            logger.warning("Using legacy visual_prompt format, duplicating to 3 scenes")

        # Validate storyboard has 3 scenes
        if len(storyboard) != 3:
            logger.warning(f"Expected 3 scenes, got {len(storyboard)}. Padding/trimming.")
            if len(storyboard) < 3:
                while len(storyboard) < 3:
                    storyboard.append(f"Scene {len(storyboard)+1}: continuation of the action")
            else:
                storyboard = storyboard[:3]

        # Validate each scene is a non-empty string
        cleaned_storyboard = []
        for i, scene in enumerate(storyboard):
            if not isinstance(scene, str):
                raise ShowrunnerError(
                    f"Scene {i+1} must be a string, got {type(scene).__name__}"
                )
            cleaned_scene = scene.strip()
            if not cleaned_scene:
                cleaned_scene = f"Scene {i+1}: continuation of the action"
            cleaned_storyboard.append(cleaned_scene)

        # Parse subject_visual (new field for subject anchoring)
        subject_visual = data.get("subject_visual", "")
        if not subject_visual or not subject_visual.strip():
            subject_visual = "the main character"
            logger.warning("Missing subject_visual, using generic fallback")
        else:
            subject_visual = subject_visual.strip()

        # Parse video_prompts (new field for T2V)
        if has_video_prompts:
            video_prompts = data["video_prompts"]
            if not isinstance(video_prompts, list):
                raise ShowrunnerError(
                    f"Field 'video_prompts' must be a list, got {type(video_prompts).__name__}"
                )
        else:
            # Generate video_prompts from storyboard + subject + style
            logger.warning("Missing video_prompts, generating from storyboard")
            video_prompts = []
            for scene in cleaned_storyboard:
                prompt = f"{subject_visual}, {scene}. {ADULT_SWIM_STYLE}"
                video_prompts.append(prompt)

        # Validate video_prompts has 3 entries
        if len(video_prompts) != 3:
            logger.warning(f"Expected 3 video_prompts, got {len(video_prompts)}. Padding/trimming.")
            if len(video_prompts) < 3:
                while len(video_prompts) < 3:
                    prompt = f"{subject_visual}, Scene {len(video_prompts)+1}. {ADULT_SWIM_STYLE}"
                    video_prompts.append(prompt)
            else:
                video_prompts = video_prompts[:3]

        # Validate each video_prompt is a non-empty string
        cleaned_video_prompts = []
        for i, prompt in enumerate(video_prompts):
            if not isinstance(prompt, str):
                raise ShowrunnerError(
                    f"video_prompt {i+1} must be a string, got {type(prompt).__name__}"
                )
            cleaned_prompt = prompt.strip()
            if not cleaned_prompt:
                cleaned_prompt = f"{subject_visual}, Scene {i+1}. {ADULT_SWIM_STYLE}"
            cleaned_video_prompts.append(cleaned_prompt)

        return Script(
            setup=data["setup"].strip(),
            punchline=data["punchline"].strip(),
            subject_visual=subject_visual,
            storyboard=cleaned_storyboard,
            video_prompts=cleaned_video_prompts,
        )
```

---

## Task 4: Create YAML Fallback Scripts Config

**Files:**
- Create: `src/vortex/models/configs/fallback_scripts.yaml`

**Step 1: Create the YAML config file**

Create file `src/vortex/models/configs/fallback_scripts.yaml`:

```yaml
# Fallback scripts for when Ollama is unavailable
# Each template follows Interdimensional Cable's absurdist style
# Used by Showrunner.get_fallback_script() when LLM is not available

style_suffix: "2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

templates:
  # Fake commercials - products that shouldn't exist
  - setup: "[clears throat] Are you tired of your... regular teeth?"
    punchline: "Try Teeth-B-Gone! [laughs] Now your mouth is just a SMOOTH hole!"
    subject_visual: "a frustrated middle-aged cartoon man with brown hair and a blue shirt"
    storyboard:
      - "Scene 1: A frustrated cartoon man pointing at his normal teeth in a bathroom mirror, infomercial lighting, 1990s aesthetic"
      - "Scene 2: The man applies a glowing product to his teeth, sparkles and magical effects, bright colors, transformation sequence"
      - "Scene 3: The man smiles revealing a completely smooth toothless mouth, thumbs up to camera, surreal body horror, disturbingly happy"
    video_prompts:
      - "A frustrated middle-aged cartoon man with brown hair and a blue shirt stands in a cramped bathroom with mint-green tiles, pointing accusingly at his perfectly normal teeth in a foggy mirror. Harsh fluorescent lighting casts unflattering shadows on the dingy wallpaper. The camera slowly zooms in on his exaggerated disappointed expression as he opens his mouth wide to inspect his teeth. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A frustrated middle-aged cartoon man with brown hair and a blue shirt squeezes a glowing neon-pink tube labeled TEETH-B-GONE, applying the sparkling gel to his teeth with manic enthusiasm. Magical particles and sparkles swirl around his head in a vortex of transformation energy. The camera pulls back to show his whole body vibrating with crackling magical effects as his teeth begin to dissolve. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A frustrated middle-aged cartoon man with brown hair and a blue shirt grins directly at camera revealing a completely smooth featureless pink gum line where teeth should be. He gives two enthusiastic thumbs up while standing in a pristine white void with sparkles falling around him. The camera slowly zooms into the disturbing empty mouth hole as he maintains his unsettling smile. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  - setup: "Introducing the new Plumbus 2.0."
    punchline: "It does the same thing, but now it's blue!"
    subject_visual: "a pink alien plumbus device with weird protrusions and organic texture"
    storyboard:
      - "Scene 1: A pink alien plumbus device on a white pedestal, product showcase lighting, mysterious alien technology"
      - "Scene 2: Factory workers dipping the plumbus in blue dye, assembly line aesthetic, cartoon style, neon lighting"
      - "Scene 3: The blue plumbus spinning majestically, confetti falling, product reveal moment, absurd celebration, same weird shape"
    video_prompts:
      - "A pink alien plumbus device with weird fleshy protrusions and organic pulsating texture sits on a pristine white pedestal in a sterile showroom. Dramatic product showcase lighting creates a halo effect around the mysterious alien technology. The camera slowly orbits the strange device as it gently throbs and glistens under the spotlights. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A pink alien plumbus device with weird protrusions is dunked into a vat of electric blue dye by robotic factory arms on an assembly line. Neon purple and blue lights illuminate the industrial setting as cartoon factory workers in hazmat suits observe the transformation process. The camera tracks along the conveyor belt showing multiple plumbuses being dipped in sequence. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A now-blue alien plumbus device spins majestically on a rotating platform as golden confetti and streamers fall from above in celebration. Dramatic spotlights sweep across the product reveal stage while the plumbus maintains its same weird organic shape but in vibrant blue. The camera pushes in triumphantly on the spinning plumbus as sparkle effects burst around it. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  - setup: "[sighs] Do your hands keep falling off?"
    punchline: "Stick-It-Back Hand Glue - because duct tape is for QUITTERS!"
    subject_visual: "a cartoon person with detachable hands and a worried expression wearing casual clothes"
    storyboard:
      - "Scene 1: A cartoon person looking sadly at their detached hand on the floor, suburban living room, surreal body horror comedy"
      - "Scene 2: Close-up of glue bottle being applied to wrist stump, infomercial demonstration style, bright studio lighting"
      - "Scene 3: Person waving both hands triumphantly at camera, one slightly crooked, big smile, product success moment"
    video_prompts:
      - "A cartoon person with a worried expression wearing casual clothes looks down sadly at their own detached hand lying on the beige carpet of a suburban living room. The severed wrist shows a clean cartoonish break with no gore. Warm afternoon light streams through venetian blinds as the camera slowly pans from the floor up to their distressed face. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A cartoon person with detachable hands squeezes a large purple bottle labeled STICK-IT-BACK onto the smooth stump of their wrist in close-up. Bright infomercial studio lighting illuminates the demonstration as thick glowing glue oozes onto the wound. The camera holds steady on the application process showing the glue spreading and bubbling with magical energy. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A cartoon person with reattached hands waves both arms triumphantly at the camera with a huge satisfied smile. One hand is slightly crooked and at a wrong angle but they seem thrilled anyway. Confetti falls in the bright studio as the camera slowly zooms out to show their full victorious pose. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  # Talk shows with weird hosts
  - setup: "[sighs] Welcome back to Cooking with Regret... I'm your host, a sentient cloud of disappointment."
    punchline: "Today we're making my father's approval - [laughs] just kidding, that's IMPOSSIBLE!"
    subject_visual: "a sad gray cloud with cartoon eyes wearing a chef's hat floating in mid-air"
    storyboard:
      - "Scene 1: A sad gray cloud with eyes floating behind a kitchen counter, pastel studio set, cooking show format, surreal cartoon style"
      - "Scene 2: The cloud attempts to mix ingredients but they phase through it, bowls and spoons floating, existential sadness, studio lighting"
      - "Scene 3: Empty plate presentation with a single tear drop from the cloud, dramatic close-up, cooking show finale lighting, melancholy"
    video_prompts:
      - "A sad gray cloud with droopy cartoon eyes wearing a white chef's hat floats behind a pastel pink kitchen counter on a cooking show set. Studio lights illuminate the surreal scene as mixing bowls and ingredients sit untouched on the counter. The camera slowly pushes in on the melancholy cloud as it hovers in place with a perpetually disappointed expression. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A sad gray cloud with cartoon eyes attempts to stir a mixing bowl but the wooden spoon phases directly through its misty form. Ingredients float chaotically in the air around the helpless cloud as it tries and fails to grasp anything solid. The camera captures the existential frustration as flour poofs through the cloud's body under bright studio cooking show lighting. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A sad gray cloud with cartoon eyes presents an empty white plate to the camera with solemn ceremony as a single cartoon tear drops from its misty form. Dramatic cooking show finale lighting creates a spotlight effect on the pathetically empty dish. The camera slowly zooms in on the glistening tear as it falls onto the pristine empty plate. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  - setup: "This is Personal Space, the show that explores the boundaries of personal space."
    punchline: "Step one: stay out of my personal space. Step two: stay out of my personal space."
    subject_visual: "a nervous bald man in a spotlight with wide anxious eyes and sweaty forehead"
    storyboard:
      - "Scene 1: A nervous bald man in a spotlight on empty black stage, uncomfortable close-up, sweat visible, talk show format"
      - "Scene 2: The man drawing a chalk circle around himself frantically, paranoid expression, dramatic shadows, surreal"
      - "Scene 3: Extreme close-up of the man's face filling entire screen, wild eyes, ironic violation of personal space, unsettling"
    video_prompts:
      - "A nervous bald man with wide anxious eyes and visible sweat droplets on his forehead stands alone in a harsh spotlight on an empty black stage. The dramatic lighting creates stark shadows on his twitching face as he glances around paranoidly. The camera slowly creeps closer as he shifts uncomfortably in the uncomfortable silence of the void. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A nervous bald man frantically draws a white chalk circle around himself on the black stage floor with shaking hands. His paranoid wide eyes dart around as dramatic shadows stretch across the void behind him. The camera orbits around him as he completes the protective circle and hugs himself defensively. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A nervous bald man's sweaty face fills the entire screen in an extreme uncomfortable close-up as his wild bulging eyes stare directly into the camera. Every pore and bead of sweat is visible in the harsh lighting as his face twitches involuntarily. The camera pushes even closer in an ironic violation of the very personal space he was discussing. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  # News broadcasts with absurd topics
  - setup: "[clears throat] Breaking news: local man discovers his reflection has been living a BETTER life."
    punchline: "[gasps] The reflection reportedly has a nicer apartment and remembers birthdays!"
    subject_visual: "a sad cartoon man with messy brown hair in casual clothes staring at a mirror"
    storyboard:
      - "Scene 1: News anchor at desk with breaking news graphics, professional broadcast style, dramatic lighting"
      - "Scene 2: Split screen showing sad man vs happy reflection in mirror, reflection's side has nicer furniture, surreal comparison"
      - "Scene 3: The reflection waving smugly from inside the mirror, holding a birthday cake, man crying outside, news format"
    video_prompts:
      - "A professional news anchor sits at a curved desk with BREAKING NEWS graphics flashing red behind them on multiple screens. Dramatic broadcast lighting illuminates the serious set as urgent music plays implied. The camera pushes in slowly on the anchor's concerned face as they deliver the strange headline with gravitas. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A split screen shows a sad cartoon man with messy brown hair in a dingy apartment on the left staring at his reflection on the right who lives in a luxurious modern space. The reflection's side has designer furniture and warm lighting while the real man's side is cluttered and dim. The camera holds steady on the surreal comparison as both figures react to each other. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A happy reflection waves smugly from inside a mirror while holding a birthday cake with lit candles, as the sad cartoon man on the outside cries with tears streaming down his face. The reflection's side of the mirror shows a party happening while the real world is empty and lonely. The camera slowly pushes in on the crying man's devastated face. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  # Public service announcements gone wrong
  - setup: "This is a public service announcement: your furniture has feelings too."
    punchline: "That chair you never sit in? It knows. It knows and it's sad."
    subject_visual: "a living room chair with cartoon eyes and a sad expression in a corner"
    storyboard:
      - "Scene 1: PSA title card with serious font and warning colors, government broadcast aesthetic, dramatic music implied"
      - "Scene 2: Living room with furniture that has cartoon eyes, the chair in corner looks lonely, melancholy lighting"
      - "Scene 3: Close-up of the sad chair with a single tear, cobwebs forming, family laughing on couch in background, surreal guilt"
    video_prompts:
      - "A stark PSA title card with bold serious font reading YOUR FURNITURE HAS FEELINGS appears against warning yellow and black stripes. The government broadcast aesthetic creates an ominous official feeling as the text pulses urgently. The camera holds steady on the dramatic announcement card as implied emergency broadcast tones play. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A living room where all the furniture has sad cartoon eyes stares at the camera with various expressions of loneliness and neglect. In the corner a dusty armchair with particularly droopy eyes sits forgotten and unused. Melancholy blue lighting casts long shadows as the camera slowly pans across the emotionally wounded furniture. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A sad corner armchair with cartoon eyes sheds a single glistening tear as cobwebs form on its armrests from years of neglect. In the blurred background a happy family laughs together on a newer couch completely ignoring the lonely chair. The camera slowly zooms in on the chair's devastated expression as the tear rolls down its fabric surface. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  - setup: "Remember kids, always look both ways before crossing into a parallel dimension."
    punchline: "You might see yourself, and honestly, that guy is a jerk!"
    subject_visual: "a cartoon child with a backpack at a crosswalk near a swirling portal"
    storyboard:
      - "Scene 1: Cartoon child at a crosswalk but instead of street there is a swirling portal, educational PSA style"
      - "Scene 2: Child looking left and right seeing alternate versions of self, one is rude and sticking tongue out, warning colors"
      - "Scene 3: Child and alternate self in fistfight, safety mascot shrugging in corner, cartoon violence, PSA gone wrong"
    video_prompts:
      - "A cartoon child with a colorful backpack stands at a crosswalk but instead of a street there is a swirling purple and blue interdimensional portal crackling with energy. Educational PSA style graphics and safety colors frame the strange scene. The camera holds at child eye level as the kid looks both ways at the mysterious void with innocent caution. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A cartoon child with a backpack looks left and right at a dimensional crosswalk seeing alternate evil versions of themselves on each side. One alternate self sticks out their tongue rudely while another makes an obscene gesture with warning yellow and red colors flashing. The camera whip pans between the shocked child and their rude doppelgangers as alarm bells ring implied. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A cartoon child and their alternate dimension self engage in a cartoon fistfight with action lines and impact stars as a safety mascot character shrugs helplessly in the corner. The educational PSA has gone completely wrong as the two identical kids pummel each other in a dust cloud. The camera pulls back to show the chaos while the mascot holds up a SAFETY FIRST sign ironically. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  # More infomercials
  - setup: "Have you ever wanted to taste colors?"
    punchline: "Introducing Synesthesia Snacks - now purple tastes exactly how you'd expect!"
    subject_visual: "a person with wide eyes staring longingly at a rainbow in dramatic lighting"
    storyboard:
      - "Scene 1: Person staring longingly at a rainbow, dramatic lighting, existential yearning, infomercial problem setup"
      - "Scene 2: Package of Synesthesia Snacks glowing with prismatic colors, product hero shot, psychedelic background, trippy visuals"
      - "Scene 3: Person eating snacks while colors visibly enter their mouth, ecstatic expression, synesthetic explosion, vibrant surreal"
    video_prompts:
      - "A person with wide yearning eyes stares longingly at a beautiful rainbow arcing across a dramatic sky with tears forming in their eyes. Theatrical lighting creates an emotional infomercial problem setup as they reach toward the unreachable colors. The camera slowly zooms in on their desperate expression as they mouth the words why can't I taste you. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A glowing package of SYNESTHESIA SNACKS radiates with shifting prismatic rainbow colors on a rotating pedestal against a psychedelic swirling background. The product hero shot lighting makes the bag appear magical and transcendent as colors pulse and flow across its surface. The camera slowly orbits the miraculous product as trippy visual effects emanate from it. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A person eats Synesthesia Snacks with an ecstatic expression as visible streams of color flow from the chips directly into their open mouth. Their eyes roll back in pleasure as purple blue and orange literally enter their body in a synesthetic explosion. The camera captures the transcendent moment as rainbow light bursts from their head in overwhelming sensory bliss. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"

  - setup: "[sighs] Can't stop thinking about that embarrassing thing from eight years ago?"
    punchline: "Memory Hole - just pour it in your ear and forget RESPONSIBLY! [laughs]"
    subject_visual: "a person lying awake in bed at 3am with anxious eyes and dark circles"
    storyboard:
      - "Scene 1: Person lying awake at 3am with thought bubble showing cringe moment, dark bedroom, anxious expression, relatable horror"
      - "Scene 2: Cheerful person pouring glowing liquid into their own ear, retro infomercial style, bright colors, unsettling smile"
      - "Scene 3: Person with empty eyes and peaceful smile, thought bubble now blank, maybe too peaceful, slightly disturbing satisfaction"
    video_prompts:
      - "A person with dark circles under their anxious wide eyes lies awake in bed at 3am staring at the ceiling as a thought bubble shows an embarrassing memory playing on loop. The dark bedroom is lit only by a digital clock reading 3:47 AM as they cringe at the relatable horror of the recollection. The camera slowly pushes in on their tortured sleepless face as the cringe memory replays. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A cheerful person with an unsettling frozen smile tilts their head and pours glowing green liquid from a bottle labeled MEMORY HOLE directly into their own ear canal. Bright retro infomercial colors and product lighting create a jarring contrast with the disturbing action. The camera holds steady on the bizarre scene as the liquid flows into their brain with a satisfied expression. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
      - "A person sits with completely empty glazed eyes and a peaceful but slightly disturbing smile as their thought bubble above shows only static white noise. They appear almost too calm and content with a blank expression that suggests something fundamental has been erased. The camera slowly zooms in on their vacant satisfied face as they stare at nothing. 2D cel-shaded cartoon, flat colors, rough expressive linework, adult swim aesthetic, exaggerated proportions, squash and stretch animation"
```

---

## Task 5: Update Showrunner to Load YAML Fallbacks

**Files:**
- Modify: `src/vortex/models/showrunner.py`

**Step 1: Add YAML loading for fallback templates**

Replace the `FALLBACK_TEMPLATES` list and `get_fallback_script` method. At the top of the file after imports, add:

```python
from pathlib import Path

# Load fallback templates from YAML config
def _load_fallback_templates() -> tuple[str, list[Script]]:
    """Load fallback script templates from YAML config file."""
    config_path = Path(__file__).parent / "configs" / "fallback_scripts.yaml"

    if not config_path.exists():
        logger.warning(f"Fallback scripts config not found at {config_path}")
        return ADULT_SWIM_STYLE, []

    with open(config_path) as f:
        data = yaml.safe_load(f)

    style_suffix = data.get("style_suffix", ADULT_SWIM_STYLE)
    templates = []

    for t in data.get("templates", []):
        templates.append(Script(
            setup=t["setup"],
            punchline=t["punchline"],
            subject_visual=t.get("subject_visual", "the main character"),
            storyboard=t.get("storyboard", []),
            video_prompts=t.get("video_prompts", []),
        ))

    return style_suffix, templates


# Lazy-loaded fallback templates
_FALLBACK_STYLE: str | None = None
_FALLBACK_TEMPLATES: list[Script] | None = None


def _get_fallback_templates() -> list[Script]:
    """Get fallback templates, loading from YAML if needed."""
    global _FALLBACK_STYLE, _FALLBACK_TEMPLATES
    if _FALLBACK_TEMPLATES is None:
        _FALLBACK_STYLE, _FALLBACK_TEMPLATES = _load_fallback_templates()
    return _FALLBACK_TEMPLATES
```

**Step 2: Remove the old FALLBACK_TEMPLATES list**

Delete the entire `FALLBACK_TEMPLATES: list[Script] = [...]` block (lines 86-411).

**Step 3: Update get_fallback_script method**

Replace the method with:

```python
    def get_fallback_script(
        self,
        theme: str,
        tone: ToneType = "absurd",
        seed: int | None = None,
    ) -> Script:
        """Get a random pre-written script when Ollama is unavailable.

        Args:
            theme: Used to seed selection for determinism when no seed provided.
            tone: Unused, kept for interface consistency.
            seed: Optional seed for deterministic selection.

        Returns:
            Script from the fallback templates.
        """
        templates = _get_fallback_templates()

        if not templates:
            # Emergency fallback if YAML fails to load
            return Script(
                setup="Welcome to Interdimensional Cable!",
                punchline="Where anything can happen!",
                subject_visual="a cartoon TV host",
                storyboard=["Scene 1: Host waves", "Scene 2: Host talks", "Scene 3: Host exits"],
                video_prompts=[
                    f"A cartoon TV host waves at the camera in a colorful studio. {ADULT_SWIM_STYLE}",
                    f"A cartoon TV host talks animatedly with exaggerated gestures. {ADULT_SWIM_STYLE}",
                    f"A cartoon TV host exits stage left with a flourish. {ADULT_SWIM_STYLE}",
                ],
            )

        # Use theme hash + seed for deterministic selection
        rng = random.Random()
        if seed is not None:
            rng.seed(seed)
        else:
            rng.seed(hash(theme))

        script = rng.choice(templates)

        logger.info(
            "Using fallback script template",
            extra={
                "theme": theme,
                "tone": tone,
                "seed": seed,
                "setup_preview": script.setup[:50],
                "num_scenes": len(script.storyboard),
            },
        )

        return script
```

**Step 4: Add yaml import at top of file**

Ensure `yaml` is imported at the top (it should already be there, but verify).

**Step 5: Add backward compatibility alias**

After the `_get_fallback_templates` function, add:

```python
# Backward compatibility alias for tests
FALLBACK_TEMPLATES = property(lambda self: _get_fallback_templates())
```

Wait, that won't work as a module-level variable. Instead, make it a function that returns the templates:

```python
# Backward compatibility for tests that import FALLBACK_TEMPLATES directly
def get_fallback_templates() -> list[Script]:
    """Get fallback templates (for backward compatibility)."""
    return _get_fallback_templates()

# Alias for imports
FALLBACK_TEMPLATES = get_fallback_templates()  # Evaluated at import time
```

Actually, this is tricky because the YAML won't be loaded until first access. Let's use a different approach - make it a lazy property via a class, or just update the tests. For simplicity, we'll update tests in Task 6.

---

## Task 6: Update CogVideoX from I2V to T2V

**Files:**
- Modify: `src/vortex/models/cogvideox.py`

**Step 1: Update module docstring**

Replace lines 1-24:

```python
"""CogVideoX-5B Text-to-Video model wrapper for Vortex pipeline.

This module provides the CogVideoXModel class that generates video directly
from text prompts using CogVideoX-5B with INT8 quantization for memory efficiency.

The CogVideoX model is part of the Narrative Chain pipeline (Phase 3) and:
- Generates video directly from text prompts (no keyframe required)
- Uses INT8 quantization to fit in 12GB VRAM (down from ~26GB)
- Supports CPU offloading for memory efficiency
- Returns video frames as torch tensors for downstream processing

VRAM Budget: ~10-11 GB (INT8 quantized with CPU offload)
Output: 49 frames at 720x480 at 8fps (~6 seconds)

Example:
    >>> model = CogVideoXModel()
    >>> model.load()
    >>> frames = await model.generate_chunk(
    ...     prompt="A cartoon man waving at the camera in a colorful studio",
    ...     seed=42
    ... )
    >>> print(frames.shape)  # [49, 3, 480, 720]
"""
```

**Step 2: Update model_id default**

In `CogVideoXModel` dataclass (line 120), change:

```python
    model_id: str = "THUDM/CogVideoX-5b"  # T2V model (was: CogVideoX-5b-I2V)
```

**Step 3: Update class docstring**

Replace lines 93-118:

```python
@dataclass
class CogVideoXModel:
    """CogVideoX-5B Text-to-Video model wrapper.

    Generates video directly from text prompts using CogVideoX-5B with INT8
    quantization for memory efficiency on 12GB GPUs.

    This model uses:
    - INT8 weight quantization via torchao to reduce VRAM from ~26GB to ~10GB
    - Model CPU offloading for sequential component loading
    - bfloat16 compute dtype for inference quality

    Attributes:
        model_id: HuggingFace model ID for CogVideoX-5B T2V
        device: Target device ("cuda" or "cpu")
        enable_cpu_offload: Whether to use model CPU offload for VRAM efficiency
        cache_dir: Optional cache directory for model weights

    Example:
        >>> model = CogVideoXModel(enable_cpu_offload=True)
        >>> model.load()
        >>> frames = await model.generate_chunk(
        ...     prompt="A cartoon character dancing in a studio"
        ... )
        >>> model.unload()
    """
```

**Step 4: Update load() to use CogVideoXPipeline**

Replace lines 160-165 import:

```python
        try:
            from diffusers import (
                CogVideoXPipeline,  # Changed from CogVideoXImageToVideoPipeline
                PipelineQuantizationConfig,
                TorchAoConfig,
            )
```

And line 185:

```python
            # Load pipeline with quantization config
            self._pipe = CogVideoXPipeline.from_pretrained(  # Changed
                self.model_id,
                quantization_config=pipeline_quant_config,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )
```

**Step 5: Update generate_chunk to remove image parameter**

Replace the entire `generate_chunk` method (lines 246-342):

```python
    async def generate_chunk(
        self,
        prompt: str,
        config: VideoGenerationConfig | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate a video chunk from a text prompt.

        Takes a text prompt and generates a video sequence directly
        without requiring a keyframe image.

        Args:
            prompt: Text prompt describing the desired video content
            config: Optional generation configuration (uses defaults if None)
            seed: Optional seed for deterministic/reproducible generation

        Returns:
            Video frames tensor of shape [num_frames, channels, height, width]
            with values in 0-1 range (float32)

        Raises:
            CogVideoXError: If generation fails or model not loaded
            ValueError: If prompt is empty

        Example:
            >>> frames = await model.generate_chunk(
            ...     prompt="A cartoon man waving at camera, flat colors, cartoon style",
            ...     config=VideoGenerationConfig(num_frames=49),
            ...     seed=42
            ... )
            >>> print(frames.shape)  # [49, 3, 480, 720]
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()

        if config is None:
            config = VideoGenerationConfig()

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Set up generator for deterministic results
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            "Generating video chunk (T2V)",
            extra={
                "prompt_length": len(prompt),
                "num_frames": config.num_frames,
                "guidance_scale": config.guidance_scale,
                "num_inference_steps": config.num_inference_steps,
                "seed": seed,
            },
        )

        try:
            # Run generation in executor to not block async event loop
            loop = asyncio.get_event_loop()
            video_frames = await loop.run_in_executor(
                None,
                self._generate_sync,
                prompt,
                config,
                generator,
            )

            # Convert list of PIL images to tensor [T, C, H, W]
            frames_tensor = self._frames_to_tensor(video_frames)

            logger.info(
                "Video chunk generated successfully",
                extra={
                    "output_shape": list(frames_tensor.shape),
                    "dtype": str(frames_tensor.dtype),
                },
            )

            return frames_tensor

        except Exception as e:
            logger.error(f"Video generation failed: {e}", exc_info=True)
            raise CogVideoXError(f"Video generation failed: {e}") from e
```

**Step 6: Update _generate_sync to remove image parameter**

Replace the method (lines 535-566):

```python
    def _generate_sync(
        self,
        prompt: str,
        config: VideoGenerationConfig,
        generator: torch.Generator | None,
    ) -> list:
        """Synchronous video generation (called from executor).

        Args:
            prompt: Text prompt
            config: Generation configuration
            generator: Optional random generator for determinism

        Returns:
            List of PIL image frames
        """
        result = self._pipe(
            prompt=prompt,
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            use_dynamic_cfg=config.use_dynamic_cfg,
            num_inference_steps=config.num_inference_steps,
            generator=generator,
        )

        # CogVideoX returns frames[0] as list of PIL images
        return result.frames[0]
```

**Step 7: Update generate_montage to remove keyframes parameter**

Replace the entire `generate_montage` method (lines 447-533):

```python
    async def generate_montage(
        self,
        prompts: list[str],
        config: VideoGenerationConfig | None = None,
        seed: int | None = None,
        trim_frames: int = 40,  # Trim each 49-frame clip to 40 frames (~5s)
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """Generate video montage from multiple text prompts.

        Generates each clip independently from its own prompt (T2V),
        then concatenates them with hard cuts.

        Args:
            prompts: List of text prompts (one per scene)
            config: Optional generation config
            seed: Optional base seed (scene seeds = seed + scene_idx)
            trim_frames: Frames to keep per clip (default 40 = 5s @ 8fps).
                        Set to 0 to disable trimming and keep all frames.
            progress_callback: Optional callback(scene_num, total_scenes) called
                              before each scene and after completion

        Returns:
            Concatenated video tensor [total_frames, C, H, W]

        Example:
            >>> video = await model.generate_montage(
            ...     prompts=["Scene 1 prompt...", "Scene 2 prompt...", "Scene 3 prompt..."],
            ...     seed=42,
            ... )
            >>> print(video.shape)  # [120, 3, 480, 720] for 3x40 frames
        """
        if config is None:
            config = VideoGenerationConfig()

        if len(prompts) == 0:
            raise ValueError("prompts list cannot be empty")

        num_scenes = len(prompts)
        logger.info(f"Generating {num_scenes}-scene montage (T2V)...")

        clips = []
        for i, prompt in enumerate(prompts):
            if progress_callback:
                progress_callback(i, num_scenes)

            # Derived seed for determinism
            scene_seed = seed + i if seed is not None else None

            logger.info(f"Generating scene {i+1}/{num_scenes}: {prompt[:50]}...")

            # Generate clip from prompt (T2V - no keyframe)
            clip = await self.generate_chunk(
                prompt=prompt,
                config=config,
                seed=scene_seed,
            )

            # Trim to target frames (remove potential tail degradation)
            if trim_frames and clip.shape[0] > trim_frames:
                clip = clip[:trim_frames]
                logger.debug(f"Trimmed scene {i+1} to {trim_frames} frames")

            clips.append(clip)
            logger.info(f"Scene {i+1} complete: {clip.shape[0]} frames")

        # Hard cut concatenation
        video = torch.cat(clips, dim=0)

        if progress_callback:
            progress_callback(num_scenes, num_scenes)

        logger.info(
            f"Montage complete: {video.shape[0]} frames "
            f"({video.shape[0] / config.fps:.1f}s @ {config.fps}fps)"
        )

        return video
```

**Step 8: Remove or update generate_chain method**

The `generate_chain` method used I2V chaining. For T2V, we can either remove it or update it. Let's keep it but adapt for T2V (using last frame as context for prompt, though T2V doesn't use images). Actually, for T2V we should just remove `generate_chain` since it doesn't make sense without images. But to avoid breaking changes, let's deprecate it with a warning:

```python
    async def generate_chain(
        self,
        keyframe: torch.Tensor,
        prompt: str,
        target_duration: float,
        config: VideoGenerationConfig | None = None,
        seed: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        """DEPRECATED: Use generate_montage() instead.

        This method was designed for I2V chaining. With T2V, use generate_montage()
        with multiple prompts for each scene instead.
        """
        import warnings
        warnings.warn(
            "generate_chain() is deprecated for T2V. Use generate_montage() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Fall back to single-prompt montage
        if config is None:
            config = VideoGenerationConfig()

        num_chunks = max(1, int(np.ceil(target_duration / (config.num_frames / config.fps))))
        prompts = [prompt] * num_chunks

        return await self.generate_montage(
            prompts=prompts,
            config=config,
            seed=seed,
            trim_frames=0,  # Don't trim for chain-like behavior
            progress_callback=progress_callback,
        )
```

**Step 9: Remove _to_pil_image method**

Delete the entire `_to_pil_image` method (lines 568-634) since T2V doesn't need image conversion.

**Step 10: Remove PIL Image type hint**

Remove the `TYPE_CHECKING` import and `Image` type hint since we no longer use PIL images as input:

```python
# Remove these lines:
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from PIL import Image
```

---

## Task 7: Update Renderer to Remove Flux

**Files:**
- Modify: `src/vortex/renderers/default/renderer.py`

**Step 1: Update module docstring**

Replace lines 1-30:

```python
"""Default Lane 0 renderer implementation with T2V Montage architecture.

This renderer implements the T2V Montage pipeline for Lane 0 video generation:
1. Showrunner (LLM) generates comedic script with 3-scene storyboard and video_prompts
2. Bark synthesizes script to audio (sets duration)
3. CogVideoX generates 3 independent video clips from video_prompts (T2V)
4. Clips concatenated with hard cuts for 15s montage
5. CLIP verifies semantic consistency

Architecture:
    - Showrunner (external Ollama): Generates script with setup/punchline/storyboard[3]/video_prompts[3]
    - Bark (FP16, ~1.5GB): Text-to-speech synthesis at 24kHz with emotion support
    - CogVideoX-5B (INT8, ~10-11GB): Generates 3 video clips from text prompts (T2V)
    - CLIP ViT-B-32 + ViT-L-14 (FP16, 0.6GB): Dual ensemble semantic verification

Pipeline Flow (T2V Montage):
    Recipe -> Showrunner (script with video_prompts) -> Bark (TTS) -> Audio

    video_prompts[3] -> CogVideoX T2V (3 clips) -> concatenate -> Video

    Video -> CLIP (verify) -> Embedding

VRAM Budget: ~11GB peak during CogVideoX phase
Target Duration: 15 seconds (3 scenes × 5s each) at 8fps = 120 frames
"""
```

**Step 2: Remove Flux import**

Remove line 51 (or wherever the flux import is):

```python
# DELETE this line:
# from vortex.models.flux import FluxModel
```

**Step 3: Update _ModelRegistry docstring and remove flux loading**

Update the docstring for `_ModelRegistry` (around line 78):

```python
class _ModelRegistry:
    """Registry for loaded models with sequential loading for Narrative Chain.

    The Narrative Chain pipeline requires sequential model loading due to
    CogVideoX's large VRAM footprint (~10-11GB with INT8 quantization).

    Models:
    - bark: TTS audio synthesis with emotion support (~1.5GB)
    - cogvideox: Video generation (10-11GB with INT8)
    - clip_ensemble: Semantic verification (0.6GB)
    """
```

**Step 4: Remove Flux loading from load_all_models**

In `load_all_models` method, remove the Flux loading block (lines 159-164):

```python
            # DELETE this block:
            # # Load Flux-Schnell keyframe generator
            # from vortex.models.flux import FluxModel
            # logger.info("Loading model: flux")
            # flux = FluxModel(device=self.device, cache_dir=self._cache_dir)
            # # Lazy load - will load when first used (via generate() method)
            # self._models["flux"] = flux
            # log_vram_snapshot("after_flux_load")
```

**Step 5: Remove "flux" from required models in health_check**

Update line 999:

```python
        # Check all required models are loaded
        required_models = ["bark", "cogvideox", "clip_ensemble"]  # Removed "flux"
```

**Step 6: Remove _actor_buffer allocation**

In `_allocate_buffers` method, remove the actor buffer code (lines 386-395):

```python
    def _allocate_buffers(self, config: dict[str, Any]) -> None:
        """Pre-allocate output buffers to prevent fragmentation."""
        buf_cfg = config.get("buffers", {})

        # Audio buffer (24kHz * 15 seconds = 360000 samples max)
        audio_cfg = buf_cfg.get("audio", {})
        self._audio_buffer = torch.zeros(
            audio_cfg.get("samples", 360000),  # 15 seconds @ 24kHz
            device=self._device,
            dtype=torch.float32,
        )

        logger.info(
            "Output buffers pre-allocated",
            extra={
                "audio_shape": tuple(self._audio_buffer.shape),
            },
        )
```

**Step 7: Remove _actor_buffer from __init__**

Remove line 316:

```python
        # DELETE:
        # self._actor_buffer: torch.Tensor | None = None
```

**Step 8: Delete _generate_keyframe method entirely**

Delete the entire `_generate_keyframe` method (lines 685-751).

**Step 9: Update _generate_video to use T2V**

Replace the entire `_generate_video` method (lines 753-851):

```python
    async def _generate_video(
        self,
        script: Script,
        recipe: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Generate video montage from script video_prompts using T2V.

        Uses the T2V montage architecture: video_prompts -> CogVideoX T2V clips
        -> concatenate with hard cuts.

        Args:
            script: Script object with video_prompts list
            recipe: Recipe with video configuration
            seed: Deterministic seed for reproducibility

        Returns:
            Video frames tensor [num_frames, 3, H, W] (approximately 15s at 8fps)

        Raises:
            ValueError: If script doesn't have video_prompts
        """
        assert self._model_registry is not None

        video_prompts = script.video_prompts
        if not video_prompts or len(video_prompts) < 3:
            raise ValueError(
                f"Script must have 3 video_prompts, got "
                f"{len(video_prompts) if video_prompts else 0}"
            )

        logger.info(f"Generating {len(video_prompts)}-scene T2V montage...")
        self._model_registry.prepare_for_stage("video")
        cogvideox = self._model_registry.get_cogvideox()

        # Configure for montage
        config = VideoGenerationConfig(
            num_frames=49,
            guidance_scale=3.5,  # Lower to reduce artifacts
            use_dynamic_cfg=True,
            fps=8,
        )

        video = await cogvideox.generate_montage(
            prompts=video_prompts,
            config=config,
            seed=seed,
            trim_frames=40,  # 5s per scene @ 8fps = 15s total
        )

        logger.info(
            f"T2V Montage complete: {video.shape[0]} frames ({video.shape[0]/8:.1f}s)"
        )
        return video
```

**Step 10: Update _generate_script for video_prompts**

In `_generate_script` method, update the manual script handling (around line 620):

```python
        else:
            # Use provided script from recipe
            script_data = narrative.get("script", {})
            # Handle storyboard
            storyboard = script_data.get("storyboard", [])
            if not storyboard:
                visual_prompt = script_data.get("visual_prompt", "")
                storyboard = [visual_prompt, visual_prompt, visual_prompt] if visual_prompt else []

            # Handle video_prompts
            video_prompts = script_data.get("video_prompts", [])
            if not video_prompts:
                # Generate from storyboard + style
                subject = script_data.get("subject_visual", "the main character")
                video_prompts = [
                    f"{subject}, {scene}. {CLEAN_STYLE_PROMPT}"
                    for scene in storyboard
                ]

            return Script(
                setup=script_data.get("setup", ""),
                punchline=script_data.get("punchline", ""),
                subject_visual=script_data.get("subject_visual", ""),
                storyboard=storyboard,
                video_prompts=video_prompts,
            )
```

**Step 11: Remove _actor_buffer from shutdown**

In `shutdown` method, remove the actor buffer clearing (around line 1035):

```python
        # DELETE:
        # self._actor_buffer = None
```

**Step 12: Remove prepare_for_stage "image" case**

In `_ModelRegistry.prepare_for_stage`, remove the "image" stage:

```python
        # Map stages to required models
        stage_models = {
            "audio": "bark",
            "video": None,  # CogVideoX handles its own offloading
            "clip": "clip_ensemble",
        }
```

---

## Task 8: Delete Flux Files and Update Imports

**Files:**
- Delete: `src/vortex/models/flux.py`
- Modify: `src/vortex/models/__init__.py`
- Modify: `src/vortex/renderers/default/manifest.yaml`

**Step 1: Delete flux.py**

```bash
rm /home/matt/nsn/vortex/src/vortex/models/flux.py
```

**Step 2: Update models/__init__.py**

Replace with:

```python
"""Vortex model wrappers.

This package provides wrappers for:
- Bark: Expressive TTS with paralinguistic sounds
- Showrunner: LLM-based script generation via Ollama
- CogVideoX: Text-to-video generation with INT8 quantization
"""

from vortex.models.bark import BarkVoiceEngine, load_bark
from vortex.models.cogvideox import (
    CogVideoXError,
    CogVideoXModel,
    VideoGenerationConfig,
    load_cogvideox,
)
from vortex.models.showrunner import Script, Showrunner, ShowrunnerError

__all__ = [
    "BarkVoiceEngine",
    "CogVideoXError",
    "CogVideoXModel",
    "Script",
    "Showrunner",
    "ShowrunnerError",
    "VideoGenerationConfig",
    "load_bark",
    "load_cogvideox",
]
```

**Step 3: Update manifest.yaml**

Replace with:

```yaml
schema_version: "1.0"
name: "default-narrative-chain"
version: "3.0.0"
entrypoint: "renderer.py:DefaultRenderer"
description: "NSN Lane 0 Narrative Chain renderer using Showrunner (Ollama) for scripts with video_prompts, Bark for TTS, CogVideoX T2V for video, and dual CLIP for verification."
deterministic: true
resources:
  vram_gb: 11.0
  max_latency_ms: 150000
model_dependencies:
  - cogvideox-5b-int8
  - bark-tts
  - clip-vit-b-32-fp16
  - clip-vit-l-14-fp16
```

---

## Task 9: Update Tests

**Files:**
- Modify: `tests/unit/test_showrunner.py`
- Modify: `tests/unit/test_cogvideox.py`
- Modify: `tests/unit/test_renderer_montage.py`

**Step 1: Update test_showrunner.py for video_prompts**

Add new test class after existing tests:

```python
class TestScriptVideoPrompts:
    """Test suite for Script.video_prompts field."""

    def test_script_has_video_prompts_attribute(self):
        """Script must have video_prompts attribute."""
        script = Script(
            setup="Test setup",
            punchline="Test punchline",
            subject_visual="test subject",
            storyboard=["Scene 1", "Scene 2", "Scene 3"],
            video_prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
        )
        assert hasattr(script, "video_prompts")
        assert len(script.video_prompts) == 3

    def test_visual_prompt_returns_first_video_prompt(self):
        """visual_prompt property should return first video_prompt."""
        script = Script(
            setup="Test",
            punchline="Test",
            subject_visual="subject",
            storyboard=["S1", "S2", "S3"],
            video_prompts=["First prompt", "Second prompt", "Third prompt"],
        )
        assert script.visual_prompt == "First prompt"

    def test_visual_prompt_empty_for_empty_video_prompts(self):
        """visual_prompt returns empty string if video_prompts is empty."""
        script = Script(
            setup="Test",
            punchline="Test",
            subject_visual="subject",
            storyboard=["S1", "S2", "S3"],
            video_prompts=[],
        )
        assert script.visual_prompt == ""
```

Update `TestFallbackTemplates` to check for video_prompts:

```python
    def test_fallback_templates_have_video_prompts(self):
        """Each template must have 3 video_prompts."""
        templates = _get_fallback_templates()
        for i, template in enumerate(templates):
            assert hasattr(template, "video_prompts"), f"Template {i} missing 'video_prompts'"
            assert isinstance(template.video_prompts, list), f"Template {i} video_prompts must be a list"
            assert len(template.video_prompts) == 3, f"Template {i} must have 3 video_prompts"
            for j, prompt in enumerate(template.video_prompts):
                assert isinstance(prompt, str), f"Template {i} video_prompt {j} must be string"
                assert len(prompt) >= 50, f"Template {i} video_prompt {j} should be >= 50 chars"
```

**Step 2: Update test_cogvideox.py for T2V**

Update `TestGenerateChainSignature` and method tests to remove image parameter, and update `TestGenerateMontageSignature` to remove keyframes:

```python
class TestGenerateChunkSignature:
    """Tests for the generate_chunk method signature (T2V)."""

    def test_generate_chunk_parameters(self) -> None:
        """Test generate_chunk has correct parameters for T2V."""
        sig = inspect.signature(CogVideoXModel.generate_chunk)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "prompt",  # No more "image" parameter
            "config",
            "seed",
        ]
        assert params == expected_params


class TestGenerateMontageSignature:
    """Tests for the generate_montage method signature (T2V)."""

    def test_generate_montage_parameters(self) -> None:
        """Test generate_montage has correct parameters for T2V."""
        sig = inspect.signature(CogVideoXModel.generate_montage)
        params = list(sig.parameters.keys())
        expected_params = [
            "self",
            "prompts",  # No more "keyframes" parameter
            "config",
            "seed",
            "trim_frames",
            "progress_callback",
        ]
        assert params == expected_params
```

**Step 3: Update test_renderer_montage.py**

Remove all references to keyframes, _generate_keyframe, and _actor_buffer:

```python
# Update mock_renderer fixture to remove _actor_buffer
@pytest.fixture
def mock_renderer(self) -> DefaultRenderer:
    """Create a DefaultRenderer with mocked dependencies."""
    renderer = DefaultRenderer.__new__(DefaultRenderer)

    # Mock model registry
    renderer._model_registry = MagicMock()
    renderer._model_registry.offloading_enabled = True
    renderer._model_registry.prepare_for_stage = MagicMock()

    # Mock CogVideoX model
    mock_cogvideox = MagicMock()
    renderer._model_registry.get_cogvideox.return_value = mock_cogvideox

    # No more _actor_buffer needed

    return renderer
```

Update test methods to use `video_prompts` instead of storyboard for video generation:

```python
@pytest.fixture
def valid_script(self) -> Script:
    """Create a valid Script with video_prompts."""
    return Script(
        setup="Welcome to the interdimensional cable network!",
        punchline="Where everything is made up and the points don't matter!",
        subject_visual="a charismatic TV host with slicked-back hair in a shiny suit",
        storyboard=[
            "Scene 1: Host in studio",
            "Scene 2: Host gesturing",
            "Scene 3: Host exits",
        ],
        video_prompts=[
            "A charismatic TV host with slicked-back hair stands in futuristic studio with neon lights, waving at camera. 2D cel-shaded cartoon style.",
            "A charismatic TV host with slicked-back hair gestures wildly at floating holographic screens, dramatic lighting. 2D cel-shaded cartoon style.",
            "A charismatic TV host with slicked-back hair steps into a swirling portal, waving goodbye. 2D cel-shaded cartoon style.",
        ],
    )
```

---

## Task 10: Run Tests and Verify

**Step 1: Run unit tests**

```bash
cd /home/matt/nsn/vortex && source .venv/bin/activate && python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -50
```

**Step 2: Run import verification**

```bash
cd /home/matt/nsn/vortex && source .venv/bin/activate && python -c "
from vortex.models import CogVideoXModel, Script, Showrunner
from vortex.models.showrunner import ADULT_SWIM_STYLE, _get_fallback_templates

# Verify Script has video_prompts
s = Script(
    setup='test', punchline='test', subject_visual='test',
    storyboard=['a', 'b', 'c'],
    video_prompts=['p1', 'p2', 'p3']
)
print(f'Script.video_prompts: {len(s.video_prompts)} items')
print(f'Script.visual_prompt: {s.visual_prompt}')

# Verify fallback templates load
templates = _get_fallback_templates()
print(f'Fallback templates: {len(templates)} loaded')
if templates:
    print(f'First template video_prompts: {len(templates[0].video_prompts)} items')

# Verify CogVideoX model ID
model = CogVideoXModel()
print(f'CogVideoX model_id: {model.model_id}')
assert 'I2V' not in model.model_id, 'Should be T2V, not I2V!'

print('All imports OK!')
"
```

**Step 3: Run e2e test**

```bash
cd /home/matt/nsn/vortex && source .venv/bin/activate && python scripts/e2e_narrative_test.py --theme "bizarre infomercial" --duration 15 --verbose
```

---

## Verification Checklist

- [ ] Script dataclass has video_prompts field
- [ ] ADULT_SWIM_STYLE constant defined
- [ ] SCRIPT_PROMPT_TEMPLATE updated for T2V
- [ ] _parse_script_response handles video_prompts
- [ ] fallback_scripts.yaml created with dense prompts
- [ ] Showrunner loads fallbacks from YAML
- [ ] CogVideoX uses T2V pipeline (not I2V)
- [ ] CogVideoX.generate_chunk has no image parameter
- [ ] CogVideoX.generate_montage has no keyframes parameter
- [ ] Renderer removes Flux loading
- [ ] Renderer removes _generate_keyframe method
- [ ] Renderer removes _actor_buffer
- [ ] Renderer uses script.video_prompts for T2V
- [ ] flux.py deleted
- [ ] models/__init__.py has no Flux exports
- [ ] manifest.yaml has no flux dependency
- [ ] Unit tests pass
- [ ] E2E test produces animated video
