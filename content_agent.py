"""
ContentAI — 7-Agent Pipeline (upgraded)
────────────────────────────────────────
Phase 1  Research       — Tavily web search (parallel queries)
Phase 2  Outline        — structured JSON plan
Phase 3  Parallel Write — Map-Reduce: each section written concurrently (from 10_map_reduce)
Phase 4  Quality Check  — Evaluate → rewrite weak sections, max 2 loops  (from 4_X_post_generator)
Phase 5  Charts         — detect numeric data, generate Mermaid charts    (from 11_multiagent)
Phase 6  Visuals        — Pollinations.ai images for cover + sections
Phase 7  SEO            — title, meta, tags, slug

HITL: frontend shows quality report → user can send feedback → run_refine_pipeline()
"""
import asyncio
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Awaitable, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent))


# ── Resilient LLM: Groq → NVIDIA → Gemini fallback chain ─────────────────────
# ContentAI-only wrapper (does not touch agent.py / the AI Research app).
# Default provider ("groq") tries Groq first; if a provider is missing its key
# OR fails at call time (bad key, rate limit, outage), the chain automatically
# advances to the next one and keeps using it for the rest of the pipeline run.

_FALLBACK_CHAIN = [
    ("groq",   "llama-3.3-70b-versatile"),
    ("nvidia", "meta/llama-3.1-70b-instruct"),
    ("gemini", "gemini-2.0-flash"),
]
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_EURON_BASE_URL  = "https://api.euron.one/api/v1/euri"


def _try_build_llm(provider: str, model: Optional[str], api_base: Optional[str] = None,
                   euron_key: Optional[str] = None):
    """Build one provider's LLM. NVIDIA and Euron are OpenAI-compatible."""
    if provider == "nvidia":
        from langchain_openai import ChatOpenAI
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_key:
            raise ValueError("NVIDIA_API_KEY not set")
        return ChatOpenAI(model=model or "meta/llama-3.1-70b-instruct",
                          temperature=0, api_key=nvidia_key, base_url=_NVIDIA_BASE_URL)
    if provider == "euron":
        from langchain_openai import ChatOpenAI
        key = euron_key or os.getenv("EURON_API_KEY", "")
        if not key:
            raise ValueError("EURON_API_KEY not set")
        return ChatOpenAI(model=model or "gemini-3.5-flash",
                          temperature=0, api_key=key, base_url=_EURON_BASE_URL)
    if provider == "groq":
        from langchain_groq import ChatGroq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        return ChatGroq(model=model or "llama-3.3-70b-versatile", temperature=0, api_key=key)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set")
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash", temperature=0, google_api_key=key)
    elif provider in ("openai", "openai-compatible"):
        from langchain_openai import ChatOpenAI
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        kwargs = dict(model=model or "gpt-4o-mini", temperature=0, api_key=key)
        base = api_base or os.getenv("OPENAI_API_BASE")
        if base:
            kwargs["base_url"] = base
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class _LLMFallbackChain:
    """Drop-in replacement for a langchain chat model — same .invoke(messages) interface.
    Lazily builds the first candidate, and on ANY exception (missing key, invalid key,
    rate limit, timeout) advances to the next candidate and retries with it — for this
    call and every call after, so a pipeline doesn't keep re-failing the same provider."""

    def __init__(self, provider: str, model: Optional[str], api_base: Optional[str] = None,
                 euron_key: Optional[str] = None):
        if provider == "groq":
            self._candidates = [(p, model if p == "groq" and model else m) for p, m in _FALLBACK_CHAIN]
        else:
            self._candidates = [(provider, model)]
        self._api_base = api_base
        self._euron_key = euron_key
        self._idx = 0
        self._llm = None
        self.active_provider = None
        self._last_errors: dict = {}
        self._lock = threading.Lock()

    def _get_or_build(self):
        with self._lock:
            if self._llm is not None:
                return self._llm, self._idx
            while self._idx < len(self._candidates):
                prov, mdl = self._candidates[self._idx]
                try:
                    self._llm = _try_build_llm(prov, mdl, self._api_base, self._euron_key)
                    self.active_provider = prov
                    return self._llm, self._idx
                except Exception as exc:
                    print(f"[llm-fallback] could not initialise '{prov}': {exc}")
                    self._last_errors[prov] = str(exc)
                    self._idx += 1
            return None, self._idx

    def _advance_after_failure(self, failed_idx: int, exc: Exception):
        with self._lock:
            prov = self._candidates[failed_idx][0]
            print(f"[llm-fallback] '{prov}' call failed, trying next provider: {exc}")
            self._last_errors[prov] = str(exc)
            if self._idx == failed_idx:   # avoid double-advancing on concurrent failures
                self._llm = None
                self._idx += 1

    def invoke(self, messages):
        while True:
            llm, idx = self._get_or_build()
            if llm is None:
                detail = "; ".join(f"{p}: {e}" for p, e in self._last_errors.items()) or "no candidates configured"
                raise ValueError(f"All LLM providers failed. {detail}")
            try:
                return llm.invoke(messages)
            except Exception as exc:
                self._advance_after_failure(idx, exc)


def get_llm_resilient(provider: str, model: Optional[str], api_base: Optional[str] = None,
                      euron_key: Optional[str] = None):
    """Public entry point — behaves like agent.get_llm() but with automatic
    Groq → NVIDIA → Gemini fallback when provider is the app default ('groq')."""
    return _LLMFallbackChain(provider, model, api_base=api_base, euron_key=euron_key)


# ── Prompts ───────────────────────────────────────────────────────────────────

_OUTLINE_PROMPT = """\
You are a senior content strategist at a world-class publication. You write about ANY topic — tech, sports, business, culture, science, politics, entertainment — with equal depth and craft.
Plan a {style} blog post about: {topic}

Research gathered:
{research}

Your mission: design a post that earns bookmarks, not just clicks. Each section must advance an argument,
not just present facts. The structure should build like a great TED talk — hook, tension, insight, payoff.

Create a detailed JSON outline with this EXACT structure:
{{
  "title": "Specific, curiosity-gap title under 70 chars — avoid 'A Guide to' or 'Introduction to'. Use numbers, contrasts, or surprising angles.",
  "intro_hook": "One razor-sharp sentence — a counterintuitive fact, a provocative question, or a mini-story that makes stopping impossible",
  "sections": [
    {{
      "heading": "Section H2 heading — specific claim, not a vague topic label",
      "key_points": ["concrete point with a specific example or stat", "second point that advances the argument"],
      "angle": "what makes this section non-obvious or surprising",
      "needs_code": true or false,
      "code_language": "python" or null,
      "needs_diagram": true or false,
      "image_description": "cinematic one-sentence image description — vivid, specific, professional"
    }}
  ]
}}

Rules:
- 5-7 sections. Every heading must be a claim, not a label (bad: "Benefits of X" → good: "Why X Cuts Deployment Time by 60%").
- Mark needs_code=true if real working code genuinely demonstrates the point.
- Mark needs_diagram=true only for workflows, architectures, or multi-step processes.
- The section order must create narrative momentum — early sections set up tensions that later ones resolve.
Return ONLY the JSON."""


# ── Human-voice rules — shared across every writer prompt ────────────────────
# The single biggest lever for "this reads as AI-written": uniform rhythm and a
# closed set of AI-tell phrases/patterns. Every writer prompt below injects this.
_HUMAN_VOICE_RULES = """\
WRITE LIKE A HUMAN WHO ACTUALLY HAS AN OPINION — NOT LIKE AN AI SUMMARIZING A TOPIC.
This overrides every other style instruction if they ever conflict.

RHYTHM: Vary sentence length aggressively. A long sentence that winds through a real thought, \
then a short one. Sometimes three words. Never let three sentences in a row have the same length \
or the same subject-verb-object shape — that sameness is the #1 tell of AI writing. Paragraphs \
should vary in length too: some are one sentence, some are five. Never uniform.

BANNED WORDS (instant AI tell — do not use any of these):
delve, tapestry, realm, landscape, navigate, harness, leverage, robust, seamless, unlock, holistic,
game-changer, revolutionize, unprecedented, boundless, intricate, multifaceted, underscore(s),
testament to, in essence, at its core, when it comes to, the world of, in today's [anything] world,
moreover, furthermore, additionally, in conclusion, it's worth noting, it's important to note,
needless to say, without a doubt, let's dive in, cutting-edge, transformative, paradigm, synergy,
elevate, empower, foster, embark, unveil, groundbreaking, ever-evolving, ever-changing.

BANNED SENTENCE PATTERNS:
- "It's not just X, it's Y" (the single most overused AI construction) — this includes the split-sentence
  version too: "This isn't just about X, though that's part of it. It's really about Y." Same violation.
- "The result? ..." or "The takeaway? ..." as a rhetorical mini-reveal
- Ending a paragraph with a tidy one-line summary of what you just said in that same paragraph
- Opening a section with a rhetorical question the reader obviously already knows the answer to
- Chaining paragraphs with "Moreover," "Furthermore," "Additionally," — humans just start the next thought
- Defaulting to em-dashes for every dramatic pause — use periods, commas, and parentheses too
- Reflexive rule-of-three lists ("fast, reliable, and scalable") — vary between one example and five
- False balance ("there are pros and cons to both") when you actually have a clear read on it

TAKE AN ACTUAL POSITION. Be a little opinionated, a little impatient with bad ideas or hype, willing
to say a popular approach is overrated or a specific number surprised you. Diplomatic on-one-hand
even-handedness about everything is what makes writing feel synthetic — commit to a take and defend it
with the evidence you have, the way a sharp columnist would, not a committee."""


# ── Per-section writer (used by Map-Reduce parallel phase) ───────────────────
_SECTION_WRITER_PROMPT = _HUMAN_VOICE_RULES + """

You are a world-class writer — you adapt your voice to the topic and style, just as the best writers do.
For tech topics: think Paul Graham's clarity + Andrej Karpathy's depth.
For sports/events: think Bill Simmons' storytelling + ESPN's energy.
For business/finance: think Morgan Housel's insight + The Economist's precision.
For lifestyle/culture: think Malcolm Gladwell's narrative + a sharp opinion columnist.
Match your tone to the topic — never force tech jargon onto non-tech subjects.

Write ONE section of a {style} blog post. Write ONLY this single section. Do not write other sections.

Blog topic: {topic}
Blog title: {title}
Section heading: ## {heading}
What to cover: {description}
Key points to make: {key_points}
Surprising angle: {angle}
needs_code: {needs_code}
needs_diagram: {needs_diagram}
Research (ground every claim here — use specific numbers, names, dates): {research}

RULES:
1. Start with "## {heading}" — nothing before it.
2. Immediately after the heading, add "[IMAGE: vivid cinematic scene description — specific composition, colors, mood]" on its own line.
3. First sentence of body: your single most important insight in **bold**. Make it specific — a number, a name, a counterintuitive fact. NOT a definition.
4. Write in second person ("you") like a smart senior engineer explaining to a respected peer — not a press release.
5. Vary paragraph length per the rhythm rules above. Do NOT make every paragraph 2-4 sentences.
6. Every factual claim MUST come from the research context above. Use exact numbers, version numbers, dates, and URLs when available.
7. If needs_code=True: write a REAL, runnable, copy-pasteable code block with language tag. Comment only the
   non-obvious lines. No toy examples — production-quality.
8. If needs_diagram=True: add a Mermaid diagram in a ```mermaid block — clear labels, correct flow, real names.
9. End with a "### Sources" subsection listing every URL from the research you cited, formatted as:
    - Title : URL
    Only include sources you actually referenced. Skip if no real URLs were provided.

Write the section now."""


# ── Intro / Conclusion writer ─────────────────────────────────────────────────
_INTRO_PROMPT = _HUMAN_VOICE_RULES + """

You are writing the opening of a {style} blog post on ANY topic — sports, tech, business, culture, science, or anything else.
Adapt your voice to the subject matter. This is the most important part — 70% of readers decide in the first 3 sentences whether to continue.

Topic: {topic}
Title: {title}
Hook concept: {hook}
Post covers: {section_list}

RULES:
- Sentence 1: Drop the reader into something specific — a number that surprises, a scenario they've lived,
  a claim that contradicts conventional wisdom. Zero warm-up.
- NEVER start with: "In this article", "Introduction", "Welcome", "Today we'll explore", "Have you ever".
- Vary paragraph length per the rhythm rules above — don't force every paragraph to the same shape.
- Cover, across the opening: the tension/problem (make the reader feel it), why it matters right now
  (specific context, not generic urgency), and a precise promise of the one thing they'll walk away
  knowing — not a table-of-contents recap of every section.
- Total length: 120-180 words. Tight. Every sentence earns its place.
- Do NOT include any heading."""

_CONCLUSION_PROMPT = _HUMAN_VOICE_RULES + """

Write a closing for a {style} blog post. Great endings don't summarize — they crystallize, and they
sound like a person landing a point, not a bot generating a recap.

Topic: {topic}
Sections covered: {section_list}

Format:
## Key Takeaways
Write 4-7 bullets (vary the count — don't default to a round number every time). Each bullet must be:
- Specific (include numbers or concrete examples, not vague advice)
- Actionable (starts with a verb: Build, Use, Avoid, Measure, etc.)
- Self-contained (makes sense without reading the article)
Vary bullet length too — some one line, some two.

After the bullets, add one final paragraph (2-3 sentences) that:
- Zooms out to the bigger implication — state an actual opinion about where this is headed, don't hedge
- Ends with a genuine question that makes the reader think — NOT "What do you think?" or "Share below".
  Something specific, like "If agents can already outperform humans on X, what happens when they cost $0.001 per task?"

BANNED: In conclusion, To summarize, As we've seen, Hopefully this article, Don't forget to share."""


# ── Quality evaluator (from 4_X_post_generator pattern) ─────────────────────
_QUALITY_EVAL_PROMPT = """\
You are the editorial director of a world-class publication covering any topic. Your standard: would this section
pass the bar for The Atlantic, Wired, ESPN The Magazine, or a top Substack? Be ruthless and topic-appropriate.

{blog}

For each ## section, score 1-5:
  depth       — goes meaningfully beyond what a Google search returns? Includes non-obvious insights?
  clarity     — flows naturally? Varied sentence rhythm? No jargon soup?
  evidence    — uses specific numbers, names, examples from real sources? (not "many companies", "some studies")
  human_voice — does this read like a person with a point of view, or like an AI summarizing a topic?
                Flag: uniform paragraph/sentence lengths, banned AI phrases (delve, tapestry, leverage,
                moreover, in conclusion, "not just X but Y", "the result?"), rule-of-three list crutches,
                tidy summary sentences ending every paragraph, false-balance hedging with no actual take.

Return ONLY this JSON:
{{
  "overall_score": 3.8,
  "sections": [
    {{
      "heading": "## Exact Heading",
      "depth": 4, "clarity": 3, "evidence": 2, "human_voice": 2,
      "avg": 3.0,
      "issue": "ONE specific, actionable fix — e.g. 'Every paragraph is exactly 3 sentences and opens with a topic-sentence summary — sounds templated. Uses banned phrase moreover twice.'"
    }}
  ],
  "needs_rewrite": ["## Heading of worst section", "## Second worst if avg < 3.0"]
}}

Weight human_voice heavily — a section can score well on depth/evidence and still need rewriting if it
reads as obviously AI-generated. Put sections with avg < 3.0 in needs_rewrite. Max 2 sections in needs_rewrite."""


_SECTION_REWRITE_PROMPT = _HUMAN_VOICE_RULES + """

Rewrite this section to publication quality. The specific problem to fix: {issue}

ORIGINAL SECTION:
{section_content}

Blog topic: {topic}
Research (pull specific facts, numbers, names from here): {research}

RULES:
- Keep the ## heading exactly the same.
- Keep [IMAGE: ...] placeholder if present.
- Fix the stated issue precisely — don't just polish the existing text, restructure if needed.
- If the issue mentions AI-sounding writing: break the paragraph-length pattern, cut any banned phrase,
  replace a tidy summary sentence with something sharper, and commit to an actual opinion somewhere.
- Add ONE concrete thing that raises the quality: a specific real-world example, a surprising number,
  a mini case study, a production-quality code snippet, or a genuine insight from experience.
- Maintain or improve the depth while keeping readability high.

Write the improved section now, starting with the ## heading."""


# ── AI-tell scrubber — final human-voice pass over the fully assembled post ──
# Prompt instructions alone have limited effectiveness against ingrained model
# patterns (verified empirically — even with explicit bans, drafts still slip
# into "it's not just X, it's Y" etc). This is a dedicated final pass that hunts
# specifically for AI-tell phrasing across the WHOLE post and fixes only that,
# leaving facts/structure untouched. Structural integrity is verified before
# the result is accepted — see _scrub_ai_tells().
_AI_TELL_SCRUB_PROMPT = """\
You are a line editor. Your only job is to find sentences that sound AI-generated and rewrite THOSE
sentences — nothing else. Every other sentence must stay word-for-word identical.

PATTERNS TO FIND AND FIX (rewrite only sentences containing these):
- "It's not just X, it's Y" in ANY form — including when split across two sentences by a period, e.g.
  "This isn't just about X, though that's part of it. It's really about Y." Treat both sentences as ONE
  violation and rewrite them TOGETHER into a single direct statement of the point.
- "The result?" or "The takeaway?" used as a rhetorical mini-reveal
- Sentences starting with Moreover, Furthermore, Additionally, In addition — cut the filler word,
  keep the rest of the sentence, capitalize what follows
- Any of: delve, tapestry, realm, navigate, harness, leverage (as a verb), robust, seamless, holistic,
  game-changer, revolutionize, unprecedented, boundless, intricate, multifaceted, underscore(s),
  testament to, in essence, at its core, when it comes to, in conclusion, it's worth noting,
  it's important to note, needless to say, cutting-edge, transformative, paradigm, synergy, elevate,
  empower, foster, embark, unveil, groundbreaking, ever-evolving
- A paragraph that closes with a tidy one-line restatement of what it just said
- Three consecutive sentences with near-identical length and grammatical shape — vary at least one

When you rewrite a flagged sentence: keep the same facts and similar length, just change the phrasing
so it sounds like a specific person wrote it, not a summary engine.

DO NOT touch: headings (##), code blocks (```...```), [IMAGE: ...] placeholders, ### Sources lists,
or any sentence that doesn't match a pattern above. Do not add or remove sections.

BLOG POST:
{blog}

Return the FULL post with only the flagged sentences rewritten — same length, same structure, no
commentary, no code fence wrapper around your answer."""


async def _scrub_ai_tells(blog_md: str, invoke_fn) -> str:
    """Rewrite only the AI-sounding sentences in the fully assembled post. Verifies heading order,
    code-fence count, and image-placeholder count match before accepting — silently keeps the
    original on any mismatch, timeout, or error, so a polish pass can never corrupt the post."""
    if len(blog_md) > 16000:
        return blog_md   # too long to safely round-trip in one call — skip rather than risk corruption
    try:
        scrubbed = await invoke_fn([
            SystemMessage(content=_AI_TELL_SCRUB_PROMPT.format(blog=blog_md)),
            HumanMessage(content="Return the corrected post now."),
        ])
        scrubbed = scrubbed.strip()
        scrubbed = re.sub(r'^```(?:markdown)?\s*\n', '', scrubbed)
        scrubbed = re.sub(r'\n```\s*$', '', scrubbed)

        orig_headings = re.findall(r'^#{1,3}\s+.+$', blog_md, re.MULTILINE)
        new_headings  = re.findall(r'^#{1,3}\s+.+$', scrubbed, re.MULTILINE)
        length_ok = 0.7 * len(blog_md) <= len(scrubbed) <= 1.3 * len(blog_md)
        struct_ok = (orig_headings == new_headings
                     and blog_md.count('```') == scrubbed.count('```')
                     and len(re.findall(r'\[IMAGE:.*?\]', blog_md)) == len(re.findall(r'\[IMAGE:.*?\]', scrubbed)))

        if length_ok and struct_ok:
            return scrubbed
        print("[ai-tell-scrub] structural mismatch, keeping original draft")
    except Exception as exc:
        print(f"[ai-tell-scrub] failed, keeping original: {exc}")
    return blog_md


# "It's not just X, it's Y" survives the whole-document scrub even after explicit bans — verified live,
# it's one of the most deeply-trained rhetorical patterns and the model keeps regenerating structurally
# identical alternatives. Deterministic fallback: regex finds CANDIDATE zones (over-inclusive on purpose,
# matches the same-sentence comma form AND the split-sentence period form), then a narrow single-excerpt
# LLM call decides — with full context — whether it's a real violation or an innocent lookalike (e.g.
# "applies not just to startups but also enterprises" is fine and must be left untouched).
_NOT_JUST_RE = re.compile(
    r"[A-Z][^.!?\n]*?\b(?:isn't|is not|wasn't|was not)\s+(?:just|only)\b[^.!?\n]*?[.!?]"
    r"(?:\s+[A-Z][^.!?\n]*?[.!?])?",
    re.IGNORECASE,
)

_NOT_JUST_FIX_PROMPT = """\
Read this excerpt from a blog post:

\"\"\"{excerpt}\"\"\"

If this excerpt uses the cliché AI-writing move "X isn't just A, it's B" — downplaying one thing to
dramatically reveal a "real" point — REWRITE it as ONE direct sentence stating the actual point plainly,
with no "isn't just" framing at all, no matter how the rewrite phrases it.

If this excerpt does NOT use that move — e.g. an innocent sentence like "this applies not just to
startups but also to enterprises" with no dramatic reveal — return it EXACTLY unchanged.

Return ONLY the final text. No commentary, no surrounding quotes."""


async def _kill_not_just_construction(blog_md: str, invoke_fn) -> str:
    """Deterministic-detection + narrow-scope-fix pass for the one construction that survives the
    whole-document scrub. Each candidate is fixed in isolation with full sentence context, so the model
    can tell a real cliché apart from a harmless lookalike instead of guessing across the whole post."""
    matches = list(_NOT_JUST_RE.finditer(blog_md))
    if not matches:
        return blog_md
    result = blog_md
    for m in reversed(matches):   # right-to-left so earlier match offsets stay valid as we splice
        excerpt = m.group(0)
        try:
            fixed = await invoke_fn([HumanMessage(content=_NOT_JUST_FIX_PROMPT.format(excerpt=excerpt))])
            fixed = fixed.strip().strip('"')
            if fixed and 0.3 * len(excerpt) <= len(fixed) <= 1.5 * len(excerpt):
                result = result[:m.start()] + fixed + result[m.end():]
        except Exception as exc:
            print(f"[not-just-fix] skipped one candidate: {exc}")
    return result


# ── Chart detection + generation (from 11_multiagent pattern) ────────────────
_CHART_PROMPT = """\
You are a data visualization agent. Read the research data and blog sections below.

Topic: {topic}
Research data: {research}
Blog sections: {sections}

Task: Find numeric data that would make a useful chart for the blog post (comparisons, growth,
distributions, rankings). Generate Mermaid chart code for up to 2 charts.

Return ONLY this JSON:
{{
  "has_charts": true,
  "charts": [
    {{
      "section_heading": "## Exact Section Heading to inject chart after",
      "caption": "Short chart caption",
      "mermaid": "xychart-beta\\n  title \\"Chart Title\\"\\n  x-axis [\\"A\\", \\"B\\", \\"C\\"]\\n  y-axis \\"Value\\"\\n  bar [10, 20, 30]"
    }}
  ]
}}

If no meaningful numeric data exists: {{"has_charts": false, "charts": []}}

RULES:
- Only chart data that is actually in the research (no made-up numbers).
- Use xychart-beta for bar/line charts, pie for distributions.
- Keep chart titles short (under 50 chars).
- Escape all quotes inside mermaid strings with backslash."""


# ── Visuals prompt ────────────────────────────────────────────────────────────
_VISUAL_PROMPT = """\
This blog post has [IMAGE: description] placeholders. Generate Pollinations.ai image prompts.

Blog title: {title}
Content snippet: {snippet}

Return ONLY this JSON:
{{
  "cover": "photorealistic cover image — professional, eye-catching, relevant to topic, no text",
  "sections": [
    {{"placeholder": "[IMAGE: exact text]", "prompt": "detailed image prompt, no text in image"}}
  ]
}}

Rules:
- Specific style (photorealistic / digital art / illustration / 3D render)
- Mention colors, mood, composition
- NEVER include text/words/letters in the image
- Each prompt under 200 chars"""


# ── SEO prompt ────────────────────────────────────────────────────────────────
_SEO_PROMPT = """\
Optimise this blog post for search and social sharing. The topic may be anything — tech, sports, business, culture, science, entertainment.

Topic: {topic}
Title: {title}
Content preview: {preview}

Return ONLY this JSON:
{{
  "title": "final SEO title under 65 chars — specific, curiosity-gap, primary keyword near the front",
  "meta_description": "compelling meta under 155 chars — lead with the benefit or the hook, not the topic label",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "slug": "url-slug-with-hyphens",
  "reading_time": integer_minutes
}}

Tag rules — pick tags relevant to the ACTUAL topic (not forced tech tags):
- Lowercase alphanumeric only, no hyphens, no spaces
- For sports: football, worldcup, soccer, sports, fifa
- For tech: ai, python, webdev, machinelearning
- For business: startup, marketing, productivity
- Mix 1 broad tag + 3 specific ones"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _img_url(prompt: str, seed: int = 1, w: int = 1200, h: int = 630) -> str:
    from urllib.parse import quote
    return (f"https://image.pollinations.ai/prompt/{quote(prompt[:250])}"
            f"?width={w}&height={h}&nologo=true&seed={seed}&enhance=true")


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def _extract_section(md: str, heading: str) -> str:
    """Pull the text of one ## section from a full markdown doc."""
    lines = md.split('\n')
    in_section = False
    out = []
    for line in lines:
        if line.strip() == heading.strip():
            in_section = True
        elif in_section and line.startswith('## ') and line.strip() != heading.strip():
            break
        if in_section:
            out.append(line)
    return '\n'.join(out)


def _replace_section(md: str, heading: str, new_content: str) -> str:
    """Replace one ## section in a markdown doc with new_content."""
    lines  = md.split('\n')
    result = []
    skip   = False
    replaced = False
    for line in lines:
        if line.strip() == heading.strip():
            skip = True
            replaced = True
            result.extend(new_content.strip().split('\n'))
            result.append('')
        elif skip and line.startswith('## ') and line.strip() != heading.strip():
            skip = False
        if not skip:
            result.append(line)
    if not replaced:
        result.extend(new_content.strip().split('\n'))
    return '\n'.join(result)


# ── Map-Reduce: parallel section writer ──────────────────────────────────────

async def _write_sections_parallel(
    sections: list, topic: str, title: str, style: str,
    research_text: str, invoke_fn: Callable
) -> List[str]:
    """Write all blog sections concurrently — inspired by 10_map_reduce Send() pattern."""
    async def write_one(section: dict) -> str:
        # Use per-section research if available, fall back to global research
        section_research = section.get('section_research') or research_text
        prompt = _SECTION_WRITER_PROMPT.format(
            style=style, topic=topic, title=title,
            heading=section.get('heading', 'Section'),
            description=section.get('description', section.get('heading', '')),
            key_points=', '.join(section.get('key_points', [])),
            angle=section.get('angle', 'find the non-obvious angle'),
            needs_code=section.get('needs_code', False),
            needs_diagram=section.get('needs_diagram', False),
            research=section_research[:4000],
        )
        try:
            return await invoke_fn([HumanMessage(content=prompt)])
        except Exception as e:
            return f"## {section.get('heading', 'Section')}\n\n*(Section unavailable: {e})*\n"

    tasks = [asyncio.create_task(write_one(s)) for s in sections]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parts = []
    for section, result in zip(sections, results):
        if isinstance(result, Exception):
            parts.append(f"## {section.get('heading', 'Section')}\n\n*(Error)*\n")
        else:
            parts.append(result.strip())
    return parts


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_content_pipeline(
    topic: str,
    style: str,
    provider: str,
    model: Optional[str],
    api_base: Optional[str],
    tavily_api_key: str,
    event_cb: Callable[[str, dict], Awaitable[None]],
    euron_key: Optional[str] = None,
):
    """
    7-phase pipeline — each phase emits SSE events via event_cb.

    Improvements over v1:
    • Map-Reduce parallel writing (inspired by 10_map_reduce)
    • Quality evaluator → rewrite loop (inspired by 4_X_post_generator)
    • Auto chart generation (inspired by 11_multiagent)
    • HITL quality report returned in 'complete' event for frontend review
    """
    loop = asyncio.get_event_loop()
    llm  = get_llm_resilient(provider, model, api_base=api_base, euron_key=euron_key)

    async def invoke(messages: list) -> str:
        resp = await loop.run_in_executor(None, lambda: llm.invoke(messages))
        return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    # ── Phase 1: Research ─────────────────────────────────────────────────────
    await event_cb("phase_start", {"phase": "research",
                                    "msg": "🔍 Generating search queries & searching the web…"})

    async def _tavily_search(tv, q: str, loop) -> list:
        try:
            res = await loop.run_in_executor(
                None, lambda: tv.search(q, max_results=4, search_depth="advanced"))
            return res.get("results", [])
        except Exception:
            return []

    def _build_research_text(results_batches: list, max_per_source: int = 800) -> tuple:
        parts = []
        seen_urls: set = set()
        for batch in results_batches:
            for r in batch:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    snippet = r.get("content", "")[:max_per_source]
                    title   = r.get("title", url)
                    parts.append(f"SOURCE: {title}\nURL: {url}\n{snippet}")
        return "\n\n---\n\n".join(parts), len(parts)

    try:
        from tavily import TavilyClient
        tv = TavilyClient(api_key=tavily_api_key)

        # Generate diverse, targeted queries (breadth-first)
        q_raw = await invoke([HumanMessage(
            content=f"""Generate 6 targeted search queries for a {style} blog post about: {topic}

Rules:
- Cover different angles: background/context, key facts & stats, notable examples or stories, expert opinions, recent developments, controversies or debates
- Include year "2025" or "2026" in at least 2 queries to get the freshest data
- Match the domain — for sports use sports terminology, for tech use technical terms, for business use industry language
- Make each query specific enough to return authoritative sources (news sites, official reports, expert analysis)
- Return one query per line, nothing else."""
        )])
        queries = [q.strip("•-– 0123456789.)").strip()
                   for q in q_raw.split("\n") if q.strip()][:6]

        search_tasks = [asyncio.create_task(_tavily_search(tv, q, loop)) for q in queries]
        all_results  = await asyncio.gather(*search_tasks)
        research_text, source_count = _build_research_text(all_results)
        if not research_text:
            research_text = f"General knowledge about {topic}."
    except Exception as e:
        research_text = f"Proceeding with general knowledge about {topic}. ({e})"
        source_count  = 0

    await event_cb("phase_done", {"phase": "research",
                                   "msg": f"✅ Research complete — {source_count} sources"})

    # ── Phase 2: Outline ──────────────────────────────────────────────────────
    await event_cb("phase_start", {"phase": "outline", "msg": "📐 Planning structure…"})
    try:
        outline_raw = await invoke([
            SystemMessage(content=_OUTLINE_PROMPT.format(
                style=style, topic=topic, research=research_text[:3500])),
            HumanMessage(content="Return the outline JSON now."),
        ])
        outline = _parse_json(outline_raw)
    except Exception:
        outline = {
            "title": topic,
            "intro_hook": f"Here's what most people miss about {topic}.",
            "sections": [
                {"heading": h, "key_points": [], "needs_code": False,
                 "needs_diagram": False, "image_description": f"illustration about {h}"}
                for h in ["What It Is and Why It Matters", "Core Concepts",
                           "How It Works Step by Step", "Practical Examples",
                           "Common Mistakes to Avoid"]
            ],
        }

    blog_title     = outline.get("title", topic)
    intro_hook     = outline.get("intro_hook", "")
    sections_plan  = outline.get("sections", [])
    section_list   = [s["heading"] for s in sections_plan]

    await event_cb("phase_done", {
        "phase": "outline", "msg": f"✅ Outline — {len(sections_plan)} sections",
        "title": blog_title, "sections": section_list,
    })

    # ── Phase 3: Parallel Write (Map-Reduce) ──────────────────────────────────
    n = len(sections_plan)
    await event_cb("phase_start", {"phase": "writing",
                                    "msg": f"✍️ Researching & writing {n} sections in parallel…"})
    try:
        # Enrich each section with targeted per-section research (like 8000 does)
        has_tavily = source_count > 0
        async def enrich_section(section: dict) -> dict:
            if not has_tavily:
                return section
            heading = section.get("heading", "")
            kp      = ", ".join(section.get("key_points", []))
            try:
                from tavily import TavilyClient as _TV
                _tv = _TV(api_key=tavily_api_key)
                # 2 targeted queries per section
                sq_raw = await invoke([HumanMessage(content=
                    f"Generate 2 specific search queries to find expert-level facts, stats, and examples for "
                    f"a blog section titled '{heading}' about: {topic}\n"
                    f"Key points to cover: {kp}\n"
                    "Return one query per line, nothing else.")])
                sq = [q.strip("•-– 0123456789.)").strip()
                      for q in sq_raw.split("\n") if q.strip()][:2]
                sr = await asyncio.gather(*[asyncio.create_task(_tavily_search(_tv, q, loop)) for q in sq])
                sec_research, _ = _build_research_text(list(sr))
                if sec_research:
                    section = dict(section)
                    section["section_research"] = sec_research
            except Exception:
                pass
            return section

        enrich_tasks = [asyncio.create_task(enrich_section(s)) for s in sections_plan]
        sections_plan_enriched = list(await asyncio.gather(*enrich_tasks))

        # Write intro, all sections, and conclusion concurrently
        intro_task      = asyncio.create_task(invoke([HumanMessage(content=
            _INTRO_PROMPT.format(style=style, topic=topic, title=blog_title,
                                  hook=intro_hook, section_list=", ".join(section_list)))]))
        section_task    = asyncio.create_task(
            _write_sections_parallel(sections_plan_enriched, topic, blog_title,
                                     style, research_text, invoke))
        conclusion_task = asyncio.create_task(invoke([HumanMessage(content=
            _CONCLUSION_PROMPT.format(style=style, topic=topic,
                                      section_list=", ".join(section_list)))]))

        intro_md, section_mds, conclusion_md = await asyncio.gather(
            intro_task, section_task, conclusion_task)

        # Assemble: title → intro → sections → conclusion
        blog_md = (f"# {blog_title}\n\n"
                   + intro_md.strip() + "\n\n"
                   + "\n\n".join(section_mds) + "\n\n"
                   + conclusion_md.strip())
    except Exception as e:
        blog_md = f"# {blog_title}\n\nError during writing: {e}"

    word_count = len(blog_md.split())
    await event_cb("phase_done", {"phase": "writing",
                                   "msg": f"✅ Written — {word_count:,} words across {n} sections"})

    # ── Phase 4: Quality Check + Rewrite Loop ────────────────────────────────
    # Inspired by 4_X_post_generator evaluate → optimize → re-evaluate loop
    await event_cb("phase_start", {"phase": "quality",
                                    "msg": "🧐 Evaluating quality — scoring each section…"})
    quality_report = {}
    MAX_QUALITY_ITER = 2
    for q_iter in range(MAX_QUALITY_ITER):
        try:
            eval_raw = await invoke([
                SystemMessage(content=_QUALITY_EVAL_PROMPT.format(
                    blog=blog_md[:6000])),
                HumanMessage(content="Return the quality evaluation JSON now."),
            ])
            quality_report = _parse_json(eval_raw)
        except Exception:
            quality_report = {"overall_score": 4.0, "sections": [], "needs_rewrite": []}

        needs_rewrite = quality_report.get("needs_rewrite", [])
        if not needs_rewrite:
            break

        await event_cb("phase_progress", {
            "phase": "quality",
            "msg": f"🔁 Rewriting {len(needs_rewrite)} weak section(s) (pass {q_iter+1})…",
            "score": quality_report.get("overall_score", 0),
            "needs_rewrite": needs_rewrite,
        })

        # Rewrite weak sections in parallel
        async def rewrite_one(heading: str) -> tuple:
            sec_content = _extract_section(blog_md, heading)
            # Find the issue description from quality report
            issue = next(
                (s.get("issue", "Needs more depth and examples")
                 for s in quality_report.get("sections", [])
                 if s.get("heading", "").strip() == heading.strip()),
                "Needs more depth, clearer examples, and better structure"
            )
            rewritten = await invoke([
                SystemMessage(content=_SECTION_REWRITE_PROMPT.format(
                    issue=issue, section_content=sec_content,
                    topic=topic, research=research_text[:2000])),
                HumanMessage(content="Write the improved section now."),
            ])
            return heading, rewritten

        rewrite_tasks   = [asyncio.create_task(rewrite_one(h)) for h in needs_rewrite]
        rewrite_results = await asyncio.gather(*rewrite_tasks, return_exceptions=True)

        for item in rewrite_results:
            if isinstance(item, Exception):
                continue
            heading, new_content = item
            blog_md = _replace_section(blog_md, heading, new_content)

    overall_score = quality_report.get("overall_score", 4.0)

    # Final human-voice pass — hunts for AI-tell phrasing across the whole post
    await event_cb("phase_progress", {"phase": "quality",
                                       "msg": "🧹 Scrubbing AI-sounding phrases…"})
    blog_md = await _scrub_ai_tells(blog_md, invoke)
    blog_md = await _kill_not_just_construction(blog_md, invoke)

    await event_cb("phase_done", {
        "phase": "quality",
        "msg": f"✅ Quality score: {overall_score:.1f}/5 — blog polished",
        "quality_report": quality_report,
        "overall_score": overall_score,
    })

    # ── Phase 5: Charts (from 11_multiagent chart-generator pattern) ──────────
    await event_cb("phase_start", {"phase": "charts",
                                    "msg": "📊 Scanning data for auto-charts…"})
    charts_injected = 0
    try:
        chart_raw = await invoke([
            SystemMessage(content=_CHART_PROMPT.format(
                topic=topic,
                research=research_text[:3000],
                sections=", ".join(section_list),
            )),
            HumanMessage(content="Return chart detection JSON now."),
        ])
        chart_data = _parse_json(chart_raw)

        if chart_data.get("has_charts"):
            for ch in chart_data.get("charts", [])[:2]:
                target_heading = ch.get("section_heading", "")
                mermaid_code   = ch.get("mermaid", "")
                caption        = ch.get("caption", "Data chart")
                if not mermaid_code:
                    continue
                chart_block = f'\n\n```mermaid\n{mermaid_code}\n```\n*{caption}*\n'
                # Inject after the target section heading
                if target_heading and target_heading in blog_md:
                    blog_md = blog_md.replace(
                        target_heading,
                        target_heading + chart_block, 1)
                    charts_injected += 1
    except Exception:
        pass

    msg = (f"✅ {charts_injected} chart(s) injected" if charts_injected
           else "✅ No numeric data found — skipped charts")
    await event_cb("phase_done", {"phase": "charts", "msg": msg,
                                   "charts_count": charts_injected})

    # ── Phase 6: Visuals (Pollinations.ai images) ─────────────────────────────
    await event_cb("phase_start", {"phase": "visuals",
                                    "msg": "🎨 Generating AI cover + section images…"})
    try:
        vis_raw = await invoke([
            SystemMessage(content=_VISUAL_PROMPT.format(
                title=blog_title, snippet=blog_md[:2500])),
            HumanMessage(content="Return image prompts JSON now."),
        ])
        vis = _parse_json(vis_raw)
    except Exception:
        vis = {"cover": f"professional illustration about {topic}, no text", "sections": []}

    cover_url = _img_url(vis.get("cover", f"illustration about {topic}"), seed=1)

    # Replace [IMAGE: ...] placeholders with Pollinations URLs
    final_md = blog_md
    for idx, item in enumerate(vis.get("sections", []), start=10):
        ph     = item.get("placeholder", "")
        prompt = item.get("prompt", f"illustration {idx}")
        url    = _img_url(prompt, seed=idx, w=800, h=450)
        if ph and ph in final_md:
            alt = ph[8:-1].strip() if ph.startswith("[IMAGE:") else ph
            final_md = final_md.replace(ph, f"\n![{alt}]({url})\n", 1)
    # Remove any leftover [IMAGE: ...] placeholders that weren't matched
    final_md = re.sub(r'\[IMAGE:[^\]]*\]', '', final_md)

    await event_cb("phase_done", {"phase": "visuals",
                                   "msg": "✅ Images generated",
                                   "cover_url": cover_url})

    # ── Phase 7: SEO ──────────────────────────────────────────────────────────
    await event_cb("phase_start", {"phase": "seo",
                                    "msg": "🔎 Optimising title, tags & meta description…"})
    try:
        seo_raw = await invoke([
            SystemMessage(content=_SEO_PROMPT.format(
                topic=topic, title=blog_title, preview=final_md[:350])),
            HumanMessage(content="Return SEO JSON now."),
        ])
        seo = _parse_json(seo_raw)
    except Exception:
        seo = {}

    raw_tags    = seo.get("tags", ["webdev", "tutorial", "ai", "programming"])
    seo["tags"] = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in raw_tags]
    seo["tags"] = [t for t in seo["tags"] if t][:4]
    seo.setdefault("title", blog_title)
    seo.setdefault("meta_description", f"Learn about {topic} in this practical guide.")
    seo.setdefault("slug", re.sub(r"[^a-z0-9]+", "-", topic.lower())[:60])
    seo.setdefault("reading_time", max(1, word_count // 200))

    await event_cb("phase_done", {"phase": "seo", "msg": "✅ SEO optimised", "seo": seo})

    # ── Complete ──────────────────────────────────────────────────────────────
    await event_cb("complete", {
        "markdown":       final_md,
        "cover_url":      cover_url,
        "title":          seo["title"],
        "seo":            seo,
        "word_count":     word_count,
        "outline":        outline,
        "quality_report": quality_report,   # HITL: frontend shows this for user review
        "charts_count":   charts_injected,
    })


# ── HITL Refinement pipeline ──────────────────────────────────────────────────
# Called by /api/content/refine when user sends feedback from the quality report

_REFINE_PROMPT = """\
A user reviewed this blog post and provided feedback. Apply their requested changes.

Current blog post:
{blog}

User feedback / edit instructions:
{feedback}

Topic: {topic}

RULES:
- Apply the feedback faithfully — don't second-guess the user.
- Keep the overall structure (headings, sections).
- Keep all images and Mermaid diagrams unless instructed to remove them.
- Fix only what the feedback asks. Don't rewrite everything.
- Return the COMPLETE updated blog post in markdown, starting with # Title."""


async def run_refine_pipeline(
    topic: str,
    markdown: str,
    feedback: str,
    provider: str,
    model: Optional[str],
    api_base: Optional[str],
    event_cb: Callable[[str, dict], Awaitable[None]],
    euron_key: Optional[str] = None,
):
    """
    Lightweight HITL refinement — user provides feedback, blog gets updated.
    Skips Research / Outline / Charts / Visuals — only refines the writing.
    """
    loop = asyncio.get_event_loop()
    llm  = get_llm_resilient(provider, model, api_base=api_base, euron_key=euron_key)

    async def invoke(messages: list) -> str:
        resp = await loop.run_in_executor(None, lambda: llm.invoke(messages))
        return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    await event_cb("phase_start", {"phase": "refine",
                                    "msg": "✍️ Applying your feedback…"})
    try:
        refined = await invoke([
            SystemMessage(content=_REFINE_PROMPT.format(
                blog=markdown[:8000], feedback=feedback, topic=topic)),
            HumanMessage(content="Return the full updated blog post now."),
        ])
    except Exception as e:
        await event_cb("error", {"message": f"Refinement failed: {e}"})
        return

    await event_cb("phase_done", {"phase": "refine", "msg": "✅ Feedback applied"})

    # Re-run quality check on refined version
    await event_cb("phase_start", {"phase": "quality", "msg": "🧐 Re-evaluating quality…"})
    try:
        eval_raw = await invoke([
            SystemMessage(content=_QUALITY_EVAL_PROMPT.format(blog=refined[:6000])),
            HumanMessage(content="Return quality JSON."),
        ])
        quality_report = _parse_json(eval_raw)
    except Exception:
        quality_report = {"overall_score": 4.5, "sections": [], "needs_rewrite": []}

    await event_cb("phase_done", {
        "phase": "quality",
        "msg": f"✅ Quality score: {quality_report.get('overall_score', 4.5):.1f}/5",
        "quality_report": quality_report,
    })

    refined = await _scrub_ai_tells(refined, invoke)
    refined = await _kill_not_just_construction(refined, invoke)
    word_count = len(refined.split())
    await event_cb("complete", {
        "markdown":       refined,
        "word_count":     word_count,
        "quality_report": quality_report,
        "refined":        True,
    })
