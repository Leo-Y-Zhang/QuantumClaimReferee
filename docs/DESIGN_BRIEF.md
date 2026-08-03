# Design Brief — the referee report and terminal output

**Date:** 2026-08-03 · **PRD:** [PRD.md](PRD.md) · **App Flow:** [APP_FLOW.md](APP_FLOW.md)

The only designed surface in this project is text: the `chsh` summary, the self-test
scorecards, and the fixed-width block produced by `referee_report`. There is no GUI,
no TUI, no colour and no interactivity, so the web sections of the standard brief —
breakpoints, touch targets, hover states, motion — are cut rather than answered
emptily. What remains is genuinely designed, because the report is an artefact
intended to be pasted into a paper or a review and read by someone who did not run
the tool.

## Intent

**Reads like a referee's note, not a dashboard.** Factual, unhurried, and visibly
conservative: the reader should be able to see which number decided the verdict,
which numbers are context, and what the result is conditional on, without knowing
the codebase.

What it must never feel like: a scoreboard. No celebration of a `CERTIFIED`, no
emphasis that makes a p-value look better than it is, nothing that reads as a
marketing claim about the device under test. The tool's only asset is that its
output is believable; anything that looks like persuasion spends that asset.

## Who is looking at it

Two readers, and the second one is the reason for the rules:

1. The person who ran the command, in a terminal, deciding whether to spend more
   shots.
2. A **reviewer or coauthor months later**, reading the block pasted into a
   manuscript or an internal review, with no access to the command that produced it
   and no reason to trust the person who pasted it.

Reader 2 is why the report carries versions and an input hash, and why it can carry
no timestamp: it must be reproducible from the inputs alone.

## Precedents

- **`pytest` output** — the exit code *is* the result, and the summary line is
  unambiguous without colour. `qcref chsh` copies this directly: exit 0 iff
  CERTIFIED.
- **A `diff --stat` / `git log` block** — fixed-width, alignable, quotable in a
  monospace context, and identical wherever it is displayed.
- **A statistical package's `summary()`** (R's `lm`, statsmodels) — the estimate and
  its uncertainty printed together, always, so a point estimate never travels alone.

## Anti-patterns for this project

Specific enough to enforce in review:

- **No colour, no ANSI escapes, no Unicode box-drawing.** Colour would be lost the
  moment the output is pasted, and a red/green certification is exactly the
  "scoreboard" failure. Full status words carry the meaning instead.
- **No timestamps and no randomness in the report.** Determinism is a designed
  property with a test behind it (`test_report_is_deterministic`); a "generated at"
  line would break byte-identical rendering for no reader benefit.
- **No progress bars or spinners.** Everything shipped finishes in about a second,
  and a spinner would corrupt piped output.
- **Never print the naive contrast p-value without its label.** It appears only as
  `(for contrast only)` or `<- INVALID (inflated)`. An unlabelled naive p-value in a
  transcript is the exact failure this project exists to prevent.
- **Never abbreviate a status.** `UNDERPOWERED` is not `UP`; a reader must not need
  a legend.
- **No emoji, no ASCII art, no banners beyond the single rule line.**

## Typography and layout

It is all monospace by assumption, so "type" means alignment.

- **Rule line:** `=` × 68, opening and closing the report block.
- **Columns:** hypothesis name left-aligned in 34 characters, `raw p` and `adj p`
  right-aligned in 12 each, then two spaces and the status. p-values in the report
  are `%.2e`; in the `chsh` summary they are `%.3e`. Fixed exponent notation, never
  variable-width floats, so columns cannot ragged.
- **Label column in summaries:** the key is padded to a fixed width and followed by
  ` : ` so the values line up as a scannable column.
- **Indentation carries hierarchy:** one leading space for report body lines, three
  for a hypothesis's estimate and for caveat bullets. Nothing else nests.
- **Width:** the widest line in a rendered report is 80 characters, so it fits a
  default terminal and a single manuscript column.

**Known deviation, not a rule:** the header rule under the column titles is
58 + 2 + 16 characters wide while several body rows are 70–80, and the caveat lines
overrun the 68-character bar. It is legible but not aligned to a single measure. If
this surface is revisited, pick one measure (80) and hold every element to it. It is
recorded here rather than quietly tolerated.

## Emphasis: the only three devices

There is no bold, so emphasis is structural, and it is deliberately scarce:

1. **Position** — the headline `VERDICT:` is the first content line of the report,
   and `status` is the first line of a `chsh` summary.
2. **The `<-` pointer** — used for exactly three things: `<- use this` on the
   memory-robust p-value, `<- INVALID (inflated)` on the naive rate, and
   `<- bounded` / `<- CHECK` on an adversary scorecard. Adding a fourth use would
   dilute all three.
3. **Parenthetical demotion** — `(for contrast only)`, `(Azuma, loose)`,
   `(classical bound 0.7500)`: context is marked as context on the same line.

## States

The states of this surface are the four verdicts plus a usage error.

| State | What is on screen | Design obligation |
|---|---|---|
| `CERTIFIED` | verdict, p-values, declared assumptions | must not look like a celebration; the caveats block is still printed |
| `UNDERPOWERED` | as above, plus `rounds for power` | must state the assumption that the observed win rate persists, on the same line as the number |
| `NOT_CERTIFIED` | as above | must not be confusable with an error; it is a judgement |
| `ASSUMPTIONS_UNMET` | as above, plus `reason` | must name the unmet assumption and, where known, the likely cause |
| usage error | one-line `usage:` and the message | never a traceback |

Every state ends with the caveats block when rendered as a report — including a
certified one. The caveats are not a disclaimer to be skipped; they are the
conditions under which the verdict is true, which is why they are titled
`CAVEATS (read before citing)` and printed last, where the eye lands after the
result.

## Accessibility floor

Reduced to what applies to plain text, and all of it holds today:

- **Colour is never the only signal, because colour is never used.**
- Output is pure ASCII on stdout, so it survives a pipe, a log, a CI transcript and
  a screen reader unchanged.
- No cursor control, no redraw, no alternate screen buffer — the transcript is the
  whole interface, and scrollback is never invalidated.
- No interactive prompt, so the tool is fully operable without a TTY.
- Meaning is in words, not glyphs; the sole non-alphabetic marker (`<-`) always
  points at text that says the same thing.

## Done means

- [x] The number that decided the verdict is identifiable without reading the code.
- [x] Every non-decisive number is labelled as context on its own line.
- [x] The report renders byte-identical from identical inputs, and a test enforces it.
- [x] The report names the code and dependency versions that produced it.
- [x] Output survives copy-paste into a plain-text document with no loss of meaning.
- [ ] One consistent measure across the report block (see the known deviation above).
