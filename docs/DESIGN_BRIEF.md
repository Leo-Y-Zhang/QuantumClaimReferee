# Design Brief — the referee report and terminal output

The only designed surface here is text: the `chsh` summary, the self-test
scorecards, and the fixed-width block produced by `referee_report`. No GUI, no
TUI, no colour, no interactivity. What remains is genuinely designed, because the
report is an artefact meant to be pasted into a paper or a review and read by
someone who did not run the tool.

[PRD.md](PRD.md) · [APP_FLOW.md](APP_FLOW.md)

## The second reader sets the rules

Two people look at this. The one who ran the command, in a terminal, deciding
whether to spend more shots. And a **reviewer or coauthor months later**, reading
the block pasted into a manuscript or an internal review, with no access to the
command that produced it and no particular reason to trust the person who pasted
it.

The second reader is why the report carries versions and an input hash, and why
it can carry no timestamp: it has to be reproducible from the inputs alone.

## A referee's note, not a dashboard

Factual, unhurried, visibly conservative. The reader should be able to see which
number decided the verdict, which numbers are context, and what the result is
conditional on, without knowing the codebase.

It must never feel like a scoreboard. No celebration of a `CERTIFIED`, no
emphasis that makes a p-value look better than it is, nothing that reads as a
marketing claim about the device under test. The tool's only asset is that its
output is believable, and anything that looks like persuasion spends that asset.

## Enforceable in review

- **No colour, no ANSI escapes, no Unicode box-drawing.** Colour is lost the
  moment output is pasted, and a red/green certification is exactly the
  scoreboard failure. Full status words carry the meaning instead.
- **No timestamps and no randomness in the report.** Determinism is a designed
  property with a test behind it (`test_report_is_deterministic`); a "generated
  at" line would break byte-identical rendering for no reader benefit.
- **No progress bars or spinners.** Everything shipped finishes in about a
  second, and a spinner would corrupt piped output.
- **Never print the naive contrast p-value without its label.** It appears only
  as `(for contrast only)` or `<- INVALID (inflated)`. An unlabelled naive
  p-value in a transcript is the exact failure this project exists to prevent.
- **Never abbreviate a status.** `UNDERPOWERED` is not `UP`; a reader must not
  need a legend.
- **No emoji, no ASCII art, no banners beyond the single rule line.**

## Borrowed from

`pytest` output, where the exit code *is* the result and the summary line is
unambiguous without colour. `qcref chsh` copies this directly: exit 0 iff
CERTIFIED.

A `diff --stat` or `git log` block — fixed-width, alignable, quotable in a
monospace context, and identical wherever it is displayed.

A statistical package's `summary()`, as in R's `lm` or statsmodels, where the
estimate and its uncertainty are printed together always, so a point estimate
never travels alone.

## Three devices for emphasis, and no more

There is no bold available, so emphasis is structural and deliberately scarce.

**Position.** The headline `VERDICT:` is the first content line of the report,
and `status` is the first line of a `chsh` summary.

**The `<-` pointer**, used for exactly three things: `<- use this` on the
memory-robust p-value, `<- INVALID (inflated)` on the naive rate, and
`<- bounded` / `<- CHECK` on an adversary scorecard. A fourth use would dilute
all three.

**Parenthetical demotion** — `(for contrast only)`, `(Azuma, loose)`,
`(classical bound 0.7500)`. Context is marked as context on the same line.

## Alignment, which is all "type" means here

Everything is monospace by assumption.

The rule line is `=` × 68, opening and closing the report block. Columns put the
hypothesis name left-aligned in 34 characters, then `raw p` and `adj p`
right-aligned in 12 each, then two spaces and the status. p-values are `%.2e` in
the report and `%.3e` in the `chsh` summary — fixed exponent notation, never
variable-width floats, so columns cannot go ragged. In summaries the label column
is padded to a fixed width and followed by ` : `, so values line up as a
scannable column. Indentation carries hierarchy: one leading space for report
body lines, three for a hypothesis's estimate and for caveat bullets, and nothing
else nests. The widest line in a rendered report is 80 characters, so it fits a
default terminal and a single manuscript column.

One known deviation, recorded rather than quietly tolerated: the header rule
under the column titles is 58 + 2 + 16 characters wide while several body rows
are 70–80, and the caveat lines overrun the 68-character bar. It is legible but
not aligned to a single measure. If this surface is revisited, pick one measure —
80 — and hold every element to it.

## States

The four verdicts, plus a usage error.

| State | What is on screen | Design obligation |
|---|---|---|
| `CERTIFIED` | verdict, p-values, declared assumptions | must not look like a celebration; the caveats block is still printed |
| `UNDERPOWERED` | as above, plus `rounds for power` | must state the assumption that the observed win rate persists, on the same line as the number |
| `NOT_CERTIFIED` | as above | must not be confusable with an error; it is a judgement |
| `ASSUMPTIONS_UNMET` | as above, plus `reason` | must name the unmet assumption and, where known, the likely cause |
| usage error | one-line `usage:` and the message | never a traceback |

Every state ends with the caveats block when rendered as a report, a certified
one included. The caveats are not a disclaimer to be skipped; they are the
conditions under which the verdict is true. That is why they are titled
`CAVEATS (read before citing)` and printed last, where the eye lands after the
result.

## Accessibility floor

All of it holds today, and there is not much of it, because there is not much
surface.

Colour is never the only signal, because colour is never used. Output is pure
ASCII on stdout, so it survives a pipe, a log, a CI transcript and a screen
reader unchanged. No cursor control, no redraw, no alternate screen buffer — the
transcript is the whole interface and scrollback is never invalidated. No
interactive prompt, so the tool is fully operable without a TTY. And meaning is
in words rather than glyphs; the sole non-alphabetic marker, `<-`, always points
at text that says the same thing.

## Done means

- [x] The number that decided the verdict is identifiable without reading the code.
- [x] Every non-decisive number is labelled as context on its own line.
- [x] The report renders byte-identical from identical inputs, and a test enforces it.
- [x] The report names the code and dependency versions that produced it.
- [x] Output survives copy-paste into a plain-text document with no loss of meaning.
- [ ] One consistent measure across the report block (see the known deviation).
