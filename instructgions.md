# Local-change review: second pass

Reviewed 2026-09-13. This status supersedes the first-pass findings and reproduction
outputs. Existing user changes have been preserved; fixes and regression coverage
were added during this pass. No commit or publication was made.

## Original findings: cross-check

1. **Explicit titles overlapping labels:** the prior agent added title clearance
   and the `FontProperties.copy()` fix. The original centered-title examples pass,
   including the longer title already in the tests. Additional title-location
   and padding problems were found and fixed below.
2. **Labels still overlapping after layout:** the bounded refitting implementation
   passes the original array-backed cases at widths 4, 6, 7, and 8 inches.
3. **Compounding font shrink below the minimum:** fitting from saved original
   sizes and moving luminosity to a separate line handles the tested narrow
   frames. Regression tests check the size floor and repeated alignment.
4. **Missing toy-file setup instructions:** explicit `toy_data` metadata now
   drives the prerequisite block independently of shared sample parameters.
   The introductory file-based pages include setup; the arrays example does not.

## Additional issues fixed

### P2: Left/right title locations and duplicate titles

`_clear_title` inspected only `ax.title` (the center title), so a style with
`axes.titlelocation="left"` left the title overlapping the CMS name. When the
style requested a center title but outer rcParams requested a left title,
calling `ax.set_title()` during clearance created another title on the left.
It could also reset explicit title positions from outer `axes.titley`.

The fix moves the existing shared title offset and considers all three title
artists. It preserves their text, font properties, color, alignment, and
explicit vertical positions without calling `set_title()` again. This also
avoids the live-font-properties mutation that prompted this follow-up review.

Regression coverage checks left, center, and right titles under conflicting
outer rcParams, before and after saving and repeated alignment.

### P2: Custom title padding was read from outer rcParams

Clearance runs outside the style context, but read `axes.titlepad` from the
current global settings. A style requesting an 18-point gap could consequently
receive the outer context's 2-point gap.

The fix reads the original offset from the axes and retains that gap across
alignment passes. Its cache is keyed by the title artist so clearing an axes
also discards the old title's padding state. Tests verify the requested gap,
24-point font size, bold weight, color, and explicit title position.

### P2: Twelve stale gallery reference images failed CI comparisons

A full image-comparison run found mismatches in these six examples, each in
light and dark themes:

- `correlation-cms`
- `hist2d-cms`
- `expressions-lhcb`
- `object_vs_event-lhcb`
- `stack_data-lhcb`
- `profile-atlas`

The baseline/result pairs were visually inspected. The differences match the
intended label fitting and settled layout changes. Only these twelve references
were regenerated, retaining the existing RMS tolerance of 2.

## Implementation and checks

- Title clearance: `src/rootfig/plotting/style.py`.
- Regression tests: `tests/test_api.py`. The title/narrow-frame cases use the
  existing array fixture to test rendering independently of file I/O.
- `docs/plotting.md` explains the minimum size, separate luminosity line, and
  preservation of title styling.
- Focused plotting/gallery checks: 102 passed.
- Final full suite, including all gallery image comparisons and the title
  preservation assertions: **801 passed**, **97% coverage**.
- Two warnings report an unavailable ATLAS monospace font and its fallback;
  the corresponding light/dark gallery comparisons passed.
- Ruff lint and formatting, mypy (36 source files), strict documentation build,
  and `git diff --check` passed.
- All twelve regenerated references were also checked pixel-for-pixel against
  the visually inspected results.

Final test command (outside the sandbox, with writable cache paths under `/tmp`):

```bash
uv run --no-sync pytest -n 4 --mpl --cov --cov-report=term \
  --mpl-results-path=/tmp/rootfig-review2-final-results -q
```

## Environment note

Inside the sandbox, even a minimal independent uproot TTree read blocks waiting
for chunk notifications. This explains the previous stalled file-based test
attempts. The suite runs normally outside the sandbox; no application I/O code
was changed to work around that environment issue.
