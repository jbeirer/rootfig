# API reference

## Top level

::: rootfig.plot
::: rootfig.plot2d
::: rootfig.histogram
::: rootfig.api.data.histograms
    options:
      heading: rootfig.histograms
::: rootfig.load
::: rootfig.summarize
::: rootfig.correlation
::: rootfig.cutflow
::: rootfig.efficiency
::: rootfig.profile
::: rootfig.evaluate
::: rootfig.compare
::: rootfig.log_bins
::: rootfig.use_style
::: rootfig.dark_theme

## Descriptions

::: rootfig.Sample
::: rootfig.Group
::: rootfig.Variable
::: rootfig.Cut
::: rootfig.Style
::: rootfig.Systematic
::: rootfig.Bayesian
::: rootfig.ONE_SIGMA

## Batch plotting

::: rootfig.PlotBook
::: rootfig.PlotTask
::: rootfig.ALL
::: rootfig.discover_variables
::: rootfig.model.check_file_stem
::: rootfig.model.safe_file_stem

## Results

::: rootfig.Plot
::: rootfig.Histogram
::: rootfig.Comparison
::: rootfig.Uncertainty
::: rootfig.Summary
::: rootfig.SummaryTable
::: rootfig.Cutflow
::: rootfig.CutflowTable
::: rootfig.Efficiency
::: rootfig.Profile

## Errors and warnings

::: rootfig.errors

## Lower layers

These are the layers `rf.plot` is built from, usable on their own; names a module
does not list here are internal. The descriptions in `rootfig.model` are documented
above.

::: rootfig.io
    options:
      members:
        - FileSource
        - ArraySource
        - Source
        - ReadCache
        - as_source

::: rootfig.expressions
    options:
      members:
        - Expression
        - parse
        - FUNCTIONS
        - CONSTANTS

::: rootfig.selection
    options:
      members:
        - prepare
        - Columns
        - depth_of

::: rootfig.histograms
    options:
      members:
        - build_histograms
        - fill
        - sum_histograms
        - uncertainty
        - poisson_interval
        - shape_covariance
        - summarize
        - read_stored
        - compatible_binning

<!-- The function shares its name with a submodule; its defining path is unambiguous. -->
::: rootfig.histograms.normalize.normalize
    options:
      heading: normalize
      heading_level: 3

::: rootfig.plotting
    options:
      members:
        - draw_histograms
        - draw_panel
        - draw_hist2d
        - draw_correlation
        - draw_efficiencies
        - draw_profiles
        - split_stack
        - make_figure
        - Layout
        - figure_size
        - add_legend
        - add_stats_box
        - add_text
        - style_context
