"""MCM publication-figure library with backward-compatible flat modules."""

from .export_figure import (
    add_panel_label, apply_mcm_style, figure_size, label_panels, new_figure,
    new_panel_figure, save_figure, shared_legend, style_axis,
)
from .figure_audit import AuditReport, audit_export, audit_figure
from .stat_helpers import (
    classification_curves, confidence_interval, density_profile,
    five_number_summary, pareto_mask, standardize_matrix,
)
from .template_3d import make_3d_trajectory
from .template_bar import make_grouped_bar, make_stacked_bar
from .template_bar_timeseries import make_bar_timeseries, plot_from_excel
from .template_classification import make_confusion_matrix, make_pr_curve, make_roc_curve
from .template_convergence import make_convergence
from .template_diagnostics import (
    make_calibration_plot, make_error_distribution, make_qq_plot, make_residual_plot,
)
from .template_distribution import (
    make_boxplot, make_distribution, make_ecdf, make_violin,
)
from .template_errorbar import make_errorbar
from .template_flow import make_sankey
from .template_forecast import make_forecast
from .template_heatmap import make_heatmap
from .template_map import make_choropleth, make_route_map
from .template_multicriteria import make_parallel_coordinates, make_ranked_dot
from .template_network import make_network
from .template_pareto import make_pareto_front
from .template_pca import make_pca
from .template_phase import make_phase
from .template_radar import make_radar
from .template_scatter import make_scatter_fit
from .template_scenario import make_scenario_hist
from .template_schedule import make_gantt
from .template_searchpath import make_searchpath
from .template_sobol import make_sobol
from .template_tornado import make_tornado
from .template_waterfall import make_waterfall

__all__ = [name for name in globals() if not name.startswith('_')]
