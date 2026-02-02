using Plots, LaTeXStrings
import PyPlot

pyplot()
PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.matplotlib.rcParams["mathtext.fontset"] = "cm"
PyPlot.matplotlib.rcParams["text.latex.preamble"] = "\\usepackage{amssymb}"

# Generate data for master bound: q_H * L <= K
# Start q_h where the bound enters the visible region (L = 13 → q_h = K/13 ≈ 0.31)
q_h = range(0.30, 1.0, length=600)
K = 4.0
L_max = K ./ q_h

# Colors
bound_color = :black
fdr_color = colorant"#2b6cb0"
recall_color = colorant"#8c8c8c"

# Illustrative slice parameters
L_target = 2.0
q_target = 0.35

# RC attainment point on the boundary
q_rc = 0.5
L_rc = K / q_rc

# Create figure with fill under curve
fig = plot(collect(q_h), L_max,
    fillrange = 0.0,
    fillcolor = colorant"#ededed",
    linecolor = nothing,
    label = false,
    size = (1980, 780),
    dpi = 300,
    xlabel = L"Recall $q_H$",
    ylabel = L"Purity stringency $L(\varepsilon)=\log(1/\eta(\varepsilon))$",
    xlims = (0.0, 1.0),
    ylims = (0.0, 13.0),
    xticks = [0.0, 0.5, 1.0],
    yticks = [0.0, 4.0, 8.0, 12.0],
    legend = false,
    tickfontsize = 12,
    guidefontsize = 24,
    grid = false,
)

# Plot master bound curve
plot!(fig, collect(q_h), L_max,
    color = bound_color,
    linewidth = 3,
    label = L"Universal bound ($q_H\,L \leq \kappa$)")

# Plot fixed FDR target (horizontal dashed line)
hline!(fig, [L_target],
    color = fdr_color,
    linewidth = 2.5,
    linestyle = :dash,
    label = L"Fixed FDR target ($L$ fixed)")

# Plot fixed recall (vertical dashed line)
vline!(fig, [q_target],
    color = recall_color,
    linewidth = 2.5,
    linestyle = :dash,
    label = L"Fixed recall ($q_H$ fixed)")

# Plot RC attainment point
scatter!(fig, [q_rc], [L_rc],
    color = :black,
    markersize = 12,
    markerstrokewidth = 0,
    label = false)

# Add text annotation for RC point
annotate!(fig, q_rc + 0.06, L_rc - 0.5, text("RC\n(attains bound)", 18, :black, :left))

# Render the Plots.jl figure to PyPlot
display(fig)

# Now access the PyPlot figure and axes
pyfig = PyPlot.gcf()
ax = PyPlot.gca()

# Remove top and right spines
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

# Create legend handles manually
using PyPlot: matplotlib
line_bound = matplotlib.lines.Line2D([0], [0], color="black", linewidth=3, label=L"Universal bound ($q_H\,L \leq \kappa$)")
line_fdr = matplotlib.lines.Line2D([0], [0], color="#2b6cb0", linewidth=2.5, linestyle="--", label=L"Fixed FDR target ($L$ fixed)")
line_recall = matplotlib.lines.Line2D([0], [0], color="#8c8c8c", linewidth=2.5, linestyle="--", label=L"Fixed recall ($q_H$ fixed)")

handles = [line_bound, line_fdr, line_recall]

# Position legend at upper right
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
          fontsize=24, frameon=false)

# Save figures
PyPlot.savefig("non-empirical-figures/tradeoff_plane_schematic.pdf", bbox_inches="tight", facecolor="white")

# Save in overleaf folder
#pyfig.savefig("../overleaf/tex/v8_figures/tradeoff_plane_schematic.pdf", bbox_inches="tight", transparent=true)
