using Plots, LaTeXStrings
import PyPlot

pyplot()
PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.matplotlib.rcParams["mathtext.fontset"] = "cm"

# Generate data
q_h = range(0.0, 1.0, length=600)
short = clamp.(0.75 .* (q_h .^ 1.25), 0.0, 1.0)
forced = clamp.(0.55 .* (q_h .^ 1.25), 0.0, 1.0)

# Quota line parameters
pi_val = 0.25
rho_bar = 0.55
q0_iso = (rho_bar .- pi_val .* q_h) ./ (1.0 - pi_val)

# Mask for valid quota line range
mask = (q0_iso .>= 0.0) .& (q0_iso .<= 1.0)
q_h_masked = collect(q_h)[mask]
q0_iso_masked = q0_iso[mask]
short_masked = short[mask]
forced_masked = forced[mask]

# Colors
tighten_color = colorant"#009E73"
disclose_color = colorant"#B31B1B"
quota_color = colorant"#2563eb"

# Find intersection points
idx_short = argmin(abs.(short_masked .- q0_iso_masked))
idx_forced = argmin(abs.(forced_masked .- q0_iso_masked))

qh_short = q_h_masked[idx_short]
q0_short = q0_iso_masked[idx_short]
qh_forced = q_h_masked[idx_forced]
q0_forced = q0_iso_masked[idx_forced]

# Tightened quota line (parallel to original, with lower rho)
rho_tight = 0.42  # Lower than rho_bar to shift line left
q0_tight = (rho_tight .- pi_val .* q_h) ./ (1.0 - pi_val)

# Mask for valid tightened quota line range
mask_tight = (q0_tight .>= 0.0) .& (q0_tight .<= 1.0)
q_h_tight = collect(q_h)[mask_tight]
q0_tight_masked = q0_tight[mask_tight]
short_tight = short[mask_tight]
forced_tight = forced[mask_tight]

# Find intersection points for tightened quota line
idx_short_tight = argmin(abs.(short_tight .- q0_tight_masked))
idx_forced_tight = argmin(abs.(forced_tight .- q0_tight_masked))

qh_short_tight = q_h_tight[idx_short_tight]
q0_short_tight = q0_tight_masked[idx_short_tight]
qh_forced_tight = q_h_tight[idx_forced_tight]
q0_forced_tight = q0_tight_masked[idx_forced_tight]

# Arrow endpoints: from original quota intersections to tightened quota intersections
x0, y0 = qh_short, q0_short
x1, y1 = qh_short_tight, q0_short_tight

x0b, y0b = qh_forced, q0_forced
x1b, y1b = qh_forced_tight, q0_forced_tight

# Create figure with fill between curves (no label for fill)
fig = plot(collect(q_h), short,
    fillrange = forced,
    fillcolor = colorant"#ededed",
    linecolor = nothing,
    label = false,
    size = (1980, 780),
    dpi = 300,
    xlabel = L"True-positive rate $q_H$",
    ylabel = L"False-positive rate $q_0$",
    xlims = (0.0, 1.0),
    ylims = (0.0, 1.0),
    xticks = [0.0, 0.5, 1.0],
    yticks = [0.0, 0.5, 1.0],
    legend = false,
    tickfontsize = 12,
    guidefontsize = 24,
    grid = false,
)

# Plot short disclosure line separately (for legend to show line, not box)
plot!(fig, collect(q_h), short,
    color = colorant"#5a5a5a",
    linewidth = 3,
    label = "Short disclosure")

# Plot forced disclosure curve
plot!(fig, collect(q_h), forced,
    color = :black,
    linewidth = 3,
    label = "Forced disclosure")

# Plot quota line (dashed)
plot!(fig, q_h_masked, q0_iso_masked,
    color = quota_color,
    linewidth = 3,
    linestyle = :dash,
    label = L"Quota ($\rho=\bar\rho$)")

# Plot tightened quota line segment (parallel to original quota line)
plot!(fig, [qh_short_tight, qh_forced_tight], [q0_short_tight, q0_forced_tight],
    color = quota_color,
    linewidth = 3,
    linestyle = :dashdot,
    label = L"Tightened quota ($\rho<\bar\rho$)")

# Draw arrows using quiver
# "Force disclosure" arrow (red) - half length
quiver!(fig, [qh_short], [q0_short],
    quiver = ([0.5 * (qh_forced - qh_short)], [0.5 * (q0_forced - q0_short)]),
    color = disclose_color,
    linewidth = 4.5,
    arrow = arrow(:closed, :head, 1, 1))

# "Tighten standards" arrows (green)
quiver!(fig, [x0], [y0],
    quiver = ([x1 - x0], [y1 - y0]),
    color = tighten_color,
    linewidth = 4.5,
    arrow = arrow(:closed, :head, 1, 1))

quiver!(fig, [x0b], [y0b],
    quiver = ([x1b - x0b], [y1b - y0b]),
    color = tighten_color,
    linewidth = 4.5,
    arrow = arrow(:closed, :head, 1, 1))

# Add text annotations
# Position "Force disclosure"
annotate!(fig, 0.85, 0.53, text(L"\textbf{Force}\\\textbf{disclosure}", 24, disclose_color, :center))
# Position "Tighten standards"
annotate!(fig, 0.57, 0.45, text(L"\textbf{Tighten}\\\textbf{standards}", 24, tighten_color, :center))

# Plot markers at intersection points
scatter!(fig, [qh_short], [q0_short],
    color = colorant"#737373",
    markersize = 15,
    markerstrokewidth = 1,
    markerstrokecolor = :white,
    label = false)

scatter!(fig, [qh_forced], [q0_forced],
    color = :black,
    markersize = 15,
    markerstrokewidth = 1,
    markerstrokecolor = :white,
    label = false)

# Render the Plots.jl figure to PyPlot
display(fig)

# Now access the PyPlot figure and axes
pyfig = PyPlot.gcf()
ax = PyPlot.gca()

# Remove top and right spines
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

# Create legend handles manually since Plots.jl legend=false doesn't register them
using PyPlot: matplotlib
line_short = matplotlib.lines.Line2D([0], [0], color="#5a5a5a", linewidth=3, label="Short disclosure")
line_forced = matplotlib.lines.Line2D([0], [0], color="black", linewidth=3, label="Forced disclosure")
line_quota = matplotlib.lines.Line2D([0], [0], color="#2563eb", linewidth=3, linestyle="--", label=L"Quota ($\rho=\bar\rho$)")
line_tight = matplotlib.lines.Line2D([0], [0], color="#2563eb", linewidth=3, linestyle="-.", label=L"Tightened quota ($\rho<\bar\rho$)")

handles = [line_short, line_forced, line_quota, line_tight]

# Position legend at top right of plot area (right-justified, not running over)
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
          ncol=2, fontsize=24, frameon=false)

# Save figures
pyfig.savefig("non-empirical-figures/frontier_schematic.pdf", bbox_inches="tight", facecolor="white")

# Save in overleaf folder
pyfig.savefig("../overleaf/tex/v8_figures/frontier_schematic.pdf", bbox_inches="tight", transparent=true)