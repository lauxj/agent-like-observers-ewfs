import matplotlib.pyplot as plt
import numpy as np

# --- 1. Setup Data ---
values = np.array([3.97236210e-01, -5.92599402e-01, -5.23290988e-01, 7.53292542e-01])
errors = np.array([1.42131922e-03, 1.42188021e-03, 1.09935003e-03, 7.47616135e-04])
theory_values = np.array([0.500000, -0.707107, -0.500000, 0.707107])

# Calculate 2 Standard Deviations
errors_2sigma = errors * 2

# Colors and Labels
colors = ['#FFFFE0', '#5c5cff', '#d9f0ff', '#800080']
labels = [r'$\langle B\rangle_0$', r'$-\langle B\rangle_2$',
          r'$-\langle AB\rangle_0$', r'$\langle AB\rangle_1$']

# --- 2. Create Figure and Layout ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                               gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.15})

x_pos = np.arange(4)
bar_width = 0.8

# --- 3. Top Plot: Experimental Bars & Theoretical Outlines ---
for i in range(4):
    # Experimental (Filled) - Using 2-sigma errors
    ax1.bar(x_pos[i], values[i], width=bar_width, color=colors[i],
            edgecolor='black', yerr=errors_2sigma[i], capsize=5, zorder=2)
   
    # Theoretical (Red Dotted Outline)
    ax1.bar(x_pos[i], theory_values[i], width=bar_width, fill=False,
            edgecolor='red', linestyle='--', linewidth=2, zorder=3)

# Formatting Top Plot
ax1.set_xticks([])
ax1.set_xlim(-1, 4)
ax1.set_ylim(-1.0, 1.0)

# Custom Y-axis with LaTeX formatting
ax1.set_yticks([-1.0, -0.707, -0.5, 0.0, 0.5, 0.707, 1.0])
ax1.set_yticklabels(['-1.0', r'$-\frac{\sqrt{2}}{2}$', '-0.5', '0.0', '0.5', r'$\frac{\sqrt{2}}{2}$', '1.0'], fontsize=14)
ax1.axhline(0, color='black', linewidth=0.8, zorder=1)

# --- 4. Bottom Plot: Cumulative Sum Bar ---
left_exp = 0
left_th = 0
bar_height = 0.5

for i in range(4):
    width_exp = abs(values[i])
    width_th = abs(theory_values[i])
    err_width = errors_2sigma[i] # The gap width in data coordinates
   
    # Experimental Segment (Filled)
    ax2.barh(0, width_exp, height=bar_height, left=left_exp,
             color=colors[i], edgecolor='none', linewidth=1.5, zorder=2)
   
    # Theoretical Segment (Red Dotted Outline)
    ax2.barh(0, width_th, height=bar_height + 0.1, left=left_th,
             fill=False, edgecolor='red', linestyle='--', linewidth=1.5, zorder=3)
   
    # Add labels above the segments
    ax2.text(left_exp + (width_exp/2), bar_height/2 + 0.15, labels[i],
             ha='center', va='bottom', fontsize=12)
   
    left_exp += width_exp
    left_th += width_th
   
    # Plot the 2-sigma error as a solid black block (acts as the gap)
    ax2.barh(0, err_width, height=bar_height, left=left_exp, color='black', zorder=4)
   
    # Advance the experimental tracker by the error width
    left_exp += err_width

# --- 5. Add Threshold Lines and Region Labels ---
tsirelson_bound = 1 + np.sqrt(2) # ~2.414
axis_max = tsirelson_bound

# Red vertical lines
for threshold in [2, tsirelson_bound]:
    ax2.axvline(x=threshold, color='red', linewidth=2.5, ymin=0.1, ymax=0.9, zorder=1)

# Region labels
ax2.text(1.0, -0.6, 'CLASSICAL', ha='center', fontsize=11)
ax2.text((2 + tsirelson_bound)/2, -0.6, 'QUANTUM', ha='center', fontsize=11)

# Formatting Bottom Plot
ax2.set_xlim(0, axis_max + 0.1)
ax2.set_ylim(-1, 1)
ax2.set_yticks([])
for spine in ['top', 'left', 'right']:
    ax2.spines[spine].set_visible(False)

ax2.set_xticks([0, 2, tsirelson_bound])
ax2.set_xticklabels(['0', '2', r'$1 + \sqrt{2}$'], fontsize=14)
ax2.spines['bottom'].set_linewidth(1.5)

# --- 6. Final Polish and Export ---
plt.tight_layout() # Ensures no labels are cut off

# Save as a high-resolution PDF for publication
# plt.savefig("bell_inequality_plot.pdf", format="pdf", bbox_inches="tight", dpi=300)

# You can also save a PNG version if you need it for a presentation (PowerPoint, etc.)
# plt.savefig("bell_inequality_plot.png", format="png", bbox_inches="tight", dpi=300)

plt.show()