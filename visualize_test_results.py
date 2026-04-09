"""
Generates a visual dashboard of test coverage and test results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data from test run ---

modules = {
    'detection_utils.py': {'stmts': 46, 'miss': 0, 'cover': 100, 'tests': 19},
    'nfl_contact_detector.py': {'stmts': 122, 'miss': 27, 'cover': 78, 'tests': 25},
    'create_video_overlay.py': {'stmts': 147, 'miss': 39, 'cover': 73, 'tests': 28},
}

test_categories = {
    'Core Detection\nAlgorithm': 10,
    'Deceleration\nCalculation': 5,
    'Video Overlay\nDetection': 6,
    'Coordinate\nTransforms': 4,
    'Video I/O\n(Mocked)': 8,
    'Visualization\n(Mocked)': 3,
    'Input\nValidation': 6,
    'Shared\nUtilities': 12,
    'Edge Cases\n& NaN': 5,
    'Init &\nConfig': 4,
}

# Uncovered areas breakdown
uncovered = {
    'main() demo\nfunctions': 55,
    'Video frame\nloop internals': 8,
    'Covered\ncode': 234,
}

# --- Create figure ---
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('NFL Contact Detection - Test Coverage Dashboard',
             fontsize=22, fontweight='bold', color='white', y=0.97)

# Color palette
GREEN = '#3fb950'
YELLOW = '#d29922'
RED = '#f85149'
BLUE = '#58a6ff'
PURPLE = '#bc8cff'
GRAY = '#8b949e'
DARK_BG = '#161b22'
CARD_BG = '#21262d'

# =========================================================
# Plot 1: Coverage by module (horizontal bars)
# =========================================================
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(DARK_BG)

names = list(modules.keys())
covers = [modules[n]['cover'] for n in names]
colors = [GREEN if c >= 90 else YELLOW if c >= 70 else RED for c in covers]

y_pos = np.arange(len(names))
bars = ax1.barh(y_pos, covers, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

# Background bars for 100%
ax1.barh(y_pos, [100]*len(names), color='#30363d', height=0.6, zorder=0)

for i, (bar, cover) in enumerate(zip(bars, covers)):
    ax1.text(cover + 1.5, i, f'{cover}%', va='center', ha='left',
             fontsize=14, fontweight='bold', color=colors[i])

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=12, color='white', fontfamily='monospace')
ax1.set_xlim(0, 110)
ax1.set_xlabel('Coverage %', fontsize=12, color=GRAY)
ax1.set_title('Coverage by Module', fontsize=16, fontweight='bold', color='white', pad=15)
ax1.tick_params(colors=GRAY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color(GRAY)
ax1.spines['left'].set_color(GRAY)

# 70% threshold line
ax1.axvline(70, color=YELLOW, linestyle='--', alpha=0.6, linewidth=1.5)
ax1.text(71, len(names) - 0.1, '70% threshold', color=YELLOW, fontsize=9, alpha=0.8)

# =========================================================
# Plot 2: Tests by category (donut chart)
# =========================================================
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor(DARK_BG)

cat_names = list(test_categories.keys())
cat_values = list(test_categories.values())
cat_colors = [BLUE, '#1f6feb', PURPLE, GREEN, YELLOW, '#f0883e',
              RED, '#58a6ff', '#3fb950', GRAY]

wedges, texts, autotexts = ax2.pie(
    cat_values, labels=None, autopct='%1.0f%%',
    colors=cat_colors, startangle=90, pctdistance=0.8,
    wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2)
)

for t in autotexts:
    t.set_fontsize(8)
    t.set_color('white')
    t.set_fontweight('bold')

# Center text
ax2.text(0, 0, f'72\ntests', ha='center', va='center',
         fontsize=20, fontweight='bold', color='white')

ax2.set_title('Tests by Category', fontsize=16, fontweight='bold', color='white', pad=15)

# Legend
ax2.legend(wedges, [f'{n} ({v})' for n, v in zip(cat_names, cat_values)],
           loc='center left', bbox_to_anchor=(-0.35, 0.5),
           fontsize=8, frameon=False, labelcolor='white')

# =========================================================
# Plot 3: Statements covered vs uncovered per module
# =========================================================
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor(DARK_BG)

x = np.arange(len(names))
covered = [modules[n]['stmts'] - modules[n]['miss'] for n in names]
missed = [modules[n]['miss'] for n in names]

bars1 = ax3.bar(x, covered, 0.6, label='Covered', color=GREEN, edgecolor='white', linewidth=0.5)
bars2 = ax3.bar(x, missed, 0.6, bottom=covered, label='Uncovered', color='#30363d',
                edgecolor=GRAY, linewidth=0.5)

for i, (c, m) in enumerate(zip(covered, missed)):
    ax3.text(i, c/2, str(c), ha='center', va='center',
             fontsize=13, fontweight='bold', color='white')
    if m > 0:
        ax3.text(i, c + m/2, str(m), ha='center', va='center',
                 fontsize=11, fontweight='bold', color=GRAY)

ax3.set_xticks(x)
ax3.set_xticklabels(names, fontsize=10, color='white', fontfamily='monospace')
ax3.set_ylabel('Statements', fontsize=12, color=GRAY)
ax3.set_title('Statements: Covered vs Uncovered', fontsize=16, fontweight='bold',
              color='white', pad=15)
ax3.legend(loc='upper right', fontsize=10, frameon=False, labelcolor='white')
ax3.tick_params(colors=GRAY)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_color(GRAY)
ax3.spines['left'].set_color(GRAY)

# =========================================================
# Plot 4: Summary scorecard
# =========================================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(DARK_BG)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Big coverage number
ax4.text(5, 8.2, '78%', ha='center', va='center',
         fontsize=72, fontweight='bold', color=GREEN)
ax4.text(5, 6.8, 'OVERALL COVERAGE', ha='center', va='center',
         fontsize=14, fontweight='bold', color=GRAY)

# Stats cards
stats = [
    ('72', 'Tests Passed', GREEN),
    ('0', 'Tests Failed', GREEN),
    ('4', 'Source Files', BLUE),
    ('297', 'Total Stmts', PURPLE),
    ('231', 'Covered Stmts', GREEN),
    ('1.86s', 'Run Time', YELLOW),
]

for i, (val, label, color) in enumerate(stats):
    col = i % 3
    row = i // 3
    x = 1.5 + col * 2.8
    y = 4.5 - row * 2.5

    # Card background
    rect = mpatches.FancyBboxPatch(
        (x - 1.1, y - 1.0), 2.2, 2.0,
        boxstyle="round,pad=0.15", facecolor=CARD_BG,
        edgecolor=color, linewidth=1.5, alpha=0.8
    )
    ax4.add_patch(rect)

    ax4.text(x, y + 0.3, val, ha='center', va='center',
             fontsize=22, fontweight='bold', color=color)
    ax4.text(x, y - 0.4, label, ha='center', va='center',
             fontsize=9, fontweight='bold', color=GRAY)

ax4.set_title('Summary', fontsize=16, fontweight='bold', color='white', pad=15)

plt.tight_layout(rect=[0, 0, 1, 0.94])
output_path = 'visualizations/test_coverage_dashboard.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Dashboard saved to: {output_path}")
