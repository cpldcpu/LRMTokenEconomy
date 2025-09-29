#!/usr/bin/env python3
"""
Latency Analysis Script

Analyzes per-sample request latency captured in evaluation_stats.csv (column: latency_ms)
producing multiple figures and summary statistics.

Outputs (saved to --output-dir, default figures/latency):
  1. latency_boxplot_by_model.png            Boxplot of latency (s) per model
  2. latency_heatmap_mean.png                Heatmap (model x prompt) mean latency (s)
  3. tokens_vs_latency_scatter_<type>.png    Scatter per prompt type with regression lines per model
  4. tokens_vs_latency_slopes.png            Bar chart of regression slope (ms per token) per model
  5. tokens_vs_latency_intercepts.png        Bar chart of regression intercept (ms) per model
  6. tps_bar_chart_mean.png                  Mean tokens/sec per model
  7. tps_bar_chart_median.png                Median tokens/sec per model
  8. avg_latency_vs_success_rate.png         Scatter of mean latency vs mean success rate per model
  9. reasoning_tokens_vs_latency_scatter.png (optional if tokens_reasoning present & non-zero) 
 10. latency_hist_per_model.png              Overlaid histogram / KDE of latency distribution

Also: latency_summary_stats.csv (descriptive stats for latency_s and tps) & regression_stats.csv.

Assumptions:
  - latency_ms may be missing for some rows; those rows are ignored in latency-specific charts.
  - tokens_completion > 0 & latency_s > 0 for regression & TPS metrics.

"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Global style (mirrors patterns in other analysis scripts: bold labels, grid, consistent palette)
sns.set_theme(style='whitegrid', context='notebook', font_scale=1.0)
plt.rcParams.update({
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.borderpad': 0.4,
    'legend.fontsize': 9,
})

REQUIRED_COLUMNS = ['model_name', 'prompt_id', 'tokens_completion', 'latency_ms']
OPTIONAL_COLUMNS = ['type', 'success_rate', 'tokens_reasoning', 'full_cot']


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze latency metrics from evaluation stats CSV"
    )
    parser.add_argument('--input', default='evaluation_stats.csv',
                        help='Path to evaluation_stats.csv (default: evaluation_stats.csv)')
    parser.add_argument('--output-dir', default='figures/latency',
                        help='Directory to store output figures (default: figures/latency)')
    parser.add_argument('--figsize', default='10,6',
                        help="Figure size as width,height (default: 10,6)")
    parser.add_argument('--min-samples', type=int, default=3,
                        help='Minimum data points per model for regression (default: 3)')
    parser.add_argument('--no-kde', action='store_true',
                        help='Disable KDE overlay for latency histogram')
    return parser.parse_args()


def validate_input(path: Path):
    if not path.exists():
        print(f"Error: Input file not found: {path}")
        sys.exit(1)
    if not path.is_file():
        print(f"Error: Input path is not a file: {path}")
        sys.exit(1)


def create_output_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {path}")


def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows without latency
    if 'latency_ms' not in df.columns:
        print('Error: latency_ms column not present. Regenerate evaluation_stats.csv after adding latency measurement.')
        sys.exit(1)

    # Convert latency to seconds
    df = df.copy()
    df['latency_s'] = df['latency_ms'] / 1000.0

    # Guard against zero / negative latencies
    df.loc[df['latency_s'] <= 0, 'latency_s'] = np.nan

    # Tokens per second (use completion tokens)
    df['tps'] = df.apply(lambda r: r['tokens_completion'] / r['latency_s']
                         if pd.notna(r['latency_s']) and r['latency_s'] > 0 and r['tokens_completion'] > 0 else np.nan, axis=1)

    # Filter to valid latency entries for latency-centric plots
    return df


def linear_regression_tokens_latency(df: pd.DataFrame, min_samples: int):
    """Compute per-model linear regression latency_ms = m * tokens_completion + b."""
    results = []
    for model, g in df.groupby('model_name'):
        sub = g[['tokens_completion', 'latency_ms']].dropna()
        # Valid points: tokens > 0 & latency > 0
        sub = sub[(sub['tokens_completion'] > 0) & (sub['latency_ms'] > 0)]
        if len(sub) < min_samples:
            continue
        x = sub['tokens_completion'].values.astype(float)
        y = sub['latency_ms'].values.astype(float)
        # Handle constant tokens edge case
        if np.all(x == x[0]):
            # Slope undefined; treat as NaN
            slope = np.nan
            intercept = np.mean(y) if len(y) else np.nan
            r_value = 0.0
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2 if not np.isnan(r_value) else np.nan
        results.append({
            'model_name': model,
            'points': len(sub),
            'slope_ms_per_token': slope,  # ms per token
            'intercept_ms': intercept,
            'r_squared': r_squared
        })
    return pd.DataFrame(results)


def plot_latency_boxplot(df: pd.DataFrame, outdir: Path, figsize):
    sub = df[['model_name', 'latency_s']].dropna()
    if sub.empty:
        return
    plt.figure(figsize=figsize)
    order = sub.groupby('model_name')['latency_s'].median().sort_values().index
    sns.boxplot(data=sub, x='model_name', y='latency_s', order=order, showfliers=False, linewidth=0.8, color='#D9E8F5')
    sns.stripplot(data=sub, x='model_name', y='latency_s', order=order, color='black', size=3, alpha=0.35)
    plt.ylabel('Latency (s)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.title('Latency Distribution by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outdir / 'latency_boxplot_by_model.png', dpi=150)
    plt.close()


def plot_latency_hist(df: pd.DataFrame, outdir: Path, figsize, disable_kde: bool):
    sub = df['latency_s'].dropna()
    if sub.empty:
        return
    plt.figure(figsize=figsize)
    sns.histplot(sub, bins=40, kde=not disable_kde, color='#4682B4', edgecolor='black', linewidth=0.3)
    plt.xlabel('Latency (s)', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Overall Latency Distribution')
    plt.tight_layout()
    plt.savefig(outdir / 'latency_hist_overall.png', dpi=150)
    plt.close()

    # Per model overlay
    per = df[['model_name', 'latency_s']].dropna()
    if per.empty:
        return
    plt.figure(figsize=figsize)
    for model, g in per.groupby('model_name'):
        sns.kdeplot(g['latency_s'], label=model, fill=False, linewidth=1.2)
    plt.xlabel('Latency (s)', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('Latency Density per Model')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / 'latency_density_per_model.png', dpi=150)
    plt.close()


def plot_heatmap_tps(df: pd.DataFrame, outdir: Path, figsize):
    """Plot heatmap of mean TPS (Tokens Per Second) by model and prompt."""
    sub = df[['model_name', 'prompt_id', 'tps', 'type', 'open_weights']].dropna(subset=['tps'])
    if sub.empty:
        return
    
    # Calculate mean TPS per model-prompt combination
    pivot_data = sub.groupby(['model_name', 'prompt_id', 'open_weights', 'type'])['tps'].mean().reset_index()
    table = pivot_data.pivot(index='model_name', columns='prompt_id', values='tps')
    
    if table.empty:
        return
    
    # Sort models by open_weights (closed first, then open) and model name
    model_groups = pivot_data.groupby('model_name')['open_weights'].first().reset_index()
    model_order = model_groups.sort_values(['open_weights', 'model_name'])['model_name'].tolist()
    
    # Sort prompts by type and prompt_id
    prompt_groups = pivot_data.groupby('prompt_id')['type'].first().reset_index()
    prompt_order = prompt_groups.sort_values(['type', 'prompt_id'])['prompt_id'].tolist()
    
    # Reorder table with available models/prompts only
    available_models = [m for m in model_order if m in table.index]
    available_prompts = [p for p in prompt_order if p in table.columns]
    table = table.reindex(index=available_models, columns=available_prompts)
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(figsize[0]*1.4, figsize[1]*1.1))
    
    # Create heatmap with consistent styling
    sns.heatmap(
        table, 
        cmap='plasma', 
        annot=True, 
        fmt='.1f',
        cbar_kws={'label': 'Tokens Per Second'},
        linewidths=0.1,
        square=False,
        ax=ax,
        annot_kws={'fontsize': 8}
    )
    
    # Add group separators (white lines between different types/weights)
    add_group_separators_tps(ax, pivot_data, table)
    
    # Style consistently with other heatmaps
    ax.set_title('Mean Tokens Per Second (TPS) by LLM and Prompt\n(Grouped by Open Weights & Prompt Type)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Prompt ID (Grouped by Type)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Model Name (Grouped by Open Weights)', fontsize=10, fontweight='bold')
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Adjust layout to accommodate group labels
    plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
    
    plt.savefig(outdir / 'tps_heatmap_mean.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.clf()


def add_group_separators_tps(ax, pivot_data: pd.DataFrame, pivot_table: pd.DataFrame) -> None:
    """Add visual separators to group models by open_weights and prompts by type for TPS heatmap."""
    
    # Get grouping information
    model_groups = pivot_data.groupby('model_name')['open_weights'].first()
    prompt_groups = pivot_data.groupby('prompt_id')['type'].first()
    
    # Add vertical lines to separate prompt types
    current_type = None
    x_pos = 0
    type_positions = {}
    for prompt_id in pivot_table.columns:
        if prompt_id in prompt_groups.index:
            prompt_type = prompt_groups[prompt_id]
            if current_type is not None and prompt_type != current_type:
                ax.axvline(x=x_pos, color='white', linewidth=3, alpha=0.8)
            if prompt_type not in type_positions:
                type_positions[prompt_type] = x_pos
            current_type = prompt_type
        x_pos += 1
    
    # Add horizontal lines to separate model types (open_weights)
    current_open_weights = None
    y_pos = 0
    weight_positions = {}
    for model_name in pivot_table.index:
        if model_name in model_groups.index:
            open_weights = model_groups[model_name]
            if current_open_weights is not None and open_weights != current_open_weights:
                ax.axhline(y=y_pos, color='white', linewidth=3, alpha=0.8)
            weight_type = 'Open Weights' if open_weights else 'Closed Weights'
            if weight_type not in weight_positions:
                weight_positions[weight_type] = y_pos
            current_open_weights = open_weights
        y_pos += 1
    
    # Add group labels as text annotations
    # Add prompt type labels at the bottom
    for ptype, start_pos in type_positions.items():
        type_prompts = [p for p in pivot_table.columns if prompt_groups.get(p) == ptype]
        if type_prompts:
            start_idx = list(pivot_table.columns).index(type_prompts[0])
            end_idx = list(pivot_table.columns).index(type_prompts[-1])
            mid_pos = (start_idx + end_idx) / 2
            ax.text(mid_pos, -0.5, ptype.title(), ha='center', va='top', 
                   fontweight='bold', fontsize=9, color='#B22222')
    
    # Add model type labels on the left
    for weight_type, start_pos in weight_positions.items():
        weight_models = [m for m in pivot_table.index 
                        if model_groups.get(m) == (weight_type == 'Open Weights')]
        if weight_models:
            start_idx = list(pivot_table.index).index(weight_models[0])  
            end_idx = list(pivot_table.index).index(weight_models[-1])
            mid_pos = (start_idx + end_idx) / 2
            ax.text(-0.5, mid_pos, weight_type, ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='#B22222', rotation=90)


def plot_heatmap_latency(df: pd.DataFrame, outdir: Path, figsize):
    sub = df[['model_name', 'prompt_id', 'latency_s', 'type', 'open_weights']].dropna(subset=['latency_s'])
    if sub.empty:
        return
    
    # Calculate mean latency per model-prompt combination
    pivot_data = sub.groupby(['model_name', 'prompt_id', 'open_weights', 'type'])['latency_s'].mean().reset_index()
    table = pivot_data.pivot(index='model_name', columns='prompt_id', values='latency_s')
    
    if table.empty:
        return
    
    # Sort models by open_weights (closed first, then open) and model name
    model_groups = pivot_data.groupby('model_name')['open_weights'].first().reset_index()
    model_order = model_groups.sort_values(['open_weights', 'model_name'])['model_name'].tolist()
    
    # Sort prompts by type and prompt_id
    prompt_groups = pivot_data.groupby('prompt_id')['type'].first().reset_index()
    prompt_order = prompt_groups.sort_values(['type', 'prompt_id'])['prompt_id'].tolist()
    
    # Reorder table with available models/prompts only
    available_models = [m for m in model_order if m in table.index]
    available_prompts = [p for p in prompt_order if p in table.columns]
    table = table.reindex(index=available_models, columns=available_prompts)
    
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(figsize[0]*1.4, figsize[1]*1.1))
    
    # Create heatmap with consistent styling
    sns.heatmap(
        table, 
        cmap='viridis', 
        annot=True, 
        fmt='.2f',
        cbar_kws={'label': 'Latency (s)'},
        linewidths=0.1,
        square=False,
        ax=ax,
        annot_kws={'fontsize': 8}
    )
    
    # Add group separators (white lines between different types/weights)
    add_group_separators_latency(ax, pivot_data, table)
    
    # Style consistently with other heatmaps
    ax.set_title('Mean Latency (s) by LLM and Prompt\n(Grouped by Open Weights & Prompt Type)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Prompt ID (Grouped by Type)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Model Name (Grouped by Open Weights)', fontsize=10, fontweight='bold')
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Adjust layout to accommodate group labels
    plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
    
    plt.savefig(outdir / 'latency_heatmap_mean.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.clf()


def add_group_separators_latency(ax, pivot_data: pd.DataFrame, pivot_table: pd.DataFrame) -> None:
    """Add visual separators to group models by open_weights and prompts by type."""
    
    # Get grouping information
    model_groups = pivot_data.groupby('model_name')['open_weights'].first()
    prompt_groups = pivot_data.groupby('prompt_id')['type'].first()
    
    # Add vertical lines to separate prompt types
    current_type = None
    x_pos = 0
    type_positions = {}
    for prompt_id in pivot_table.columns:
        if prompt_id in prompt_groups.index:
            prompt_type = prompt_groups[prompt_id]
            if current_type is not None and prompt_type != current_type:
                ax.axvline(x=x_pos, color='white', linewidth=3, alpha=0.8)
            if prompt_type not in type_positions:
                type_positions[prompt_type] = x_pos
            current_type = prompt_type
        x_pos += 1
    
    # Add horizontal lines to separate model types (open_weights)
    current_open_weights = None
    y_pos = 0
    weight_positions = {}
    for model_name in pivot_table.index:
        if model_name in model_groups.index:
            open_weights = model_groups[model_name]
            if current_open_weights is not None and open_weights != current_open_weights:
                ax.axhline(y=y_pos, color='white', linewidth=3, alpha=0.8)
            weight_type = 'Open Weights' if open_weights else 'Closed Weights'
            if weight_type not in weight_positions:
                weight_positions[weight_type] = y_pos
            current_open_weights = open_weights
        y_pos += 1
    
    # Add group labels as text annotations
    # Add prompt type labels at the bottom
    for ptype, start_pos in type_positions.items():
        type_prompts = [p for p in pivot_table.columns if prompt_groups.get(p) == ptype]
        if type_prompts:
            start_idx = list(pivot_table.columns).index(type_prompts[0])
            end_idx = list(pivot_table.columns).index(type_prompts[-1])
            mid_pos = (start_idx + end_idx) / 2
            ax.text(mid_pos, -0.5, ptype.title(), ha='center', va='top', 
                   fontweight='bold', fontsize=9, color='#B22222')
    
    # Add model type labels on the left
    for weight_type, start_pos in weight_positions.items():
        weight_models = [m for m in pivot_table.index 
                        if model_groups.get(m) == (weight_type == 'Open Weights')]
        if weight_models:
            start_idx = list(pivot_table.index).index(weight_models[0])  
            end_idx = list(pivot_table.index).index(weight_models[-1])
            mid_pos = (start_idx + end_idx) / 2
            ax.text(-0.5, mid_pos, weight_type, ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='#B22222', rotation=90)


def plot_tokens_vs_latency(df: pd.DataFrame, outdir: Path, figsize):
    # Scatter by prompt type if available, else one global scatter
    types = ['__ALL__']
    if 'type' in df.columns and df['type'].notna().any():
        types = sorted(df['type'].dropna().unique().tolist())
    for ptype in types:
        if ptype == '__ALL__':
            subset = df
            title_suffix = 'All Types'
            fname_suffix = 'all'
        else:
            subset = df[df['type'] == ptype]
            title_suffix = ptype.title()
            fname_suffix = ptype.replace(' ', '_')
        sub = subset[['model_name', 'tokens_completion', 'latency_ms']].dropna()
        sub = sub[(sub['tokens_completion'] > 0) & (sub['latency_ms'] > 0)]
        if sub.empty:
            continue
        plt.figure(figsize=figsize)
        models = sorted(sub['model_name'].unique())
        palette = sns.color_palette('tab20', n_colors=len(models))
        color_map = {m: palette[i] for i, m in enumerate(models)}
        # Plot each model separately to have separate regression lines manually
        legend_handles = []
        for m in models:
            mg = sub[sub['model_name'] == m]
            scatter = plt.scatter(mg['tokens_completion'], mg['latency_ms'], s=25, alpha=0.6, color=color_map[m], label=m)
            legend_handles.append(scatter)
            # Regression line
            if len(mg) >= 3 and mg['tokens_completion'].nunique() > 1:
                x = mg['tokens_completion'].values.astype(float)
                y = mg['latency_ms'].values.astype(float)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, color=color_map[m], linestyle='--', linewidth=1.5)
        
        plt.xlabel('Completion Tokens', fontweight='bold')
        plt.ylabel('Latency (ms)', fontweight='bold')
        plt.title(f'Tokens vs Latency (ms) - {title_suffix}')
        plt.grid(alpha=0.3)
        
        # Add legend if we have multiple models
        if len(models) > 1:
            plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(outdir / f'tokens_vs_latency_scatter_{fname_suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_regression_bars(reg_df: pd.DataFrame, outdir: Path, figsize):
    if reg_df.empty:
        return
    reg_sorted = reg_df.sort_values('slope_ms_per_token')
    # Optionally color by open_weights if available
    open_map = {}
    if 'open_weights' in reg_df.columns:
        open_map = dict(zip(reg_df['model_name'], reg_df['open_weights']))
    # Slope bar chart
    plt.figure(figsize=figsize)
    colors = []
    for m in reg_sorted['model_name']:
        if open_map:
            colors.append('#2F85A7' if open_map.get(m, False) else '#5C6B75')
        else:
            colors.append('#4C72B0')
    sns.barplot(data=reg_sorted, x='model_name', y='slope_ms_per_token', palette=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Slope (ms/token)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.title('Tokens→Latency Regression Slope (ms per token)')
    for i, row in enumerate(reg_sorted.itertuples()):
        val = row.slope_ms_per_token
        if pd.notna(val):
            plt.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    if open_map:
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor='#5C6B75', label='Closed Weights'), Patch(facecolor='#2F85A7', label='Open Weights')]
        plt.legend(handles=legend_elems, loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / 'tokens_vs_latency_slopes.png', dpi=150)
    plt.close()

    # Intercept bar chart
    plt.figure(figsize=figsize)
    reg_sorted_i = reg_df.sort_values('intercept_ms')
    colors_i = []
    for m in reg_sorted_i['model_name']:
        if open_map:
            colors_i.append('#2F85A7' if open_map.get(m, False) else '#5C6B75')
        else:
            colors_i.append('#DD8452')
    sns.barplot(data=reg_sorted_i, x='model_name', y='intercept_ms', palette=colors_i)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Intercept (ms)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.title('Tokens→Latency Regression Intercept (ms)')
    for i, row in enumerate(reg_sorted_i.itertuples()):
        val = row.intercept_ms
        if pd.notna(val):
            plt.text(i, val, f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    if open_map:
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor='#5C6B75', label='Closed Weights'), Patch(facecolor='#2F85A7', label='Open Weights')]
        plt.legend(handles=legend_elems, loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / 'tokens_vs_latency_intercepts.png', dpi=150)
    plt.close()


def plot_tps_bars(df: pd.DataFrame, outdir: Path, figsize):
    sub = df[['model_name', 'tps']].dropna()
    if sub.empty:
        return
    agg = sub.groupby('model_name')['tps'].agg(['mean', 'median', 'count']).reset_index()
    # Determine open vs closed if column exists
    open_map = {}
    if 'open_weights' in df.columns:
        open_map = dict(zip(df.groupby('model_name')['open_weights'].first().index, df.groupby('model_name')['open_weights'].first().values))

    # Mean chart
    order = agg.sort_values('mean')['model_name']
    plt.figure(figsize=figsize)
    colors = []
    for m in order:
        if open_map:
            colors.append('#2F85A7' if open_map.get(m, False) else '#5C6B75')
        else:
            colors.append('#4C72B0')
    sns.barplot(data=agg, x='model_name', y='mean', order=order, palette=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Tokens / Second (Mean)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.title('Mean Throughput (Tokens per Second)')
    for i, row in enumerate(agg.set_index('model_name').loc[order].itertuples()):
        plt.text(i, row.mean, f'{row.mean:.1f}', ha='center', va='bottom', fontsize=8)
    if open_map:
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor='#5C6B75', label='Closed Weights'), Patch(facecolor='#2F85A7', label='Open Weights')]
        plt.legend(handles=legend_elems, loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'tps_bar_chart_mean.png', dpi=150)
    plt.close()

    # Median chart
    order2 = agg.sort_values('median')['model_name']
    plt.figure(figsize=figsize)
    colors2 = []
    for m in order2:
        if open_map:
            colors2.append('#2F85A7' if open_map.get(m, False) else '#5C6B75')
        else:
            colors2.append('#DD8452')
    sns.barplot(data=agg, x='model_name', y='median', order=order2, palette=colors2)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Tokens / Second (Median)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.title('Median Throughput (Tokens per Second)')
    for i, row in enumerate(agg.set_index('model_name').loc[order2].itertuples()):
        plt.text(i, row.median, f'{row.median:.1f}', ha='center', va='bottom', fontsize=8)
    if open_map:
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor='#5C6B75', label='Closed Weights'), Patch(facecolor='#2F85A7', label='Open Weights')]
        plt.legend(handles=legend_elems, loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'tps_bar_chart_median.png', dpi=150)
    plt.close()


def plot_latency_vs_success(df: pd.DataFrame, outdir: Path, figsize):
    if 'success_rate' not in df.columns:
        return
    sub = df[['model_name', 'latency_s', 'success_rate']].dropna()
    if sub.empty:
        return
    agg = sub.groupby('model_name').agg(mean_latency_s=('latency_s', 'mean'),
                                        mean_success_rate=('success_rate', 'mean')).reset_index()
    plt.figure(figsize=figsize)
    plt.scatter(agg['mean_latency_s'], agg['mean_success_rate'], s=60, color='teal', alpha=0.8)
    for _, r in agg.iterrows():
        plt.text(r['mean_latency_s'], r['mean_success_rate'], r['model_name'], fontsize=8, ha='center', va='bottom')
    plt.xlabel('Mean Latency (s)', fontweight='bold')
    plt.ylabel('Mean Success Rate', fontweight='bold')
    plt.title('Mean Latency vs Mean Success Rate (per Model)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'avg_latency_vs_success_rate.png', dpi=150)
    plt.close()


def plot_reasoning_latency(df: pd.DataFrame, outdir: Path, figsize):
    if 'tokens_reasoning' not in df.columns:
        return
    # Only consider rows with reasoning tokens > 0
    sub = df[(df['tokens_reasoning'] > 0) & (df['latency_ms'] > 0)][['model_name', 'tokens_reasoning', 'latency_ms', 'full_cot']].dropna()
    if sub.empty:
        return
    # Optionally filter to full_cot models if that column exists
    if 'full_cot' in sub.columns and sub['full_cot'].notna().any():
        sub = sub[sub['full_cot'] == True]
    if sub.empty:
        return
    plt.figure(figsize=figsize)
    models = sorted(sub['model_name'].unique())
    palette = sns.color_palette('tab10', n_colors=len(models))
    color_map = {m: palette[i] for i, m in enumerate(models)}
    for m in models:
        mg = sub[sub['model_name'] == m]
        plt.scatter(mg['tokens_reasoning'], mg['latency_ms'], label=m, s=30, alpha=0.65, color=color_map[m])
    plt.xlabel('Reasoning Tokens', fontweight='bold')
    plt.ylabel('Latency (ms)', fontweight='bold')
    plt.title('Reasoning Tokens vs Latency (Full CoT Models)')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'reasoning_tokens_vs_latency_scatter.png', dpi=150)
    plt.close()


def save_summary_stats(df: pd.DataFrame, reg_df: pd.DataFrame, outdir: Path):
    lat_sub = df[['model_name', 'latency_s', 'tps']].dropna(subset=['latency_s'])
    if lat_sub.empty:
        return
    stats_records = []
    for model, g in lat_sub.groupby('model_name'):
        stats_records.append({
            'model_name': model,
            'count': len(g),
            'latency_mean_s': g['latency_s'].mean(),
            'latency_median_s': g['latency_s'].median(),
            'latency_p95_s': g['latency_s'].quantile(0.95),
            'latency_min_s': g['latency_s'].min(),
            'latency_max_s': g['latency_s'].max(),
            'tps_mean': g['tps'].mean(),
            'tps_median': g['tps'].median(),
        })
    stats_df = pd.DataFrame(stats_records)
    stats_df.to_csv(outdir / 'latency_summary_stats.csv', index=False)
    if not reg_df.empty:
        reg_df.to_csv(outdir / 'regression_stats.csv', index=False)


def main():
    args = parse_arguments()

    # Parse figsize
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Error: Invalid figsize format. Use 'width,height'")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    validate_input(input_path)
    create_output_directory(output_dir)

    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    check_columns(df)

    df = prepare_data(df)

    if df['latency_s'].dropna().empty:
        print('No valid latency data found (latency_ms all missing/invalid). Exiting.')
        sys.exit(0)

    # Regression (per model)
    reg_df = linear_regression_tokens_latency(df, args.min_samples)
    print(f"Computed regression for {len(reg_df)} models (>= {args.min_samples} samples each)")

    # Plots
    plot_latency_boxplot(df, output_dir, figsize)
    plot_latency_hist(df, output_dir, figsize, args.no_kde)
    plot_heatmap_latency(df, output_dir, figsize)
    plot_heatmap_tps(df, output_dir, figsize)
    plot_tokens_vs_latency(df, output_dir, figsize)
    plot_regression_bars(reg_df, output_dir, figsize)
    plot_tps_bars(df, output_dir, figsize)
    plot_latency_vs_success(df, output_dir, figsize)
    plot_reasoning_latency(df, output_dir, figsize)

    # Summaries
    save_summary_stats(df, reg_df, output_dir)

    print(f"Latency analysis complete. Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
