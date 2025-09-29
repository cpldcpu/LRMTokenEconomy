#!/usr/bin/env python3
"""
Model Trends Analysis Script

This script analyzes token usage patterns grouped by lab and model,
creating grouped bar charts to show trends across different research organizations.

Usage:
    python analyze_model_trends.py --types knowledge,logic_puzzle,math --output-dir figures/trends_eco_all
    python analyze_model_trends.py --types knowledge --output-dir figures/trends_eco_knowledge
    python analyze_model_trends.py --types math --output-dir figures/trends_eco_math
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for better performance
plt.switch_backend('Agg')

# Type mappings for filtering and display
TYPE_MAPPINGS = {
    'knowledge': 'knowledge',
    'logic_puzzle': 'logic puzzle', 
    'math': 'Math'
}

TYPE_DISPLAY_NAMES = {
    'knowledge': 'Knowledge',
    'logic_puzzle': 'Logic Puzzle',
    'math': 'Math'
}

# Preset configurations for common analysis patterns
PRESETS = {
    'all': {
        'types': ['knowledge', 'logic_puzzle', 'math'],
        'output_dir': 'figures/trends_eco_all',
        'description': 'Analyze all prompt types'
    },
    'knowledge': {
        'types': ['knowledge'],
        'output_dir': 'figures/trends_eco_knowledge', 
        'description': 'Analyze knowledge prompts only'
    },
    'logic_puzzle': {
        'types': ['logic_puzzle'],
        'output_dir': 'figures/trends_eco_logic_puzzles',
        'description': 'Analyze logic puzzle prompts only'
    },
    'math': {
        'types': ['math'],
        'output_dir': 'figures/trends_eco_math',
        'description': 'Analyze math prompts only'
    },
    'reasoning': {
        'types': ['logic_puzzle', 'math'],
        'output_dir': 'figures/trends_eco_reasoning',
        'description': 'Analyze reasoning-heavy prompts (logic puzzles + math)'
    },
    'knowledge_math': {
        'types': ['knowledge', 'math'],
        'output_dir': 'figures/trends_eco_knowledge_math',
        'description': 'Analyze knowledge and math prompts'
    }
}


def extract_lab_from_model_path(model_path: str) -> str:
    """Extract lab/organization from model path."""
    # Special case for models containing "hermes" (case insensitive)
    if "hermes" in model_path.lower():
        return "nous"
    
    # Standard case: extract lab from path (e.g., "mistralai/model" -> "mistralai")
    if "/" in model_path:
        return model_path.split("/")[0]
    
    # Fallback: return "unknown" if no "/" separator found
    return "unknown"


def extract_model_name(model_path: str) -> str:
    """Extract clean model name from model path."""
    # Remove lab prefix if present
    if "/" in model_path:
        model_name = model_path.split("/", 1)[1]
    else:
        model_name = model_path
    
    # Clean up common suffixes and prefixes
    model_name = model_name.replace('-instruct', '').replace('-chat', '')
    model_name = model_name.replace('_instruct', '').replace('_chat', '')
    
    return model_name


def create_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")


def load_evaluation_stats(csv_path: Path) -> pd.DataFrame:
    """Load evaluation statistics from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded evaluation stats with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading evaluation stats: {e}")
        sys.exit(1)


def determine_calculation_method(df: pd.DataFrame, selected_types: Set[str]) -> Dict[str, str]:
    """Determine which calculation method to use for each LLM based on data validity."""
    # Filter for selected types
    type_conditions = [df['type'] == TYPE_MAPPINGS[t] for t in selected_types]
    filtered_df = df[pd.concat(type_conditions, axis=1).any(axis=1)].copy()
    
    method_per_llm = {}
    
    for model_name in filtered_df['model_name'].unique():
        model_data = filtered_df[filtered_df['model_name'] == model_name]
        if model_data.empty:
            continue
            
        # Check if token method is valid
        token_method_valid = True
        has_nonzero_reasoning = False
        
        for _, row in model_data.iterrows():
            tokens_reasoning = row.get('tokens_reasoning', 0)
            tokens_completion = row.get('tokens_completion', 0)
            
            # Check if we have valid token data
            if (isinstance(tokens_reasoning, (int, float)) and 
                isinstance(tokens_completion, (int, float)) and tokens_completion > 0):
                
                # Ensure tokens_reasoning is not negative
                if tokens_reasoning > 0:
                    has_nonzero_reasoning = True
                    # If tokens_reasoning > tokens_completion, invalid
                    if tokens_reasoning > tokens_completion:
                        token_method_valid = False
                        break
        
        # Decide method
        if token_method_valid and has_nonzero_reasoning:
            method_per_llm[model_name] = "tokens"
            print(f"Using token method for {model_name}")
        elif not has_nonzero_reasoning:
            method_per_llm[model_name] = "chars"
            print(f"Using character method for {model_name} (tokens_reasoning is zero)")
        else:
            method_per_llm[model_name] = "chars"
            print(f"Using character method for {model_name} (tokens_reasoning > tokens_completion detected)")
    
    return method_per_llm

def calculate_reasoning_ratios(metrics: Dict[str, Any], method: str) -> List[float]:
    """Calculate reasoning ratios for each individual run using the specified method."""
    ratios = []
    
    for i in range(len(metrics['completion_tokens'])):
        if method == "tokens":
            # Method 1: Use tokens_reasoning / tokens_completion
            if (i < len(metrics['tokens_reasoning']) and 
                metrics['tokens_reasoning'] and 
                isinstance(metrics['tokens_reasoning'][i], (int, float)) and
                metrics['tokens_reasoning'][i] > 0):
                
                tokens_reasoning = float(metrics['tokens_reasoning'][i])
                tokens_completion = float(metrics['completion_tokens'][i])
                if tokens_completion > 0:
                    ratio = tokens_reasoning / tokens_completion
                    ratios.append(ratio)
                else:
                    ratios.append(0.0)
            else:
                ratios.append(0.0)
                
        else:  # method == "chars"
            # Method 2: Use char_reasoning / char_completion
            if (i < len(metrics['char_reasoning']) and 
                i < len(metrics['char_completion']) and
                metrics['char_reasoning'] and 
                metrics['char_completion'] and
                isinstance(metrics['char_reasoning'][i], (int, float)) and
                isinstance(metrics['char_completion'][i], (int, float))):
                
                char_reasoning = float(metrics['char_reasoning'][i])
                char_completion = float(metrics['char_completion'][i])
                if char_completion > 0:
                    ratio = char_reasoning / char_completion
                    ratios.append(ratio)
                else:
                    ratios.append(0.0)
            else:
                ratios.append(0.0)
    
    return ratios

def load_and_process_data(df: pd.DataFrame, selected_types: Set[str]) -> List[Dict[str, Any]]:
    """Load and process CSV data, extracting metrics for selected prompt types only."""
    all_metrics = []
    
    # Filter for selected types
    type_conditions = [df['type'] == TYPE_MAPPINGS[t] for t in selected_types]
    filtered_df = df[pd.concat(type_conditions, axis=1).any(axis=1)].copy()
    
    print(f"Filtered to {len(filtered_df)} results for types: {', '.join(selected_types)}")
    
    # Determine calculation method for each LLM
    print(f"\nDetermining calculation methods for each LLM...")
    method_per_llm = determine_calculation_method(df, selected_types)
    
    # Group by model and prompt_id to aggregate multiple runs
    grouped = filtered_df.groupby(['model_name', 'prompt_id'])
    
    for (model_name, prompt_id), group in grouped:
        # Extract metrics for this model/prompt combination
        completion_tokens = group['tokens_completion'].tolist()
        tokens_reasoning = group['tokens_reasoning'].tolist()
        tokens_output = group['tokens_output'].tolist()
        char_output = group['char_output'].tolist()
        char_reasoning = group['char_reasoning'].tolist()
        char_completion = group['char_completion'].tolist()
        success_rates = group['success_rate'].tolist()
        
        # Skip if no valid completion token data
        if not completion_tokens or not all(isinstance(t, (int, float)) and t > 0 for t in completion_tokens):
            continue
        
        # Get model configuration from the CSV data
        first_row = group.iloc[0]
        calculation_method = method_per_llm.get(model_name, "chars")
        
        # Get lab from CSV data with fallback to extraction from model name
        csv_lab = first_row.get('lab', 'unknown')
        
        # If the lab column contains a model name instead of a lab name, extract from model_name
        known_labs = {'openai', 'google', 'anthropic', 'mistralai', 'deepseek', 'minimax', 
                     'qwen', 'nous', 'nvidia', 'x-ai', 'z-ai'}
        
        if csv_lab in known_labs:
            lab = csv_lab
        else:
            # Fallback: extract lab from model name
            lab = extract_lab_from_model_path(model_name)
        
        clean_model_name = extract_model_name(model_name)
        
        metrics = {
            'llm': model_name,
            'lab': lab,
            'clean_model_name': clean_model_name,
            'prompt_id': prompt_id,
            'type': first_row['type'],
            'completion_tokens': completion_tokens,
            'tokens_reasoning': tokens_reasoning,
            'tokens_output': tokens_output,
            'char_output': char_output,
            'char_reasoning': char_reasoning,
            'char_completion': char_completion,
            'success_rates': success_rates,
            'full_cot': first_row['full_cot'],
            'open_weights': first_row['open_weights'],
            'total_responses': len(completion_tokens),
            'calculation_method': calculation_method
        }
        
        # Calculate reasoning ratios using the determined method
        metrics['reasoning_ratios'] = calculate_reasoning_ratios(metrics, calculation_method)
        
        all_metrics.append(metrics)
    
    print(f"\nSummary:")
    print(f"Total processed results: {len(all_metrics)}")
    print(f"Unique models: {len(set(m['llm'] for m in all_metrics))}")
    print(f"Unique labs: {len(set(m['lab'] for m in all_metrics))}")
    print(f"Unique prompts: {len(set(m['prompt_id'] for m in all_metrics))}")
    
    return all_metrics

def get_analysis_title_suffix(selected_types: Set[str]) -> str:
    """Get appropriate title suffix based on selected types."""
    if len(selected_types) == 1:
        return TYPE_DISPLAY_NAMES[list(selected_types)[0]]
    elif len(selected_types) == len(TYPE_MAPPINGS):
        return "All Prompt Types"
    else:
        type_names = [TYPE_DISPLAY_NAMES[t] for t in sorted(selected_types)]
        return " + ".join(type_names)

def wrap_model_name(name: str, max_chars_per_line: int = 12) -> str:
    """Wrap long model names into two lines."""
    if len(name) <= max_chars_per_line:
        return name
    
    # Try to find a good break point (hyphen, underscore, or space)
    break_chars = ['-', '_', ' ']
    best_break = -1
    
    for char in break_chars:
        pos = name.rfind(char, 0, max_chars_per_line + 3)  # Allow slight overflow for better breaks
        if pos > max_chars_per_line // 2:  # Don't break too early
            best_break = pos
            break
    
    if best_break > 0:
        return name[:best_break] + '\n' + name[best_break + 1:]
    else:
        # No good break point found, split at max_chars_per_line
        return name[:max_chars_per_line] + '\n' + name[max_chars_per_line:]

def create_grouped_token_composition_chart(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                         figsize: Tuple[float, float], title_suffix: str) -> None:
    """Create grouped bar chart showing reasoning and output token composition by lab and model."""
    
    # Aggregate data by lab and model
    lab_model_data = defaultdict(lambda: defaultdict(lambda: {
        'completion_tokens': [],
        'reasoning_tokens': [],
        'output_tokens': [],
        'open_weights': False,
        'calculation_method': 'chars'
    }))
    
    for entry in metrics_data:
        lab = entry['lab']
        clean_model = entry['clean_model_name']
        
        lab_model_data[lab][clean_model]['open_weights'] = entry['open_weights']
        lab_model_data[lab][clean_model]['calculation_method'] = entry['calculation_method']
        
        # Calculate reasoning and output tokens based on method
        for i in range(len(entry['completion_tokens'])):
            completion_tokens = entry['completion_tokens'][i]
            
            if entry['calculation_method'] == 'tokens':
                # Use direct token values
                if (i < len(entry['tokens_reasoning']) and 
                    i < len(entry['tokens_output']) and
                    entry['tokens_reasoning'] and 
                    entry['tokens_output']):
                    reasoning_tokens = entry['tokens_reasoning'][i]
                    output_tokens = entry['tokens_output'][i]
                else:
                    # Fallback if token data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            else:
                # Calculate from character ratios
                if (i < len(entry['char_reasoning']) and 
                    i < len(entry['char_completion']) and
                    entry['char_reasoning'] and 
                    entry['char_completion']):
                    char_reasoning = entry['char_reasoning'][i]
                    char_completion = entry['char_completion'][i]
                    if char_completion > 0:
                        reasoning_ratio = char_reasoning / char_completion
                        reasoning_tokens = completion_tokens * reasoning_ratio
                        output_tokens = completion_tokens - reasoning_tokens
                    else:
                        reasoning_tokens = 0
                        output_tokens = completion_tokens
                else:
                    # Fallback if character data missing
                    reasoning_tokens = 0
                    output_tokens = completion_tokens
            
            lab_model_data[lab][clean_model]['completion_tokens'].append(completion_tokens)
            lab_model_data[lab][clean_model]['reasoning_tokens'].append(reasoning_tokens)
            lab_model_data[lab][clean_model]['output_tokens'].append(output_tokens)
    
    # Calculate averages for each lab/model combination
    lab_model_averages = {}
    for lab, models in lab_model_data.items():
        lab_model_averages[lab] = {}
        for model, data in models.items():
            if data['completion_tokens']:
                lab_model_averages[lab][model] = {
                    'completion': np.mean(data['completion_tokens']),
                    'reasoning': np.mean(data['reasoning_tokens']),
                    'output': np.mean(data['output_tokens']),
                    'open_weights': data['open_weights']
                }
    
    # Sort labs and models for consistent display
    sorted_labs = sorted(lab_model_averages.keys())
    
    # Filter out labs with only one model for display purposes
    labs_to_display = [lab for lab in sorted_labs if len(lab_model_averages[lab]) > 1]
    
    # Calculate max completion value for label positioning (include all labs in calculation)
    all_completion_values = []
    for lab_models in lab_model_averages.values():
        for model_data in lab_models.values():
            all_completion_values.append(model_data['completion'])
    max_completion = max(all_completion_values) if all_completion_values else 0
    
    # Create figure with proper dimensions
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions for grouped bars (only for labs to display)
    lab_positions = np.arange(len(labs_to_display))
    bar_width = 0.8
    
    # Colors for open/closed weights
    colors_reasoning_open = 'darkblue'
    colors_output_open = 'lightblue'
    colors_reasoning_closed = 'darkred'
    colors_output_closed = 'lightcoral'
    
    # Track if we've added legend entries for each combination
    added_reasoning_open = False
    added_reasoning_closed = False
    added_output_open = False
    added_output_closed = False
    
    # Plot grouped bars for each lab (only labs with multiple models)
    for i, lab in enumerate(labs_to_display):
        models = lab_model_averages[lab]
        
        # Sort models: alphabetically for specific labs, by completion tokens for others
        if lab.lower() in ['qwen', 'openai', 'deepseek']:
            sorted_models = sorted(models.keys())
        else:
            sorted_models = sorted(models.keys(), key=lambda x: models[x]['completion'], reverse=True)
        
        if not sorted_models:
            continue
            
        # Calculate positions for models within this lab
        n_models = len(sorted_models)
        model_width = bar_width / n_models
        model_positions = lab_positions[i] + np.linspace(-bar_width/2 + model_width/2, 
                                                        bar_width/2 - model_width/2, 
                                                        n_models)
        
        # Add lab label above the bars
        max_height_in_group = max([models[model]['completion'] for model in sorted_models])
        ax.text(lab_positions[i], max_height_in_group + max_completion * 0.05, lab, 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        for j, model in enumerate(sorted_models):
            data = models[model]
            reasoning_val = data['reasoning']
            output_val = data['output']
            completion_val = data['completion']
            is_open = data['open_weights']
            
            # Choose colors based on open/closed weights
            reasoning_color = colors_reasoning_open if is_open else colors_reasoning_closed
            output_color = colors_output_open if is_open else colors_output_closed
            
            # Determine legend labels
            reasoning_label = ""
            output_label = ""
            
            if is_open:
                if not added_reasoning_open:
                    reasoning_label = "Reasoning (Open)"
                    added_reasoning_open = True
                if not added_output_open:
                    output_label = "Output (Open)"
                    added_output_open = True
            else:
                if not added_reasoning_closed:
                    reasoning_label = "Reasoning (Closed)"
                    added_reasoning_closed = True
                if not added_output_closed:
                    output_label = "Output (Closed)"
                    added_output_closed = True
            
            # Create stacked bars with outlines
            pos = model_positions[j]
            ax.bar(pos, reasoning_val, model_width, 
                  color=reasoning_color, alpha=0.8, edgecolor='black', linewidth=0.5,
                  label=reasoning_label)
            ax.bar(pos, output_val, model_width, bottom=reasoning_val,
                  color=output_color, alpha=0.8, edgecolor='black', linewidth=0.5,
                  label=output_label)
            
            # Add completion token value on top of the bar
            ax.text(pos, completion_val + max_completion * 0.01, f'{completion_val:.0f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            # Add model name labels below the chart area with line wrapping and 60° rotation
            wrapped_name = wrap_model_name(model)
            ax.text(pos, 0, wrapped_name, ha='right', va='top', fontsize=7, rotation=60,
                   transform=ax.get_xaxis_transform(), clip_on=False)

    # Print info about excluded single-model labs
    excluded_labs = [lab for lab in sorted_labs if len(lab_model_averages[lab]) == 1]
    if excluded_labs:
        print(f"Note: Excluded labs with single models from display: {', '.join(excluded_labs)}")
        print(f"(Data for these labs is still included in calculations)")

    # Customize chart
    ax.set_title(f'Token Composition by Lab and Model: {title_suffix} Prompts\n(Reasoning + Output Tokens)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Tokens', fontsize=12, fontweight='bold')
    ax.set_xticks([])  # Remove x-axis tick marks since we have labels above bars
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits: start exactly from 0 and add extra space for lab labels
    ax.set_ylim(0, max_completion * 1.2)
    
    # Create legend with larger size
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)  # Increase bottom margin for rotated model names
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created grouped token composition chart: {output_path.name}")

def create_grouped_relative_tokens_chart(metrics_data: List[Dict[str, Any]], output_path: Path, 
                                       figsize: Tuple[float, float], title_suffix: str,
                                       reference_models: List[str]) -> None:
    """Create grouped bar chart showing relative completion tokens by lab and model."""
    
    # First, organize data by prompt and model to get average completion tokens
    prompt_model_tokens = defaultdict(dict)
    model_lab_mapping = {}
    model_clean_name_mapping = {}
    model_open_weights = {}
    
    for entry in metrics_data:
        prompt_id = entry['prompt_id']
        llm = entry['llm']
        lab = entry['lab']
        clean_model = entry['clean_model_name']
        avg_completion_tokens = np.mean(entry['completion_tokens']) if entry['completion_tokens'] else 0
        
        prompt_model_tokens[prompt_id][llm] = avg_completion_tokens
        model_lab_mapping[llm] = lab
        model_clean_name_mapping[llm] = clean_model
        model_open_weights[llm] = entry['open_weights']
    
    # Calculate normalization factors per prompt based on reference models
    all_prompts = sorted(prompt_model_tokens.keys())
    prompt_normalization_factors = {}
    
    for prompt in all_prompts:
        reference_tokens = []
        
        for ref_model in reference_models:
            # Try exact match first, then partial match (case insensitive)
            ref_token_value = None
            
            # Try exact match first
            for model in prompt_model_tokens[prompt]:
                if model.lower() == ref_model.lower():
                    ref_token_value = prompt_model_tokens[prompt][model]
                    break
            
            # If no exact match, try partial match
            if ref_token_value is None:
                for model in prompt_model_tokens[prompt]:
                    if ref_model.lower() in model.lower() or model.lower() in ref_model.lower():
                        ref_token_value = prompt_model_tokens[prompt][model]
                        break
            
            if ref_token_value is not None and ref_token_value > 0:
                reference_tokens.append(ref_token_value)
        
        if reference_tokens:
            prompt_normalization_factors[prompt] = np.mean(reference_tokens)
        else:
            # Fallback: use mean of all models for this prompt
            all_tokens_for_prompt = [tokens for tokens in prompt_model_tokens[prompt].values() if tokens > 0]
            if all_tokens_for_prompt:
                prompt_normalization_factors[prompt] = np.mean(all_tokens_for_prompt)
            else:
                prompt_normalization_factors[prompt] = 1.0
    
    # Calculate average relative tokens for each model across all prompts
    model_avg_relative_tokens = {}
    all_models = set(model_lab_mapping.keys())
    
    for model in all_models:
        relative_tokens_per_prompt = []
        
        for prompt in all_prompts:
            absolute_tokens = prompt_model_tokens[prompt].get(model, 0)
            normalization_factor = prompt_normalization_factors[prompt]
            relative_tokens = absolute_tokens / normalization_factor if normalization_factor > 0 else 0
            if relative_tokens > 0:  # Only include prompts where this model has data
                relative_tokens_per_prompt.append(relative_tokens)
        
        if relative_tokens_per_prompt:
            model_avg_relative_tokens[model] = np.mean(relative_tokens_per_prompt)
        else:
            model_avg_relative_tokens[model] = 0
    
    # Group models by lab
    lab_model_data = defaultdict(dict)
    for model, relative_tokens in model_avg_relative_tokens.items():
        if relative_tokens > 0:  # Only include models with valid data
            lab = model_lab_mapping[model]
            clean_model = model_clean_name_mapping[model]
            lab_model_data[lab][clean_model] = {
                'relative_tokens': relative_tokens,
                'open_weights': model_open_weights[model],
                'full_model_name': model
            }
    
    # Sort labs and prepare for plotting
    sorted_labs = sorted(lab_model_data.keys())
    
    # Filter out labs with only one model for display purposes
    labs_to_display = [lab for lab in sorted_labs if len(lab_model_data[lab]) > 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate positions for grouped bars (only for labs to display)
    lab_positions = np.arange(len(labs_to_display))
    bar_width = 0.8
    
    # Colors for open/closed weights
    color_open = 'lightblue'
    color_closed = 'lightcoral'
    
    # Track if we've added legend entries
    added_open_legend = False
    added_closed_legend = False
    
    # Plot grouped bars for each lab (only labs with multiple models)
    for i, lab in enumerate(labs_to_display):
        models = lab_model_data[lab]
        
        # Sort models: alphabetically for specific labs, by relative tokens for others
        if lab.lower() in ['qwen', 'openai', 'deepseek']:
            sorted_models = sorted(models.keys())
        else:
            sorted_models = sorted(models.keys(), key=lambda x: models[x]['relative_tokens'], reverse=True)
        
        if not sorted_models:
            continue
            
        # Calculate positions for models within this lab
        n_models = len(sorted_models)
        model_width = bar_width / n_models
        model_positions = lab_positions[i] + np.linspace(-bar_width/2 + model_width/2, 
                                                        bar_width/2 - model_width/2, 
                                                        n_models)
        
        # Add lab label above the bars
        y_max = max([max([models[m]['relative_tokens'] for m in models]) 
                    for models in lab_model_data.values() if len(models) > 1])  # Only consider displayed labs for y_max
        max_height_in_group = max([models[model]['relative_tokens'] for model in sorted_models])
        ax.text(lab_positions[i], max_height_in_group + y_max * 0.05, lab, 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        for j, model in enumerate(sorted_models):
            data = models[model]
            relative_tokens = data['relative_tokens']
            is_open = data['open_weights']
            
            # Choose color based on open/closed weights
            color = color_open if is_open else color_closed
            
            # Determine legend label
            legend_label = ""
            if is_open and not added_open_legend:
                legend_label = "Open Weight"
                added_open_legend = True
            elif not is_open and not added_closed_legend:
                legend_label = "Closed Weight"
                added_closed_legend = True
            
            # Create bar
            pos = model_positions[j]
            ax.bar(pos, relative_tokens, model_width, 
                  color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
                  label=legend_label)
            
            # Add value label on bar
            ax.text(pos, relative_tokens + 0.02, f'{relative_tokens:.2f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            # Add model name labels below the chart area with line wrapping and 60° rotation
            wrapped_name = wrap_model_name(model)
            ax.text(pos, 0, wrapped_name, ha='right', va='top', fontsize=7, rotation=60,
                   transform=ax.get_xaxis_transform(), clip_on=False)

    # Print info about excluded single-model labs
    excluded_labs = [lab for lab in sorted_labs if len(lab_model_data[lab]) == 1]
    if excluded_labs:
        print(f"Note: Excluded labs with single models from display: {', '.join(excluded_labs)}")
        print(f"(Data for these labs is still included in calculations)")

    # Customize chart
    ax.set_title(f'Average Relative Completion Tokens by Lab and Model: {title_suffix} Prompts\n(Lower is More Token-Efficient)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Relative Completion Tokens', fontsize=12, fontweight='bold')
    ax.set_xticks([])  # Remove x-axis tick marks since we have labels above bars
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits: start exactly from 0 and add extra space for lab labels
    if labs_to_display:  # Only set y-axis if there are labs to display
        y_max = max([max([models[m]['relative_tokens'] for m in models]) 
                    for lab, models in lab_model_data.items() if lab in labs_to_display])
        ax.set_ylim(0, y_max * 1.2)
    
    # Create legend with larger size
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)  # Increase bottom margin for rotated model names
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created grouped relative tokens chart: {output_path.name}")

def save_summary_data(metrics_data: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary data to CSV file."""
    
    summary_data = []
    for entry in metrics_data:
        avg_completion_tokens = np.mean(entry['completion_tokens'])
        avg_reasoning_ratio = np.mean(entry['reasoning_ratios']) if entry['reasoning_ratios'] else 0
        avg_success_rate = np.mean(entry['success_rates']) if entry['success_rates'] else 0
        
        summary_data.append({
            'llm': entry['llm'],
            'lab': entry['lab'],
            'clean_model_name': entry['clean_model_name'],
            'prompt_id': entry['prompt_id'],
            'type': entry['type'],
            'open_weights': entry['open_weights'],
            'full_cot': entry['full_cot'],
            'avg_completion_tokens': avg_completion_tokens,
            'avg_reasoning_ratio': avg_reasoning_ratio,
            'avg_success_rate': avg_success_rate,
            'total_responses': entry['total_responses']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"Saved summary data: {output_path.name}")

def main():
    """Main function to analyze model trends grouped by lab."""
    # Build preset descriptions for help text
    preset_help = "\n".join([f"  {name}: {config['description']}" for name, config in PRESETS.items()])
    
    parser = argparse.ArgumentParser(
        description="Analyze model trends grouped by research lab with specific visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
This script creates two main visualizations:
1. Grouped Token Composition Chart: Token composition (reasoning + output) by lab and model
2. Grouped Relative Tokens Chart: Average relative completion tokens by lab and model

Available presets:
{preset_help}

Examples:
    python analyze_model_trends.py --preset all
    python analyze_model_trends.py --preset knowledge
    python analyze_model_trends.py --preset reasoning
    python analyze_model_trends.py --types knowledge,math --output-dir figures/trends_eco_knowledge_math
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preset', 
                       choices=list(PRESETS.keys()),
                       help='Use a predefined configuration preset')
    group.add_argument('--types', 
                       help='Comma-separated list of prompt types to analyze. Options: knowledge, logic_puzzle, math')
    
    parser.add_argument('--csv-file', default='evaluation_stats.csv',
                       help='Path to evaluation statistics CSV file (default: evaluation_stats.csv)')
    parser.add_argument('--output-dir',
                       help='Output directory for generated files (overrides preset default)')
    parser.add_argument('--figsize', default='10,6',
                       help='Figure size as width,height in inches (default: 10,6)')
    parser.add_argument('--reference-models', default='claude-4-sonnet-0522-thinking,o4-mini-high-long,gemini-2.5-flash-high',
                       help='Comma-separated list of reference models for token normalization (default: claude-4-sonnet-0522-thinking,o4-mini-high-long,gemini-2.5-flash-high)')
    
    args = parser.parse_args()
    
    # Handle preset vs types logic
    if args.preset:
        preset_config = PRESETS[args.preset]
        selected_types = set(preset_config['types'])
        output_dir_default = preset_config['output_dir']
        print(f"Using preset '{args.preset}': {preset_config['description']}")
    elif args.types:
        # Parse and validate types
        try:
            selected_types = set(t.strip() for t in args.types.split(','))
            invalid_types = selected_types - set(TYPE_MAPPINGS.keys())
            if invalid_types:
                print(f"Error: Invalid types: {invalid_types}")
                print(f"Valid types: {list(TYPE_MAPPINGS.keys())}")
                sys.exit(1)
        except Exception as e:
            print(f"Error parsing types: {e}")
            sys.exit(1)
        output_dir_default = 'figures/trends_eco_custom'
    else:
        # Default to 'all' preset if nothing specified
        preset_config = PRESETS['all']
        selected_types = set(preset_config['types'])
        output_dir_default = preset_config['output_dir']
        print(f"No preset or types specified, using default preset 'all': {preset_config['description']}")
    
    # Parse figure size
    try:
        width, height = map(float, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print("Error: Invalid figsize format. Use 'width,height' (e.g., '12,8')")
        sys.exit(1)
    
    # Parse reference models
    reference_models = [model.strip() for model in args.reference_models.split(',')]
    print(f"Reference models for token normalization: {reference_models}")
    
    # Validate inputs
    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir if args.output_dir else output_dir_default)
    
    # Validate paths exist
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        sys.exit(1)
    
    create_output_directory(output_dir)
    
    print(f"Starting model trends analysis for types: {', '.join(sorted(selected_types))}")
    print(f"CSV file: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Figure size: {figsize[0]}x{figsize[1]} inches")
    print()
    
    # Load evaluation statistics
    df = load_evaluation_stats(csv_path)
    
    # Process data and extract metrics
    print(f"\nExtracting metrics for selected prompt types...")
    metrics_data = load_and_process_data(df, selected_types)
    
    if not metrics_data:
        print("No metrics data extracted. Exiting.")
        sys.exit(1)
    
    # Get title suffix for analysis
    title_suffix = get_analysis_title_suffix(selected_types)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Grouped Token Composition Chart
    grouped_composition_path = output_dir / "grouped_token_composition_chart.png"
    create_grouped_token_composition_chart(metrics_data, grouped_composition_path, figsize, title_suffix)
    
    # 2. Grouped Relative Tokens Chart
    grouped_relative_path = output_dir / "grouped_relative_tokens_chart.png"
    create_grouped_relative_tokens_chart(metrics_data, grouped_relative_path, figsize, title_suffix, reference_models)
    
    # Save summary data
    types_suffix = "_".join(sorted(selected_types))
    summary_path = output_dir / f"{types_suffix}_trends_analysis_summary.csv"
    save_summary_data(metrics_data, summary_path)
    
    print(f"\nCompleted! Generated {title_suffix} model trends analysis in {output_dir}")
    print(f"  - 2 visualization files")
    print(f"  - 1 summary CSV file")
    print(f"  - Total data points analyzed: {len(metrics_data)}")
    
    # Show lab summary
    labs = set(m['lab'] for m in metrics_data)
    print(f"  - Labs analyzed: {', '.join(sorted(labs))}")

if __name__ == "__main__":
    main()