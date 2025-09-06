#!/usr/bin/env python3
"""
Aggregate LLM evaluation results into a statistical DataFrame for analysis.

This script processes evaluation experiment data and creates a consolidated
CSV file with statistical metrics for token efficiency analysis.
"""

import argparse
import json
import pandas as pd
import sys
import bz2
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate LLM evaluation results into statistical DataFrame"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./data",
        help="Directory containing evaluation result JSON files (default: ./data)"
    )
    
    parser.add_argument(
        "--evalset",
        type=str,
        default="./evalset/TokenEconomyDataset_V1.json",
        help="Path to evaluation dataset JSON file (default: ./evalset/TokenEconomyDataset_V1.json)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./query_config_full.json",
        help="Path to query configuration JSON file (default: ./query_config_full.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_stats.csv",
        help="Output CSV file path (default: ./evaluation_stats.csv)"
    )
    
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Validate that all required input paths exist."""
    results_dir = Path(args.results_dir)
    evalset_path = Path(args.evalset)
    config_path = Path(args.config)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    if not evalset_path.exists():
        print(f"Error: Evalset file not found: {evalset_path}")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)


def calculate_shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy for a text string."""
    if not text:
        return 0.0
    
    # Count character frequencies
    char_counts = Counter(text)
    text_length = len(text)
    
    # Calculate entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / text_length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_bzip2_compression(text: str, level: int = 9) -> int:
    """Calculate BZIP2 compressed length for a text string."""
    if not text:
        return 0
    
    # Encode text to bytes and compress
    text_bytes = text.encode('utf-8')
    compressed_bytes = bz2.compress(text_bytes, compresslevel=level)
    
    return len(compressed_bytes)


def calculate_compression_ratios(original_length: int, compressed_length: int, entropy: float) -> tuple[float, float]:
    """Calculate compression ratios for BZIP2 and theoretical (entropy-based)."""
    if original_length == 0:
        return 0.0, 0.0
    
    # BZIP2 compression ratio (compressed/original)
    bzip2_ratio = compressed_length / original_length if original_length > 0 else 0.0
    
    # Theoretical compression ratio based on Shannon entropy
    # Theoretical minimum bits per character = entropy
    # Theoretical minimum bytes = (entropy * char_count) / 8
    theoretical_min_bytes = (entropy * original_length) / 8 if entropy > 0 else 0
    entropy_ratio = theoretical_min_bytes / original_length if original_length > 0 else 0.0
    
    return bzip2_ratio, entropy_ratio


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


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)


def discover_data_files(results_dir: Path) -> Dict[str, List[Path]]:
    """Discover and categorize JSON files in the results directory."""
    data_files = {
        'detailed_evaluations': [],
        'output_queries': [],
        'other': []
    }
    
    for json_file in results_dir.glob("*.json"):
        filename = json_file.name.lower()
        if 'detailed_evaluations' in filename:
            data_files['detailed_evaluations'].append(json_file)
        elif 'output_queries' in filename:
            data_files['output_queries'].append(json_file)
        else:
            data_files['other'].append(json_file)
    
    return data_files


def load_all_evaluation_data(data_files: Dict[str, List[Path]]) -> Dict[str, Dict[str, Any]]:
    """Load all evaluation data files and combine them."""
    all_detailed_evaluations = {}
    all_output_queries = {"results": []}
    
    # Load detailed evaluations
    for file_path in data_files['detailed_evaluations']:
        print(f"Loading detailed evaluations from: {file_path.name}")
        data = load_json_file(file_path)
        
        # Count evaluations in this file
        file_eval_count = 0
        for prompt_id, model_results in data.items():
            for model_name, evaluations in model_results.items():
                file_eval_count += len(evaluations)
        print(f"  - File contains {file_eval_count} evaluations for {len(data)} prompts")
        
        # Merge data - if prompt exists, merge models; if model exists for same prompt, extend evaluations
        for prompt_id, model_results in data.items():
            if prompt_id not in all_detailed_evaluations:
                all_detailed_evaluations[prompt_id] = {}
            
            for model_name, evaluations in model_results.items():
                if model_name not in all_detailed_evaluations[prompt_id]:
                    all_detailed_evaluations[prompt_id][model_name] = []
                
                # Extend evaluations (in case multiple files have data for same prompt-model combination)
                all_detailed_evaluations[prompt_id][model_name].extend(evaluations)
    
    # Load output queries
    for file_path in data_files['output_queries']:
        print(f"Loading output queries from: {file_path.name}")
        data = load_json_file(file_path)
        if 'results' in data:
            all_output_queries['results'].extend(data['results'])
        else:
            # If the file structure is different, add the whole file as results
            all_output_queries['results'].extend([data])
    
    return {
        'detailed_evaluations': all_detailed_evaluations,
        'output_queries': all_output_queries
    }


def create_prompt_metadata_dict(evalset_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create a lookup dictionary for prompt metadata."""
    prompt_metadata = {}
    for prompt in evalset_data.get('prompts', []):
        prompt_metadata[prompt['prompt_id']] = {
            'category': prompt.get('category', ''),
            'type': prompt.get('type', ''),
            'title': prompt.get('title', ''),
            'criteria_count': len(prompt.get('criteria', [])),
            'weight_sum': sum(prompt.get('weight', []))
        }
    return prompt_metadata


def create_model_config_dict(config_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create a lookup dictionary for model configurations."""
    model_config = {}
    for model in config_data.get('llms', []):
        model_path = model.get('model', '')
        model_config[model['name']] = {
            'model_path': model_path,
            'lab': extract_lab_from_model_path(model_path),
            'open_weights': model.get('open_weights', False),
            'full_cot': model.get('full_cot', False),
            'max_tokens': model.get('max_tokens', 0),
            'temperature': model.get('temperature', 0.0),
            'top_p': model.get('top_p', 1.0),
            'repetition_penalty': model.get('repetition_penalty', 1.0),
            'reasoning_effort': model.get('reasoning', {}).get('effort', ''),
            'reasoning_exclude': model.get('reasoning', {}).get('exclude', False)
        }
    return model_config


def filter_llms_by_config(detailed_evaluations: Dict[str, Any], 
                         model_config: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, Any], List[str], List[str]]:
    """Filter LLMs based on what's available in model config and return verification lists."""
    configured_llms = set(model_config.keys())
    found_llms = set()
    
    # Collect all LLMs found in the evaluation data
    for prompt_id, model_results in detailed_evaluations.items():
        found_llms.update(model_results.keys())
    
    # Determine included and rejected LLMs
    included_llms = list(found_llms.intersection(configured_llms))
    rejected_llms = list(found_llms - configured_llms)
    
    # Filter the detailed evaluations to only include configured LLMs
    filtered_evaluations = {}
    for prompt_id, model_results in detailed_evaluations.items():
        filtered_model_results = {
            model_name: evaluations 
            for model_name, evaluations in model_results.items() 
            if model_name in configured_llms
        }
        if filtered_model_results:  # Only include prompts that have results for configured models
            filtered_evaluations[prompt_id] = filtered_model_results
    
    return filtered_evaluations, sorted(included_llms), sorted(rejected_llms)


def filter_prompts_by_evalset(detailed_evaluations: Dict[str, Any], 
                             prompt_metadata: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, Any], List[str], List[str]]:
    """Filter prompts based on what's available in the evalset and return verification lists."""
    evalset_prompts = set(prompt_metadata.keys())
    found_prompts = set(detailed_evaluations.keys())
    
    # Determine included and rejected prompts
    included_prompts = list(found_prompts.intersection(evalset_prompts))
    rejected_prompts = list(found_prompts - evalset_prompts)
    
    # Filter the detailed evaluations to only include prompts from evalset
    filtered_evaluations = {
        prompt_id: model_results 
        for prompt_id, model_results in detailed_evaluations.items() 
        if prompt_id in evalset_prompts
    }
    
    return filtered_evaluations, sorted(included_prompts), sorted(rejected_prompts)


def create_text_lookup(output_queries: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Create a lookup dictionary for text content (output, thinking) and latency from output queries.

    Structure returned:
    {
        prompt_id: {
            model_name: {
                'output': [...],
                'thinking': [...],
                'latency_ms': [...]
            }
        }
    }
    """
    text_lookup = {}
    
    for result in output_queries.get('results', []):
        prompt_id = result.get('prompt_id')
        model_name = result.get('llm')
        output_texts = result.get('output', [])
        thinking_texts = result.get('thinking', [])
        latency_values = result.get('latency_ms', [])

        if prompt_id not in text_lookup:
            text_lookup[prompt_id] = {}
        if model_name not in text_lookup[prompt_id]:
            text_lookup[prompt_id][model_name] = {}

        text_lookup[prompt_id][model_name] = {
            'output': output_texts,
            'thinking': thinking_texts,
            'latency_ms': latency_values
        }
    
    return text_lookup


def process_detailed_evaluations(detailed_evaluations: Dict[str, Any], 
                                prompt_metadata: Dict[str, Dict[str, Any]],
                                model_config: Dict[str, Dict[str, Any]],
                                text_lookup: Dict[str, Dict[str, Dict[str, List[str]]]]) -> List[Dict[str, Any]]:
    """Process detailed evaluations into a list of statistical records."""
    records = []
    
    for prompt_id, model_results in detailed_evaluations.items():
        for model_name, evaluations in model_results.items():
            for run_idx, evaluation in enumerate(evaluations):
                # Get prompt metadata
                prompt_meta = prompt_metadata.get(prompt_id, {})
                model_meta = model_config.get(model_name, {})
                
                # Extract criteria results
                criteria_results = evaluation.get('criteria_results', [])
                criteria_met = sum(1 for cr in criteria_results if cr.get('met', False))
                criteria_total = len(criteria_results)
                
                # Extract statistics
                stats = evaluation.get('statistics', {})
                
                # Calculate derived metrics
                tokens_output = stats.get('tokens_output', 0)
                tokens_reasoning = stats.get('tokens_reasoning', 0)
                tokens_completion = stats.get('tokens_completions', 0)
                
                char_output = stats.get('character_count_output', 0)
                char_reasoning = stats.get('character_count_reasoning', 0)
                char_completion = stats.get('character_count_completion', 0)
                
                # Special handling for Claude models
                if 'claude' in model_name.lower():
                    # For Claude models, compute tokens from character data
                    tokens_output = char_output / 3.1
                    tokens_reasoning = tokens_completion - tokens_output
                    # Ensure tokens_reasoning is not negative
                    if tokens_reasoning < 0:
                        tokens_reasoning = 0
                    assert(tokens_reasoning <= tokens_completion), \
                        f"Invalid reasoning tokens for {model_name} on prompt {prompt_id}: reasoning={tokens_reasoning} should be <= completion={tokens_completion} (output={tokens_output})"
                                                                                            
                
                total_score = evaluation.get('total_score', 0.0)
                success_rate = criteria_met / max(criteria_total, 1)
                
                # Get text content for entropy and compression analysis
                output_text = ""
                thinking_text = ""
                
                latency_value = None
                if (prompt_id in text_lookup and 
                    model_name in text_lookup[prompt_id]):

                    output_list = text_lookup[prompt_id][model_name].get('output', [])
                    thinking_list = text_lookup[prompt_id][model_name].get('thinking', [])
                    latency_list = text_lookup[prompt_id][model_name].get('latency_ms', [])

                    # Get the text for this specific run
                    if run_idx < len(output_list):
                        output_text = output_list[run_idx] or ""
                    if run_idx < len(thinking_list):
                        thinking_text = thinking_list[run_idx] or ""
                    if run_idx < len(latency_list):
                        latency_value = latency_list[run_idx]
                
                # Calculate entropy metrics
                output_entropy = calculate_shannon_entropy(output_text)
                thinking_entropy = calculate_shannon_entropy(thinking_text)
                
                # Calculate compression metrics
                output_compressed_length = calculate_bzip2_compression(output_text)
                thinking_compressed_length = calculate_bzip2_compression(thinking_text)
                
                # Calculate compression ratios
                output_bzip2_ratio, output_entropy_ratio = calculate_compression_ratios(
                    len(output_text), output_compressed_length, output_entropy
                )
                thinking_bzip2_ratio, thinking_entropy_ratio = calculate_compression_ratios(
                    len(thinking_text), thinking_compressed_length, thinking_entropy
                )
                
                record = {
                    # Prompt information
                    'prompt_id': prompt_id,
                    'category': prompt_meta.get('category', ''),
                    'type': prompt_meta.get('type', ''),
                    'title': prompt_meta.get('title', ''),
                    
                    # Model information
                    'model_name': model_name,
                    'lab': model_meta.get('lab', 'unknown'),
                    'run_number': run_idx + 1,
                    'open_weights': model_meta.get('open_weights', False),
                    'full_cot': model_meta.get('full_cot', False),
                    'max_tokens': model_meta.get('max_tokens', 0),
                    
                    # Performance metrics
                    'total_score': total_score,
                    'criteria_met_count': criteria_met,
                    'criteria_total_count': criteria_total,
                    'success_rate': success_rate,
                    
                    # Token statistics
                    'tokens_output': tokens_output,
                    'tokens_reasoning': tokens_reasoning,
                    'tokens_completion': tokens_completion,
                    'char_output': char_output,
                    'char_reasoning': char_reasoning,
                    'char_completion': char_completion,
                    
                    # Output entropy and compression metrics
                    'output_entropy': output_entropy,
                    'output_bzip2_length': output_compressed_length,
                    'output_bzip2_ratio': output_bzip2_ratio,
                    'output_entropy_ratio': output_entropy_ratio,
                    
                    # Thinking entropy and compression metrics
                    'thinking_entropy': thinking_entropy,
                    'thinking_bzip2_length': thinking_compressed_length,
                    'thinking_bzip2_ratio': thinking_bzip2_ratio,
                    'thinking_entropy_ratio': thinking_entropy_ratio,

                    # Latency (ms) for this run (may be None if not recorded)
                    'latency_ms': latency_value
                }
                
                records.append(record)
    
    return records


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("Validating input paths...")
    validate_paths(args)
    
    print("Discovering data files...")
    results_dir = Path(args.results_dir)
    data_files = discover_data_files(results_dir)
    
    print(f"Found {len(data_files['detailed_evaluations'])} detailed evaluation files")
    print(f"Found {len(data_files['output_queries'])} output query files") 
    print(f"Found {len(data_files['other'])} other JSON files")
    
    print("Loading data files...")
    
    # Load all required JSON files
    evalset_data = load_json_file(Path(args.evalset))
    config_data = load_json_file(Path(args.config))
    
    # Load and combine all evaluation data
    evaluation_data = load_all_evaluation_data(data_files)
    detailed_evaluations = evaluation_data['detailed_evaluations']
    output_queries = evaluation_data['output_queries']
    
    print(f"Loaded {len(evalset_data.get('prompts', []))} prompts from evalset")
    print(f"Loaded {len(config_data.get('llms', []))} model configurations")
    print(f"Loaded detailed evaluations for {len(detailed_evaluations)} prompts")
    print(f"Loaded output queries with {len(output_queries.get('results', []))} entries")
    
    # Count total evaluations in combined data
    total_evaluations = 0
    all_models = set()
    for prompt_id, model_results in detailed_evaluations.items():
        for model_name, evaluations in model_results.items():
            total_evaluations += len(evaluations)
            all_models.add(model_name)
    
    print(f"Total combined evaluations: {total_evaluations}")
    print(f"Unique models in data: {len(all_models)} - {sorted(all_models)}")
    
    print("Processing data...")
    
    # Create lookup dictionaries
    prompt_metadata = create_prompt_metadata_dict(evalset_data)
    model_config = create_model_config_dict(config_data)
    text_lookup = create_text_lookup(output_queries)
    
    # Filter LLMs based on configuration and get verification lists
    filtered_evaluations, included_llms, rejected_llms = filter_llms_by_config(detailed_evaluations, model_config)
    
    # Filter prompts based on evalset and get verification lists
    filtered_evaluations, included_prompts, rejected_prompts = filter_prompts_by_evalset(filtered_evaluations, prompt_metadata)
    
    # Print verification lists
    print(f"\n=== LLM Filtering Results ===")
    print(f"Included LLMs ({len(included_llms)}):")
    for llm in included_llms:
        print(f"  ✓ {llm}")
    
    if rejected_llms:
        print(f"\nRejected LLMs ({len(rejected_llms)}):")
        for llm in rejected_llms:
            print(f"  ✗ {llm}")
    else:
        print(f"\nRejected LLMs (0): None")
    
    print(f"\n=== Prompt Filtering Results ===")
    print(f"Included Prompts ({len(included_prompts)}):")
    for prompt in included_prompts:
        print(f"  ✓ {prompt}")
    
    if rejected_prompts:
        print(f"\nRejected Prompts ({len(rejected_prompts)}):")
        for prompt in rejected_prompts:
            print(f"  ✗ {prompt}")
    else:
        print(f"\nRejected Prompts (0): None")
    print("=" * 30)
    
    # Process detailed evaluations into statistical records
    records = process_detailed_evaluations(filtered_evaluations, prompt_metadata, model_config, text_lookup)
    
    print(f"Created {len(records)} evaluation records from filtered data")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    print("DataFrame created with columns:", list(df.columns))
    print(f"DataFrame shape: {df.shape}")
    
    # Format numeric columns appropriately
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            # Check if column has only integer values
            if df[col].notna().all() and (df[col] % 1 == 0).all():
                # Convert to int if all values are whole numbers
                df[col] = df[col].astype('int64')
            else:
                # Round to 5 decimal places for float values
                df[col] = df[col].round(5)
    
    # Save to CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to: {output_path}")
    print("Sample of first few rows:")
    print(df.head())


if __name__ == "__main__":
    main()