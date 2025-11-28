
#######################################################################
## Danger: This script will aggregate all results from /data and
## Generate figures in the figure/ folder. Everything will be overwritten
#######################################################################


#######################################################################
# Generate figures with all llms
python aggregate_results.py --config query_config_full.json

# CoT transscipriotn figures in cot_transcription/
python analyze_cot_transcription.py

# FIgures for eco_all
python analyze_prompts.py --preset all

#######################################################################
# Generate figures with only the most recent models
python aggregate_results.py --config query_config_recent.json

# Cost related figures /cost
python analyze_cost.py 

# Figures for eco_xxx/ (excecpt eco_all/)
python analyze_prompts.py --preset math
python analyze_prompts.py --preset knowledge
python analyze_prompts.py --preset logic_puzzle

#######################################################################
# Generate figure with historical trends
python aggregate_results.py --config query_config_trends.json

# Model trend charts in trends_eco_xxxx/
python analyze_model_trends.py --preset all
python analyze_model_trends.py --preset knowledge
python analyze_model_trends.py --preset math
python analyze_model_trends.py --preset logic_puzzle







