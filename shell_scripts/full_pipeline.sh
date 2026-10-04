## Collect the parameters (all on one line, no line breaks)
DEFAULT_PARAMS="--use_parent_techniques --dspy_module chain_of_thought"

# Delete content potential content in tmp output
# rm -rf models/tmp_output/*
#
# The Methodology Mappers
echo "Running vulnerability_type"
python -m run_scripts.run_vulnerability_type \
$DEFAULT_PARAMS "$@" &

PID1=$!

echo "Running exploitation_technique method"
python -m run_scripts.run_job_exploit_method $DEFAULT_PARAMS "$@" &

PID2=$!

echo "Running affected object types"
python -m run_scripts.run_exp_affected_object $DEFAULT_PARAMS "$@" &

PID3=$!

# In-Context Learner
echo "Running in_context_ranking - exploitation_technique"
python -m run_scripts.run_in_context_learner \
--include_in_context_prompt --include_cvss \
--include_cwe --mapping_type "exploitation_technique" --use_ranking_approach $DEFAULT_PARAMS "$@" &

PID4=$!


echo "Running in_context_ranking - primary_impact"
python -m run_scripts.run_in_context_learner \
--include_in_context_prompt --include_cvss \
--include_cwe --mapping_type "primary_impact" --use_ranking_approach $DEFAULT_PARAMS "$@" &

PID5=$!

echo "Running in_context_ranking - secondary_impact"
python -m run_scripts.run_in_context_learner \
--include_in_context_prompt --include_cvss \
--include_cwe --mapping_type "secondary_impact" --use_ranking_approach $DEFAULT_PARAMS "$@" &

PID6=$!

# Wait for both background processes to finish
wait $PID1
wait $PID2
wait $PID3
wait $PID4
wait $PID5
wait $PID6

# Final step. Combining the predictions.
echo "Running combined_approach - exploitation_technique"
python -m run_scripts.run_combine_approach --mapping_type "exploitation_technique" --include_vul_type \
--include_in_context_learner --include_affected_object --include_exploitation_method $DEFAULT_PARAMS "$@"

echo "Running combined_approach - primary_impact"
python -m run_scripts.run_combine_approach --mapping_type "primary_impact" --include_vul_type \
--include_in_context_learner $DEFAULT_PARAMS "$@"

echo "Running combined_approach - secondary_impact"
python -m run_scripts.run_combine_approach --mapping_type "secondary_impact" --include_vul_type \
--include_in_context_learner $DEFAULT_PARAMS "$@"

# rm -rf models/tmp_output/*