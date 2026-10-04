### Organization
- `data/external/`: Exported experiment data tracked with Neptune.
  - `results_exact_match.csv `: Unranked categorized mapping.
  - `results_ranking_approach.csv`: Ranked categorized mapping.
  - `results_uncategorized.csv`: Uncategorized mapping.
- `data/input_data/`: Input data for experiments.
  - `kev.csv`: The original data (version: 02.13.2025) provided by MITRE mapping CVEs to 
    techniques. 
- `data/post_processed/tables_data/`: CSV files to produce tables in the paper.
  - `exp_classwise.csv`: Per-attack technique recall@10 and support for the "exploitation 
    technique" mapping type, for both train and test splits.
  - `sec_classwise.csv`: Per-attack technique recall@10 and support for the "secondary impact" mapping type, for both train and test splits.
  - `prim_classwise.csv`: Per-attack technique recall@10 and support for the "primary impact" mapping type, for both train and test splits.
  - `technique_value_counts.csv`: Value counts of attack techniques (by ID) in the labeled dataset.
  - `unranked_classification.csv`: Unranked Classification with Gpt-4o-mini. Shows individual 
    Methodology Mappers, and In-Context Learner.
  - `ranking_approach_models.csv`: GPT-4o-mini vs Llama3.3-70B on the In-Context Learner and 
    the Complete Triage Approach. Mean Average Precision (MAP) and recall-at-N metrics for ranking approach experiments, by model and mapping type, for both train and test splits.
  - `ranking_approach_ablations.csv`: Ablation study results for the In-Context Learner approach, 
    showing MAP and Recall@N (5/10) for different feature sets and number of demonstrations.
  - `mapping_types_per_cve.csv`: Frequency of mapping types per CVE.
- `data/pre-processed/`: Pre-processed data from MITRE ATT&CK, NVD, CWE, and 
    CVE Mapping Methodology (CMM). Labeled data and associated data splits.
  - Individual files in these folders are described in data/README.md.
  individual files in these folders are described in data/README.md
  - `affected_object.json`: Technique mappings based on affected object types.
  - `attack_techniques_base.csv`: Relevant data for all enterprise techniques.
  - `cve_base.csv`: CVEs considered for the project with relevant data.
  - `cves_test.csv`: Test data split.
  - `cves_train.csv`: Train data split.
  - `cwe_id_to_description.csv`: CWE descriptions.
  - `cwe_id_to_name.csv`: CWE names.
  - `exploitation_techniques_mapping.json`: CMM mapping - Exploitation technique method.
  - `functionality_labels.csv`: Labels using the CMM functionality mapping. Source: https://github.com/ehsanaghaei/CVE2TTP/blob/main/CVE2FUNC_appendix.pdf
  - `functionality_mapping.csv`: CMM mapping - Functionality method.
  - `labeled_cve_to_attack.csv`: CVEs labeled with mapping types and attack techniques
  - `methodology_attack_techniques.csv`: Attack techniques specified in CMM.
  - `tactic_technique_mapping.csv`: Mapping from tactic to "tactic level technique".
  - `vul_type_to_cwe.csv`: Mapping from vulnerability types and CWEs.
  - `vulnerability_type_mapping.csv`: Mapping from vulnerability types to attack techniques.