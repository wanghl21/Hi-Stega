
for alg in ac hc flc adg adgv2; do
  python generate_gpt2_keyword_v3.py --mode G --config_path generate_configs/Yahoo-gpt-generate_keyword_prompt_${alg}.ini
done
for alg in ac hc flc adg adgv2; do
  python generate_gpt2_keyword_v3.py --mode G --config_path generate_configs/Yahoo-gpt-generate_keyword_wo_prompt_${alg}.ini
done
