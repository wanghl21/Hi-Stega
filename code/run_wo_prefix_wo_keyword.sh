

#for alg in ac hc flc adg adgv2; do
#  python generate_gpt2_v2.py --mode G --config_path generate_configs/Yahoo-gpt-generate_${alg}.ini
#  python generate_gpt2_v2.py --mode G --config_path generate_configs/Yahoo-gpt-generate_wo_prompt_${alg}.ini
#  python generate_gpt2_v2.py --mode G --config_path generate_configs/Yahoo-gpt-generate_prompt_${alg}.ini
#done

for alg in adgv2; do
  python generate_gpt2_v2.py --mode G --config_path generate_configs/Yahoo-gpt-generate_${alg}.ini
done
