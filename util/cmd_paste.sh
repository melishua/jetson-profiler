# cmd paste for running things on jetson orin agx environment (setup slightly different from Melissa's environment lol)

# container "/nano-llm"
mkdir benchmarks
mkdir prompts

# host "/home/edgeml/experiments/jetson-profiler"
sudo docker cp jetson-nano-profiler.py a581d665ef30:/nano-llm/benchmarks/jetson-nano-profiler.py
sudo docker cp ~/Downloads/ShareGPT_V3_unfiltered_cleaned_split.json a581d665ef30:/nano-llm/prompts/ShareGPT_V3_unfiltered_cleaned_split.json


# EXPERIMENTS
# host "/home/edgeml/experiments/jetson-profiler"
python3 tegrastats-monitor.py
# container "/nano-llm"
python3 benchmarks/jetson-nano-profiler.py --num_iterations 1 --num_request_sample 100 --prompt_set prompts/ShareGPT_V3_unfiltered_cleaned_split.json

# host "/home/edgeml/experiments/jetson-profiler"
tegrastats --stop
