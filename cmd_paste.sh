# cmd paste for running things on jetson orin agx environment (setup slightly different from Melissa's environment lol)

# container "/nano-llm"
mkdir benchmarks
mkdir prompts

# host "/home/edgeml/experiments/jetson-profiler"
sudo docker cp jetson-nano-profiler.py a581d665ef30:/nano-llm/benchmarks/jetson-nano-profiler.py
sudo docker cp ~/Downloads/ShareGPT_V3_unfiltered_cleaned_split.json a581d665ef30:/nano-llm/prompts/ShareGPT_V3_unfiltered_cleaned_split.json

