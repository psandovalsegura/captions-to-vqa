# captions-to-vqa

#### Setup instructions

1. Create a new conda environment `conda create --name llama3 python=3.8`.
2. Follow [Quick Start instructions](https://github.com/meta-llama/llama3?tab=readme-ov-file#quick-start) from the llama-3 repo. 

#### QA generation instructions

We will be running `generate-qa.sh`, which uses `torchrun`.
1. Replace `--ckpt_dir` and `--tokenizer_path` with your llama-3 checkpoint and tokenizer paths.
2. If you are not using shard `output_28118.pkl`, then replace this filename in `generate_qa.py`.
3. Select image keys in `keys.py`. The image keys must within the shard you previously selected. This is how we select images to generate QA pairs from.
4. Run `bash generate-qa.sh`. This will generate an `output` folder with subdirectories named after the prompt version. Output files are named `output_{key}_{prompt_version}`.