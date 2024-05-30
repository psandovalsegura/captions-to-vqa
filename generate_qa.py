import os
import pickle
from pathlib import Path

import fire
from typing import List, Optional
from llama import Dialog, Llama

from keys import manual_review_keys
from prompts import user_prompts, system_prompts

# Inference
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    prompt_key: str = 'v001',
    use_system_prompt: bool = True,
):
    # Initialize Llama-3
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Load datacomp recaptioned
    # contains 00028118.tar and output_28118.pkl
    datacomp_recaptioned_dir = Path('/fs/vulcan-projects/stereo-detection/datacomp_recaptioned_shard')

    # open the pickle file, which contains a list of dictionaries
    # each dictionary contains the original image json plus 'vlm_model' and 'vlm_caption' keys
    with open(datacomp_recaptioned_dir / 'output_28118.pkl', 'rb') as f:
        data = pickle.load(f)
    print('Number of samples in shard:', len(data))

    # Create list of dialogs and do inference. Save outputs to file named after key and prompt version
    dialogs_list: List[Dialog] = []
    for key in manual_review_keys:
        vlm_caption = get_vlm_caption_for_key(data, key)
        dialog = dialog_from_caption(prompt_key, vlm_caption, use_system_prompt=use_system_prompt)
        print(f'Dialog for key {key}: {dialog}')
        dialogs_list.append(dialog)
        
    results = generator.chat_completion(
            dialogs_list, 
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
    )

    # make directory (at output/{prompt_key}) where output files will be placed
    output_dir = Path('output') / prompt_key
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, result in zip(manual_review_keys, results):
        filename = output_dir / f'output_{key}_{prompt_key}.txt'
        with open(filename, 'w') as f:
            f.write(result['generation']['content'])

def get_vlm_caption_for_key(data, key):
    for d in data:
        if d['key'] == key:
            return d['vlm_caption']

def dialog_from_caption(prompt_key, vlm_caption, use_system_prompt=False):
    if use_system_prompt:
        return [{"role": "system", "content": system_prompts[prompt_key]}, {"role": "user", "content": user_prompts[prompt_key] + vlm_caption}]
    else:
        return [{"role": "user", "content": user_prompts[prompt_key] + vlm_caption}]

if __name__ == "__main__":
    fire.Fire(main)