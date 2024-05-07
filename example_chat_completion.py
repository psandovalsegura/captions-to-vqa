# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Prompt the user for a chat turn and get the response from the model
    dialog: Dialog = []
    while True:
        user_input = input("Enter a prompt here: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            dialog = []
            print("[Info] Dialog cleared.")
            continue
        current_turn_user = {"role": "user", "content": user_input}
        dialog.append(current_turn_user)
        result = generator.chat_completion(
            [dialog], 
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        print(f"{result['generation']['role'].capitalize()}: {result['generation']['content']}\n")
        current_turn_assistant = {"role": "assistant", "content": result['generation']['content']}
        dialog.append(current_turn_assistant)
        print(f"[Info] Current dialog len: {len(dialog)}. Turns: {[turn['role'] for turn in dialog]}")


if __name__ == "__main__":
    fire.Fire(main)
