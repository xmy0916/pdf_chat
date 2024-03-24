import torch
import argparse
from parse_pdf import create_db, search_most_similarity_content

from fastchat.model import load_model, get_conversation_template, add_model_args

def parse_args():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--load_8bit", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.model_path = "lmsys/vicuna-7b-v1.5-16k"
    args.load_8bit = True
    return args


@torch.inference_mode()
def init_model():
    args = parse_args()
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=1,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=False,
        revision=args.revision,
        debug=args.debug
        # device_id=device_id
    )

    return model, tokenizer

@torch.inference_mode()
def get_answer(question, model, tokenizer):
    args = parse_args()
    # Build the prompt with a conversation template
    msg = question
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Run inference
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    return outputs

if __name__ == "__main__":
    db = create_db("data")
    prompt = search_most_similarity_content(db, "what is yolov1?")
    print(prompt)
    model, tokenizer = init_model()
    answer = get_answer(prompt, model, tokenizer)