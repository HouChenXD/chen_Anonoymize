
import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from tqdm import tqdm
import jsonlines
import os
import math
import torch.nn.functional as F
import numpy as np
import re
from peft import PeftModel
from transformers.generation.logits_process import LogitsProcessorList

def parse_args():
    
    parser = argparse.ArgumentParser(description="Infer with a transformer model.")
    parser.add_argument("--model_name_or_path", type=str, default="reference/deepseek-coder-6.7b-instruct",help="Name of the pre-trained model to use.")
    # mode: 1) complete: Code Completion (2)insert: Code Insertion (3)chat:Chat Model Inference
    parser.add_argument("--mode", type=str, default="chat", choices=["complete", "insert", "chat"], help="Mode of operation: complete, insert, or chat.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to the input file containing text to process.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output file to save results.")
    parser.add_argument("--results_file", type=str, default=None, help="Path to the file to save results.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the generated text.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter.")
    # max_new_tokens is used in chat model inference
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate in chat mode.")
    parser.add_argument("--use_chat_template", action="store_true", help="Whether to use chat template for the input text.")
    # do_sample, default is False, which means greedy decoding
    parser.add_argument("--do_sample", type=bool, default=False, help="Whether to use sampling during generation. If False, uses greedy decoding.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--use_beam_search", action="store_true", help="Whether to use beam search.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--use_entropy", action="store_true", help="Whether to use entropy in the generation process.")
    parser.add_argument("--entropy_mode", type=str, default="top_n")
    parser.add_argument("--top_n", type=int, default=10)
    
    parser.add_argument("--contra_model_name_or_path", type=str, help="Path to the model or model name.")
    parser.add_argument("--steer_step", type=int, default=3, help="Steps to steer the model.")
    parser.add_argument("--entropy_thresh", type=float, default=0.5, help="Threshold for entropy.")
    parser.add_argument("--is_zero_reweight", type=bool, default=False, help="Whether to use zero reweighting.")
    parser.add_argument("--is_entropy_filter", type=bool, default=True, help="Whether to use entropy filtering.")
    parser.add_argument("--lam", type=float, default=2, help="Lambda for reweighting.")
    parser.add_argument("--prefix_step", type=int, default=0,help="Generate these first k tokens without steering.")
    parser.add_argument("--dynamic", action="store_true",help="dynamic steering")


    return parser.parse_args()

@torch.inference_mode()
def predict_next_token(model, input_ids,attention_mask):
    with torch.no_grad():
        logits_last = model(input_ids,attention_mask=attention_mask).logits[:, -1, :]          # (B, V)
        logits_last = logits_last.to(torch.float32)
    # next_token  = torch.argmax(logits_last, dim=-1, keepdim=True)  # (B, 1)
    return logits_last

def token_filter(running_entropy, logits,ENTROPY_THRESH,TOP_N,cur_step):
    step_entropy,topk_logits,indices = normalized_topn_entropy(logits,TOP_N) # (batch_size,)
    running_entropy = (running_entropy * cur_step + step_entropy) / (cur_step + 1)
    need_reweight = running_entropy >= ENTROPY_THRESH
    print(f"running_entropy: {running_entropy} ENTROPY_THRESH: {ENTROPY_THRESH}")
    return need_reweight,running_entropy,topk_logits,indices


@torch.inference_mode()
def reweight_prob(
    origin_topk: torch.Tensor,      
    idx_topk:    torch.Tensor,          
    contra_model: torch.nn.Module,      
    tokenizer,
    full_input_ids:  torch.Tensor,      
    full_attn:       torch.Tensor,      
    lam: float = 2.0,                  
    debug: bool = True,
):

    B, N = origin_topk.shape
    device = origin_topk.device

    contra_full = predict_next_token(contra_model, full_input_ids, full_attn)                              # (B, V)

    contra_topk, contra_idx = torch.topk(contra_full, N, dim=-1)   # (B, N)

    origin_prob  = torch.softmax(origin_topk, dim=-1)              # (B, N)
    contra_prob  = torch.softmax(contra_topk, dim=-1)              # (B, N)

    new_logits = origin_topk.clone()
    new_prob = torch.softmax(new_logits, dim=-1)  # (B, N)

    for b in range(B):
        pos_origin = {int(tid): j for j, tid in enumerate(idx_topk[b])}
        pos_contra = {int(tid): j for j, tid in enumerate(contra_idx[b])}

        case_str = ""
        for tid in set(pos_origin).intersection(pos_contra):
            j_o = pos_origin[tid]
            j_c = pos_contra[tid]

            delta_p = contra_prob[b, j_c] - origin_prob[b, j_o]    # >0 ⇒ contra ↑
            new_prob[b, j_o] = origin_prob[b, j_o] - lam * delta_p

        if debug:
            print(f"\n[batch {b}] Top {N} tokens reweight summary:")
            print(f"{'token':^8} | {'orig_logit':^10} | {'contra_logit':^12} | {'new_logit':^10} | {'p_orig':^7} | {'p_contra':^8} | {'p_new':^7} | {'Δp':^6}")
            print("-" *  90)
            for j in range(N):
                tid = int(idx_topk[b, j])
                tok = tokenizer.decode([tid]).strip()

                o_logit = origin_topk[b, j].item()
                n_logit = new_logits[b, j].item()
                p_o     = origin_prob[b, j].item()
                p_n     = new_prob[b, j].item()

                if tid in pos_contra:
                    c_logit   = contra_topk[b, pos_contra[tid]].item()
                    p_c       = contra_prob[b, pos_contra[tid]].item()
                    delta_p   = p_c - p_o
                    pc_str    = f"{p_c:.4f}"
                    dp_str    = f"{delta_p:+.4f}"
                    cstr      = f"{c_logit:^12.3f}"
                else:
                    pc_str = "  N/A  "
                    dp_str = "  N/A "
                    cstr   = "    N/A     "

                print(f"{tok:^8} | {o_logit:^10.3f} | {cstr} | {n_logit:^10.3f} | {p_o:^7.4f} | {pc_str:^8} | {p_n:^7.4f} | {dp_str:^6}")
                case_str += f"{tok:^8} | {o_logit:^10.3f} | {cstr} | {n_logit:^10.3f} | {p_o:^7.4f} | {pc_str:^8} | {p_n:^7.4f} | {dp_str:^6}\n"

    return new_prob, idx_topk,case_str




def topk_to_dict(logits_row, idx_row, tokenizer, k=10):
    toks = tokenizer.convert_ids_to_tokens(idx_row.tolist())
    vals = logits_row.tolist()
    return {tok: round(val, 2) for tok, val in zip(toks[:k], vals[:k])}



def match_logits(logits, token_indices, regex_dict,tokenizer,input_types):
    
    for c in range(len(logits)):
        for i in range(len(logits[c])):
            typ = input_types[c]
            pos = logits[c].argmax(dim=-1, keepdim=True)
            next_token = token_indices[c][pos]
            token = tokenizer.decode(next_token)
            pattern = re.compile(regex_dict[typ][2])
            if pattern.match(token):
                break
            else:
                logits[c][i] = 0
    return logits, token_indices

@torch.inference_mode()
def generate_dynamic(
    model, contra_model, tokenizer,
    masked_input: str, api_key_type: str,
    sampling_params, steer_params, regex_dict
):
    device = model.device
    inputs = tokenizer(masked_input, return_tensors="pt").to(device)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    base_len = input_ids.size(1)

    running_entropy = torch.tensor(0.0, device=device)
    FIX_LENGTH = regex_dict[api_key_type[0]][1] - len(regex_dict[api_key_type[0]][3])

    reweight_cnt  = 0            
    total_step   = 0
    generated_tokens = []
    while True:
        if total_step >= FIX_LENGTH:
            break
        if reweight_cnt >= steer_params["STEEP_STEP"]:   
            break
        total_step += 1
        logits = predict_next_token(model, input_ids, attention_mask)
        need_rw, running_entropy, topk_logits, topk_idx = token_filter(
            running_entropy, logits,
            steer_params["ENTROPY_THRESH"],
            steer_params["TOP_N"],
            total_step
        )
        need_rw = need_rw.item()

        case_str = ""
        if need_rw:                     
            reweighting = True
            print(f"--- step {total_step+1} reweighting need reweight {need_rw}---")
            rw_logits, rw_idx,case_str = reweight_prob(
            topk_logits, topk_idx,
            contra_model, tokenizer,
            input_ids, attention_mask,
            lam=steer_params["LAM"]
        )
            if steer_params["IS_ZERO_REWEIGHT"]:
                rw_logits, rw_idx= match_logits(
                    rw_logits, rw_idx, regex_dict, tokenizer,api_key_type
                )
            pos = rw_logits.argmax(dim=-1, keepdim=True)
            nxt = torch.gather(rw_idx, 1, pos)
            reweight_cnt += 1
            input_ids      = torch.cat([input_ids, nxt], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(nxt)], dim=-1)
        else:                           
            print(f"\n--- step {total_step + 1} unweighted ---")
            nxt = logits.argmax(dim=-1, keepdim=True)
            input_ids      = torch.cat([input_ids, nxt], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(nxt)], dim=-1)                 
        generated_tokens.append(nxt.item())

    
    remaining = FIX_LENGTH - (input_ids.size(1) - base_len)
    remaining = max(0, remaining)
    cur_input = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if remaining <= 0: 
        full_text = tokenizer.decode(input_ids[0][base_len:], skip_special_tokens=True)
        full_text = full_text + regex_dict[api_key_type[0]][3]                             
        return full_text.replace("..","."),remaining,reweight_cnt,cur_input,case_str

    gen_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=remaining,
        do_sample=False,
        num_beams=1,
        eos_token_id=None, 
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        return_legacy_cache=True
    )
    full_text = tokenizer.decode(gen_out.sequences[0][base_len:], skip_special_tokens=True)
    full_text = full_text + regex_dict[api_key_type[0]][3]

    
    return full_text.replace("..","."),remaining,reweight_cnt,cur_input,case_str


def normalized_full_entropy(
    logits: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    probs    = F.softmax(logits, dim=dim)               # (B, V)
    log_probs= torch.log(probs)                        # (B, V)
    H        = -torch.sum(probs * log_probs, dim=dim)  # (B,)
    V        = logits.size(dim)
    return H / math.log(V)

def normalized_topn_entropy(
    logits: torch.Tensor,
    top_n: int,
    dim: int = -1
) -> torch.Tensor:
    """
      logits: (batch_size, vocab_size)
      top_n:   N 
      return: u, shape=(batch_size,), u ∈ [0,1]
    """
    topk, indices = torch.topk(logits, top_n, dim=dim)       # (B, N)
    probs    = F.softmax(topk, dim=dim)                # (B, N)
    log_probs= torch.log(probs)                        # (B, N)
    H        = -torch.sum(probs * log_probs, dim=dim)  # (B,)
    return H / math.log(top_n),topk, indices


def generate_from_model_chat(model, tokenizer, input_text,mode, sampling_params,use_chat_template): 
    """
    in use chat template, the input text can be:
    You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
    ### Instruction:
    ['content']
    ### Response:
    ['content']
    <|EOT|>
    ### Instruction:
    ['content']
    ### Response:
    
    
    """
    if use_chat_template:
        input_text = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n
        ### Instruction:\n
        {input_text}\n
        ### Response:\n
        """
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=sampling_params['max_new_tokens'], do_sample=False, top_k=50, top_p=sampling_params['top_p'], num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    else:
        messages=[
        { 'role': 'user', 'content': input_text}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=sampling_params['do_sample'], top_k=sampling_params['top_k'] , top_p=sampling_params['top_p'], num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return outputs
def main(args):
    torch.manual_seed(0)
    regex_dict = {
    'google_api_key': [r"AIza", 35, r"[0-9A-Za-z\-_]",""],
    'stripe_test_secret_key': [r"sk_test_", 24, r"[0-9a-zA-Z]",""],
    'tencent_cloud_secret_id': [r"AKID", 32, r"[0-9a-zA-Z]",""],
    'google_oauth_client_id': [r"", 72, r"[a-z0-9\-\.]",".apps.googleusercontent.com"],
    'alibaba_cloud_access_key_id': [r"LTAI", 20, r"[a-zA-Z0-9]",""],
    'slack_incoming_webhook_url': [r"https://hooks.slack.com/services/", 46, r"[A-Za-z0-9+\/]",""]
    }
    
    inputs = []
    if args.input_file.endswith('.jsonl'):
        with jsonlines.open(args.input_file, mode='r') as f:
            for line in f:
                inputs.append(line)
    elif args.input_file.endswith('.json'):
        with open(args.input_file, "r") as f:
            inputs = json.load(f)
    
    print("="*60)
    print(f"Loaded {len(inputs)} inputs from {args.input_file}")
    print(f"mask input:\n{inputs[0]['masked_input']}")
    print("="*60)

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if args.use_beam_search:
        print("**************use beam search**************")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True,torch_dtype=torch.bfloat16).cuda()
    model.eval()
    contra_base  = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,trust_remote_code=True).cuda()

    contra_model  = PeftModel.from_pretrained(contra_base, args.contra_model_name_or_path,torch_dtype=torch.bfloat16).cuda()
    contra_model.eval()
    
    # sampleing parameters
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "use_beam_search": args.use_beam_search,
        "use_entropy": args.use_entropy,
        "top_n": args.top_n
    }

    steer_params = {
        "STEEP_STEP": args.steer_step,
        "ENTROPY_THRESH": args.entropy_thresh,
        "IS_ZERO_REWEIGHT": args.is_zero_reweight,
        "IS_ENTROPY_FILTER": args.is_entropy_filter,
        "LAM": args.lam,
        "TOP_N": args.top_n,
        "PREFIX_STEP": args.prefix_step,

    }
    with jsonlines.open(args.output_file, mode='a') as f:   
        for idx in tqdm(range(0, len(inputs), args.batch_size)):
            batch_inputs = inputs[idx:idx+args.batch_size]
            input_texts = [dat["masked_input"] for dat in batch_inputs]
            input_types = [dat["api_key"] for dat in batch_inputs]
            if args.mode == "chat":
                outputs, logits = generate_from_model_chat(model, tokenizer, input_texts,args.mode, sampling_params,args.use_chat_template)
            else:
                outputs,remaining,reweight_cnt,cur_input,case_str = generate_dynamic(model,contra_model,tokenizer, input_texts[0],input_types, sampling_params,steer_params,regex_dict)
                print(f"outputs:{[outputs]}")
                
            
            for dat in batch_inputs:
                if "output" in dat:
                    dat["origin"] = dat["output"]
                elif "origin" in dat:
                    dat["origin"] = dat["origin"]
                dat["output"] = outputs
                dat['remaining'] = remaining
                dat['reweight_cnt'] = reweight_cnt
                dat['cur_input'] = cur_input
                dat['case_str'] = case_str

                f.write(dat)  

            print("-"*60)
            
    
        
            

if __name__ == "__main__":
    args = parse_args()
    main(args)