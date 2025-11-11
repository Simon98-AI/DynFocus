import argparse
import torch
import sys
sys.path.append("./")
sys.path.append("./../")
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--Eval_QA_root', help='Path to the ground truth file containing question.', required=None)
    parser.add_argument('--dataset_name', help='dataset name', required=None)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)

    return parser.parse_args()


def load_video(video_path):
    K = 32
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx) > 0:
        scale = 1.0 * len(frame_idx) / K
        uniform_idx = [round((i + 1) * scale - 1) for i in range(K)]
        frame_idx = [frame_idx[i] for i in uniform_idx]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def run_inference(args, dataset_qajson):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """

    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)

    qa_json = dataset_qajson[args.dataset_name]
    print(f'Dataset name:{args.dataset_name}, {qa_json=}!')
    with open(qa_json, 'r', encoding='utf-8') as f:
        gt_questions = json.load(f)
    #gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    eval_dict = {}

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name

    for (q_id, item) in tqdm(gt_questions.items()):
        video_id = item['video_id']
        question = item['question'] 
        if len(item['choices']) == 6:
            question += f"Choices: A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} F.{item['choices']['F']} \n Among the six options A, B, C, D, E, F above, the one closest to the correct answer is:"
            candidates = ['A', 'B', 'C', 'D', 'E', 'F']
            candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}", f"E.{item['choices']['E']}", f"F.{item['choices']['F']}"]
        elif len(item['choices']) == 5:
            question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} \n Among the five options A, B, C, D, E above, the one closest to the correct answer is: "
            candidates = ['A', 'B', 'C', 'D', 'E']
            candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}", f"E.{item['choices']['E']}"]
        elif len(item['choices']) == 4:
            question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} \n Among the four options A, B, C, D above, the one closest to the correct answer is:"
            candidates = ['A', 'B', 'C', 'D']
            candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}"]
        elif len(item['choices']) == 3:
            question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} \n Among the three options A, B, C above, the one closest to the correct answer is: "
            candidates = ['A', 'B', 'C']
            candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}"]
        elif len(item['choices']) == 2:
            question += f" A.{item['choices']['A']} B.{item['choices']['B']} \n Among the two options A, B above, the one closest to the correct answer is: "
            candidates = ['A', 'B']
            candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}"]
        # Load the video file
        vid_rela_path = item['vid_path']
        video_path = os.path.join(args.video_dir, vid_rela_path.split('/')[-2] + "/" + vid_rela_path.split('/')[-1])

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]
            
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        cur_prompt = question
        with torch.inference_mode():
            model.update_prompt([[cur_prompt]])
            output_ids = model.generate(
                input_ids,
                images=video,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        eval_dict[q_id] = {
            'video_id': video_id,
            'question': question,
            'output_sequence': outputs
        }  
        print(f'q_id:{q_id}, output:{outputs}!\n')
    
    eval_dataset_json = f'{args.output_dir}/{args.dataset_name}_eval.json'
    with open(eval_dataset_json, 'w', encoding='utf-8') as f:
        json.dump(eval_dict, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    dataset_qajson = {
    "Ucfcrime": f"{args.Eval_QA_root}/Eval_QA/Ucfcrime_QA_new.json",
    "Youcook2": f"{args.Eval_QA_root}/Eval_QA/Youcook2_QA_new.json",
    "TVQA": f"{args.Eval_QA_root}/Eval_QA/TVQA_QA_new.json",
    "MSVD": f"{args.Eval_QA_root}/Eval_QA/MSVD_QA_new.json",
    "MSRVTT": f"{args.Eval_QA_root}/Eval_QA/MSRVTT_QA_new.json",
    "Driving-decision-making": f"{args.Eval_QA_root}/Eval_QA/Driving-decision-making_QA_new.json",
    "NBA": f"{args.Eval_QA_root}/Eval_QA/NBA_QA_new.json",
    "SQA3D": f"{args.Eval_QA_root}/Eval_QA/SQA3D_QA_new.json",
    "Driving-exam": f"{args.Eval_QA_root}/Eval_QA/Driving-exam_QA_new.json",
    "MV": f"{args.Eval_QA_root}/Eval_QA/MV_QA_new.json",
    "MOT": f"{args.Eval_QA_root}/Eval_QA/MOT_QA_new.json",
    "ActivityNet": f"{args.Eval_QA_root}/Eval_QA/ActivityNet_QA_new.json",
    "TGIF": f"{args.Eval_QA_root}/Eval_QA/TGIF_QA_new.json"
    }
    run_inference(args, dataset_qajson)