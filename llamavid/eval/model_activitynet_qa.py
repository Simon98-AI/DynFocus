import argparse
import torch
import sys
sys.path.append("./")
sys.path.append("./../")
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import json
import os
import numpy as np
import math
from tqdm import tqdm
from PIL import Image
import PIL.Image as Image
import torchvision.transforms
import numpy as np
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
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
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
    K = 64
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # if len(frame_idx) > 0:
    #     scale = 1.0 * len(frame_idx) / K
    #     uniform_idx = [round((i + 1) * scale - 1) for i in range(K)]
    #     frame_idx = [frame_idx[i] for i in uniform_idx]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def save_video(spare_frames):
    N = spare_frames.shape[0]
    for idx in range(N):
        temp = spare_frames[idx]
        temp = np.transpose(temp, (2, 0, 1))
        size = (224, 224)
        transform = torchvision.transforms.CenterCrop(size)
        center_crop = transform(torch.Tensor(temp))
        center_crop = center_crop.numpy().astype('uint8')
        center_crop = np.transpose(center_crop, (1, 2, 0))
        new = Image.fromarray(center_crop)
        output_path = "/mnt_new/hanyudong.hyd/Slow-Fast-Vid/LVBench/data/ext_frame/{}.jpg".format(idx)
        new.save(output_path)

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    # Load both ground truth file containing questions and answers
    with open(args.gt_file_question) as file:
        gt_questions = json.load(file) # 这里面的视频其实一共只有8000
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)

    idx = 16 # 4, 
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)[idx]
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)[idx]
    gt_questions = [gt_questions]
    gt_answers = [gt_answers]
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"  # 其实并不影响最终的结果
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    # /mnt_new/hanyudong.hyd/llama-vid/work_dirs/eval_activitynet/pred.json
    ans_file = open(answers_file, "w")

    index = 0
    cnt = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        video_path = None
        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if video_path == None:
            continue

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            save_video(video)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
            # temp = video.numpy()
            video = video.half().cuda()
            # print(video.shape, "------")      
            video = [video]
            

        # try:
            # Run inference on the video and add the output to the list
            
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
                temperature=0.1,
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

        sample_set['pred'] = outputs
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    # print(cnt)

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
