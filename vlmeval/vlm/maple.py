import sys
from typing import List, Literal, Sequence, TypedDict

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from transformers import PreTrainedTokenizerFast

from ..smp import *
from .base import BaseModel

# path of MLLM_Train
sys.path.append("/yezilyu/code/MLLM_Train")
from src.models.encoder.resampler import Resampler
from src.models.mllm.maple import ContinuousLVLM

# Constants
BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"

device = "cuda"
dtype = torch.bfloat16
num_img_in_tokens = 64
num_img_out_tokens = 64
height, width = 448, 448
original_size = [1024, 1024]
num_inference_steps = 50
guidance_scale = 3.0
do_classifier_free_guidance = guidance_scale > 1.0
max_length = 100

tokenizer_cfg_path = "configs/tokenizer/llama3_1_sft.yaml"
image_transform_cfg_path = "configs/processor/transform_maple.yaml"
visual_encoder_cfg_path = "configs/visual_encoder/eva_vit_448.yaml"
llm_cfg_path = "configs/models/llm_lora_2ffn.yaml"
# sd_cfg_path = "configs/visual_decoder/sdxl.yaml"

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class ChatFormat:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"))
        tokens.extend(self.tokenizer.encode(message["role"], add_special_tokens=False))
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"))
        tokens.extend(self.tokenizer.encode("\n\n", add_special_tokens=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), add_special_tokens=False))
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>"))
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


class ContinuousLVLMEval(BaseModel):
    def __init__(
        self,
        input_resampler=nn.Linear(1792, 4096),
        output_resampler=nn.Linear(4096, 1792),
        vit_down=False,
        mse=True,
        lm_loss_scale=1.0,
        rec_loss_scale=6.0,
        pretrained_model_path=None,
    ):
        BaseModel.__init__(self)
        self.dtype = torch.bfloat16
        self.default_image_tokens = (
            BOI_TOKEN + "".join(IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)) + EOI_TOKEN
        )

        # Load configurations
        tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
        self.tokenizer = hydra.utils.instantiate(tokenizer_cfg)
        self.formatter = ChatFormat(self.tokenizer)

        image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
        self.image_transform = hydra.utils.instantiate(image_transform_cfg)

        visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
        self.visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
        self.visual_encoder.cuda().eval().to(dtype=dtype)
        print("Init visual encoder done")

        llm_cfg = OmegaConf.load(llm_cfg_path)
        self.llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
        print("Init llm done.")

        # sd_cfg = OmegaConf.load(sd_cfg_path)
        # decoder = hydra.utils.instantiate(sd_cfg).to(device, dtype=dtype).eval()
        # print("Init visual decoder done")
        self.agent_model = ContinuousLVLM.from_pretrained(
            llm=self.llm,
            input_resampler=input_resampler,
            output_resampler=output_resampler,
            vit_down=vit_down,
            mse=mse,
            lm_loss_scale=lm_loss_scale,
            rec_loss_scale=rec_loss_scale,
            pretrained_model_path=pretrained_model_path,
        )
        self.agent_model.llm.merge_and_unload()
        self.agent_model.llm.compile()
        self.agent_model.cuda().eval().to(dtype=self.dtype)

    def generate_inner(self, message, dataset=None):
        content, images, cmp_mask, gen_mask, input_ids = "", [], torch.tensor([], dtype=torch.bool), torch.tensor([], dtype=torch.bool), []
        content += "<|start of header|>user<|end of header|>\n\n"
        input_ids.extend(self.formatter.encode_header({"role": "user", "content": ""}))

        for msg in message:
            if msg["type"] == "text":
                input_ids.append(self.tokenizer.bos_token_id)
                input_ids.extend(self.tokenizer.encode(msg["value"], add_special_tokens=False))
                input_ids.append(self.tokenizer.eos_token_id)
                content += "<|begin_of_text|>" + msg["value"] + "<|eos_id|>"
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                content += self.default_image_tokens
                input_ids += self.tokenizer.encode(self.default_image_tokens, add_special_tokens=False)
                cmp_mask = torch.cat([cmp_mask, torch.tensor([True])])
                gen_mask = torch.cat([gen_mask, torch.tensor([False])])

        content += "<|start of header|>assistant<|end of header|>\n\n"
        input_ids.extend(self.formatter.encode_header({"role": "assistant", "content": ""}))

        if len(cmp_mask) == 0:
            cmp_mask = None
        if len(gen_mask) == 0:
            gen_mask = None

        gen_mask.to(device=device, dtype=torch.bool)
        cmp_mask.to(device=device, dtype=torch.bool)

        img_tensor = [self.image_transform(image).cuda().to(dtype=self.dtype) for image in images]
        img_tensor = [self.visual_encoder.encode_image(tensor.unsqueeze(0)) for tensor in img_tensor]
        img_tensors = torch.stack(img_tensor).squeeze().cuda() if len(img_tensor) > 0 else None
        boi_token_id = self.tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = self.tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
        boi_idx = input_ids.index(boi_token_id)
        eoi_idx = input_ids.index(eoi_token_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        ids_cmp_mask = [False] * len(input_ids)
        ids_gen_mask = [False] * len(input_ids)
        ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
        ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
        ids_cmp_mask[boi_idx + 1 : eoi_idx] = True
        output = self.agent_model.generate(
            tokenizer=self.tokenizer,
            input_ids=input_ids.unsqueeze(0),
            image_embeds=img_tensors.unsqueeze(0),
            num_img_gen_tokens=num_img_out_tokens,
            ids_cmp_mask=ids_cmp_mask.unsqueeze(0),
            embeds_cmp_mask=cmp_mask,
            max_new_tokens=300,
        )
        text_output = output["text"]
        return text_output
