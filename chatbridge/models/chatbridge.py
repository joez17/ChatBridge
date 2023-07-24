import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from chatbridge.common.registry import registry
from chatbridge.models.blip2 import Blip2Base, disabled_train
from chatbridge.models.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaTokenizer
import json
from torch.nn.utils.rnn import pad_sequence
import transformers
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


@registry.register_model("chatbridge")
class ChatBridge(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/chatbridge.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_beat=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_template="",
        max_txt_len=150,
        end_sym='\n',
        stage=1,
        multi_sample=False,
        num_frames=4,
        num_audios=2,
        apply_lemmatizer=False,
        inf_type=None,
        drop=False,

    ):
        super().__init__()
        self.stage = stage
        self.inf_type = inf_type
        self.tokenizer = self.init_tokenizer()
        self._apply_lemmatizer = apply_lemmatizer
        if self._apply_lemmatizer:
            import spacy
            self._lemmatizer = spacy.load("en_core_web_sm")
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')
        
        print('Loading BEAT')
        self.audio_encoder, self.ln_audio = self.init_audio_encoder()
        self.audio_trans = nn.Linear(768, 1408)
        if freeze_beat:
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False                
            self.audio_encoder = self.audio_encoder.eval()
            self.audio_encoder.train = disabled_train         
            for name, param in self.ln_audio.named_parameters():
                param.requires_grad = False
            self.ln_audio = self.ln_audio.eval()   
            self.ln_audio.train = disabled_train
            logging.info("freeze audio encoder")
            
        self.drop = drop or self.stage==4
        print('Loading Q-Former dropout:', self.drop)
        self.Qformer, self.image_query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, drop=self.drop
        )
        self.audio_query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.image_query_tokens.shape[-1])
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.image_query_tokens.requires_grad = False
            self.audio_query_tokens.requires_grad = False
            for name, param in self.audio_trans.named_parameters():
                param.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.prompt = prompt_template
           
        self.multi_sample = multi_sample
        if self.multi_sample:
            # temporal position embedding
            self.visual_temporal_pos_embeds = nn.parameter.Parameter(
                    torch.zeros(1, num_frames, 1, self.visual_encoder.num_features)
                )
            self.audio_temporal_pos_embeds = nn.parameter.Parameter(
                    torch.zeros(1, num_audios, 1, self.visual_encoder.num_features)
                )

    def encode_img(self, image):
        if self.stage==3 and len(image.shape)==5:
            img_embeds_list, att_img_list = [], []
            for i in range(image.shape[1]):
                img_embeds_i, atts_img_i = self.encode_img_(image[:, i])
                img_embeds_list.append(img_embeds_i)
                att_img_list.append(atts_img_i)
            img_embeds = torch.cat(img_embeds_list, dim=1)
            atts_img = torch.cat(att_img_list, dim=1)
        else:
            img_embeds, atts_img = self.encode_img_(image)
        return img_embeds, atts_img
    def encode_img_(self, image):
        device = image.device
        with self.maybe_autocast():
            if len(image.shape)==4:    # b,n,3,h,w
                image = image.unsqueeze(1)
            b,n,c,h,w = image.shape
            image_embeds = []
            for i in range(n):
                image_embed = self.ln_vision(self.visual_encoder(image[:, i].contiguous())) # b, 257, 768
                if self.multi_sample:
                    image_embed += self.visual_temporal_pos_embeds[:, i].contiguous()
                image_embeds.append(image_embed)
            image_embeds = torch.cat(image_embeds, dim=1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.image_query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )    
            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
    def encode_audio(self, audio):
        device = audio.device
        with self.maybe_autocast():      
            b,n,h,w = audio.shape
            audio_embeds = []
            for i in range(n):
                audio_embed = self.audio_trans(self.ln_audio(self.audio_encoder(audio[:, i].contiguous()))) # b, 257, 768
                if self.multi_sample:
                    audio_embed += self.audio_temporal_pos_embeds[:, i].contiguous()
                audio_embeds.append(audio_embed)
            audio_embeds = torch.cat(audio_embeds, dim=1)
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.audio_query_tokens.expand(audio_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )    
            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(audio.device)
        return inputs_llama, atts_llama
    def forward(self, samples):
        task = samples['task'].split('%')
        task = random.choice(task)   
        if ('tia' in task or 'tva' in task) and 'image' in samples.keys() and 'audio' in samples.keys():
            image = samples["image"]
            device = image.device
            img_embeds, atts_img = self.encode_img(image)
            audio = samples["audio"]
            aud_embeds, atts_aud = self.encode_audio(audio)
            device = audio.device
            feat_list = [(img_embeds, atts_img), (aud_embeds, atts_aud)]
        elif 'ta' in task and 'audio' in samples.keys():
            audio = samples["audio"]
            img_embeds, atts_img = self.encode_audio(audio)
            device = audio.device
            feat_list = [(img_embeds, atts_img)]
        elif ('ti' in task or 'tv' in task) and 'image' in samples.keys():
            image = samples["image"]
            device = image.device
            img_embeds, atts_img = self.encode_img(image)
            feat_list = [(img_embeds, atts_img)]

        if self.stage==2 or self.stage==3:
            img_embeds, atts_img, targets = self.process_prompt_instruct(samples['conversation'], feat_list)

            to_regress_tokens = self.llama_tokenizer(['aa'],return_tensors="pt",padding="longest",truncation=True,max_length=self.max_txt_len,add_special_tokens=False)
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],dtype=to_regress_tokens.input_ids.dtype,device=self.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]
            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img], dim=1)
            empty_targets = (
                torch.ones([atts_img.shape[0], 1],
                        dtype=torch.long).to(device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            self.llama_tokenizer.padding_side = "right"
        else:
            self.llama_tokenizer.padding_side = "right"
            if not 'text_input' in samples.keys():
                if task=='ti':
                    samples["text_input"] = samples['video_captions']
                elif task=='ta':
                    samples["text_input"] = samples['audio_captions']
                elif task=='tia':
                    samples["text_input"] = samples['multimodal_captions']
            text = [t + self.end_sym for t in samples["text_input"]]
            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(device)
            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]
            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_beat = cfg.get("freeze_beat", True)
        freeze_qformer = cfg.get("freeze_qformer", True)

        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        stage = cfg.get("stage", 1)
        multi_sample = cfg.get("multi_sample", False)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        inf_type = cfg.get("inf_type", None)
        drop = cfg.get("drop", False)
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_beat=freeze_beat,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            stage=stage,
            multi_sample=multi_sample,
            apply_lemmatizer=apply_lemmatizer,
            inf_type=inf_type,
            drop=drop,
        )

        ckpt_path = cfg.get("ckpt", "") 
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print(msg)

        return model

    def process_prompt_instruct(self, conversations, feat_list):
        inputs_embeds = []
        attention_masks = []
        targets = []
        for idx, conversation in enumerate(conversations): #0,1,...,15
            inputs_embeds_list = []
            attention_mask_list = []
            tgt_list = []
            prompts = conversation.split('<query>')
            assert len(prompts)==len(feat_list)+1
            for i in range(len(feat_list)): # 0,1
                inputs_llama, atts_llama = feat_list[i]
                inputs_llama, atts_llama = inputs_llama[idx:idx+1], atts_llama[idx:idx+1]
                device = inputs_llama.device
                prompti_embeds, prompti_atts, prompti_ids = self.encode_pt(prompts[i], device)
                inputs_embeds_list.append(prompti_embeds)
                inputs_embeds_list.append(inputs_llama)
                attention_mask_list.append(prompti_atts)
                attention_mask_list.append(atts_llama.to(inputs_llama.device))
                tgt_list.append(prompti_ids.fill_(-100))
                empty_targets = torch.ones([inputs_llama.shape[0], inputs_llama.shape[1]],dtype=torch.long).to(device).fill_(-100)  
                tgt_list.append(empty_targets)
            prompt = prompts[-1]
            start_token = '###Assistant:'
            end_token = '###'
            while start_token in prompt:
                start_idx = prompt.find(start_token)+len(start_token)
                pt1 = prompt[:start_idx]
                prompt = prompt[start_idx:]
                if prompt.find(end_token)<0:
                    pt1_embeds, pt1_atts, pt_ids = self.encode_pt(pt1, device)
                    inputs_embeds_list.append(pt1_embeds)
                    attention_mask_list.append(pt1_atts)
                    tgt_list.append(pt_ids.fill_(-100))
                    break
                end_idx = prompt.find(end_token)+len(end_token)
                tgt1 = prompt[:end_idx]
                prompt = prompt[end_idx:]
                pt1_embeds, pt1_atts, pt_ids = self.encode_pt(pt1, device)
                tgt1_embeds, tgt1_atts, tgt1_ids = self.encode_pt(tgt1, device)
                inputs_embeds_list.append(pt1_embeds)
                inputs_embeds_list.append(tgt1_embeds)
                attention_mask_list.append(pt1_atts)
                attention_mask_list.append(tgt1_atts)
                tgt_list.append(pt_ids.fill_(-100))
                tgt_list.append(tgt1_ids)
            if len(prompt)>0:
                pt1_embeds, pt1_atts, pt1_ids = self.encode_pt(prompt, device)
                inputs_embeds_list.append(pt1_embeds)
                attention_mask_list.append(pt1_atts)
                tgt_list.append(pt1_ids.fill_(-100))

            inputs_embed = torch.cat(inputs_embeds_list, dim=1)
            attention_mask = torch.cat(attention_mask_list, dim=1)
            target = torch.cat(tgt_list, dim=1)

            inputs_embed = inputs_embed[:, :self.max_txt_len]
            attention_mask = attention_mask[:, :self.max_txt_len]
            target = target[:, :self.max_txt_len]

            inputs_embeds.append(inputs_embed.squeeze(0))
            attention_masks.append(attention_mask.squeeze(0))
            targets.append(target.squeeze(0))

        flip_inputs_embeds = [ie.flip(0) for ie in inputs_embeds]
        flip_inputs_embeds = pad_sequence(flip_inputs_embeds, batch_first=True, padding_value=0.0)
        inputs_embeds = flip_inputs_embeds.flip(1)

        flip_attention_masks = [at.flip(0) for at in attention_masks]
        flip_attention_masks = pad_sequence(flip_attention_masks, batch_first=True, padding_value=0)
        attention_masks = flip_attention_masks.flip(1)

        flip_targets = [t.flip(0) for t in targets]
        flip_targets = pad_sequence(flip_targets, batch_first=True, padding_value=-100)
        targets = flip_targets.flip(1)

        return inputs_embeds, attention_masks, targets
    def encode_pt(self, prompts2, device):
        prompts2 = self.llama_tokenizer(
            prompts2,
            return_tensors="pt",
            padding="longest",
            truncation=False,    
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device) 
        input_ids = prompts2.input_ids
        prompt2_embeds = self.llama_model.model.embed_tokens(input_ids)  
        return prompt2_embeds, prompts2.attention_mask, input_ids
    def get_input_embeds(self, samples, prompt):
        if self.drop:
            self.Qformer.bert.train()
        if self.inf_type=='multi_image':
            image = samples['video']
            feat_list = []
            for i in range(5):
                inputs_llama, atts_llama = self.encode_img(image[:, i])
                feat_list.append((inputs_llama, atts_llama))
        else:
            if 'video' in samples.keys() and 'audio' in samples.keys():
                image = samples['video']
                image = image.permute(0,2,1,3,4) # B,C,T,H,W -- B,T,C,H,W
                audio = samples['audio']
                
                if self.inf_type=='imageaudio':
                    inputs_llama, atts_llama = self.encode_imageaudio(image, audio)
                    feat_list = [(inputs_llama, atts_llama)]
                elif self.inf_type=='image and audio':
                    inputs_llama, atts_llama = self.encode_img(image)
                    inputs_llama1, atts_llama1 = self.encode_audio(audio)
                    feat_list = [(inputs_llama, atts_llama), (inputs_llama1, atts_llama1)]
                elif self.inf_type=='image':
                    inputs_llama, atts_llama = self.encode_img(image)
                    feat_list = [(inputs_llama, atts_llama)]
                elif self.inf_type=='audio':
                    inputs_llama, atts_llama = self.encode_audio(audio)
                    feat_list = [(inputs_llama, atts_llama)]
            else:
                if 'video' in samples.keys():
                    image = samples['video']
                    image = image.permute(0,2,1,3,4) # B,C,T,H,W -- B,T,C,H,W
                    inputs_llama, atts_llama = self.encode_img(image)
                elif 'image' in samples.keys():
                    image = samples["image"]
                    inputs_llama, atts_llama = self.encode_img(image)
                elif 'audio' in samples.keys():
                    audio = samples['audio']
                    inputs_llama, atts_llama = self.encode_audio(audio)
                feat_list = [(inputs_llama, atts_llama)]

        if len(prompt)>0:
            if '<question>' in prompt and '<caption>' in prompt:
                questions = samples['text_input']
                captions = samples['caption']
                conversations = [prompt.replace('<question>', question).replace('<caption>', caption) 
                                    for question, caption in zip(questions, captions) ]     
            elif '<question>' in prompt:
                questions = samples['text_input']
                conversations = [prompt.replace('<question>', question) for question in questions ]   
            else:
                conversations = [prompt] * inputs_llama.shape[0]   
            img_embeds, atts_img, _ = self.process_prompt_instruct(conversations, feat_list)
        else:
            img_embeds, atts_img = inputs_llama, atts_llama
        return img_embeds, atts_img
