
"""
v2:add whisper encoder for semantic information extraction
v3:replace llama2-7b with llama3.1-8b -- only llama2 supports Chinese!!
v4:add MERT encoder for music

adding universal projection module
    replace one mlp to moe-mlp

adding lps loss
adding task-aware embedding
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.ced.audiotransformer import AudioTransformer, CEDConfig
from models.lps import LanguageContextPatchSelection, SemanticSpatialPatchCalibration, SparsePatchWordAlignment, LAPS_Loss
from peft import LoraConfig, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
from transformers import Wav2Vec2FeatureExtractor, AutoModel

class USAM(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        # encoder
        ced_config = CEDConfig()

        # the checkpoint can be downloaded from zenodo:
        # https://zenodo.org/record/8275347/files/audiotransformer_base_mAP_4999.pt?download=1
        ced_checkpoint = torch.load(
            "pretrained_models/ced/audiotransformer_base_mAP_4999.pt"
        )
        self.encoder = AudioTransformer(
            ced_config,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            outputdim=527,
            target_length=1012,
        )
        self.encoder.load_state_dict(ced_checkpoint, strict=False)
        encoder_peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.apply_encoder_strategy(encoder_peft_config)

        # 现阶段冻结
        local_model_path = 'pretrained_models/whisper-small'
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(local_model_path)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(local_model_path)
        for p in self.whisper.model.parameters():
            p.requires_grad = False


        # 现阶段冻结
        local_model_path = 'pretrained_models/MERT-v1-95M'
        self.mert = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad = False 
        self.conv = nn.Conv2d(13, 1, 1) # 13 layers -> 1


        # mlp
        self.speech_former, self.speech_query_tokens = self.build_audio_qformer(
            1, self.encoder.embed_dim*3, 2, 1
        )

        # decoder
        '''
        hf_token = "your huggingface token"
        self.tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b", token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.decoder = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b", token=hf_token
        )
        '''
        # 本地模型路径
        local_model_path = 'pretrained_models/llama3.1-8b'
        # 加载本地的tokenizer和模型
        # self.tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
        # self.tokenizer.pad_token = self.tokenizer.unk_token
        
        self.decoder = LlamaForCausalLM.from_pretrained(local_model_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(local_model_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.decoder.resize_token_embeddings(len(self.tokenizer))



        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.apply_decoder_strategy(peft_config)

        # mlp, must call after init self.decoder
        # self.enc_to_dec_proj = self.build_audio_projector(
        #     projector_type="linear", in_dim=self.speech_former.config.hidden_size
        # )

        num_experts = 3

        moe = [self.build_audio_projector(
            projector_type="mlp1x_gelu", in_dim=self.speech_former.config.hidden_size
        )]*num_experts
        self.enc_to_dec_proj = nn.Sequential(*moe)
        self.fc = nn.Linear(self.decoder.config.hidden_size, self.speech_former.config.hidden_size)
        self.router = nn.Linear(
            self.speech_former.config.hidden_size*2, num_experts, bias=False
        )


        # lps module
        self.lps_module = LanguageContextPatchSelection(
            self.decoder.config.hidden_size, 1024
        )
        self.spc_module = SemanticSpatialPatchCalibration(
            self.decoder.config.hidden_size, 1024
        )
        # spa module
        self.spa_module = SparsePatchWordAlignment()
        # lps loss
        self.lps_loss_fn = LAPS_Loss(margin=0.2, desired_ratio=0.5)



    def print_module_parameters(self):
        encoder_num_params = sum([i.numel() for i in self.encoder.parameters()])
        decoder_num_params = sum([i.numel() for i in self.decoder.parameters()])
        speech_former_num_params = sum(
            [i.numel() for i in self.speech_former.parameters()]
        )
        mlp_num_params = sum([i.numel() for proj in self.enc_to_dec_proj for i in proj.parameters()]) + sum(p.numel() for p in self.router.parameters())
        print(
            f"model params encoder: {encoder_num_params}, decoder: {decoder_num_params}, speech_former: {speech_former_num_params}, mlp: {mlp_num_params}"
        )

    def prepare_inputs_labels_for_multimodal(
        self, audio_embeds, atts, prompt, text=None
    ):
        prompt_left = []
        prompt_right = []
        for i, p in enumerate(prompt):
            l, r = p.split("<AcousticTokens>")
            prompt_left.append(self.tokenizer.bos_token + l)
            prompt_right.append(r)

        prompt_left_tokens = self.tokenizer(
            prompt_left, 
            add_special_tokens=False, 
            padding="longest",
            return_tensors="pt"
        ).to(audio_embeds.device)
        prompt_left_embeds = self.decoder.model.model.embed_tokens(
            prompt_left_tokens.input_ids
        )

        prompt_right_tokens = self.tokenizer(
            prompt_right,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        ).to(audio_embeds.device)
        prompt_right_embeds = self.decoder.model.model.embed_tokens(
            prompt_right_tokens.input_ids
        )

        input_embeds = torch.cat(
            [prompt_left_embeds, audio_embeds, prompt_right_embeds], dim=1
        )
        input_mask = torch.cat(
            [
                prompt_left_tokens.attention_mask,
                atts,
                prompt_right_tokens.attention_mask,
            ],
            dim=1,
        )

        decoder_targets = None
        if text is not None:
            new_text = []
            for t in text:
                new_text.append(t + self.tokenizer.eos_token)  # </s> is the eos_token
            text_tokens = self.tokenizer(
                new_text,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            ).to(audio_embeds.device)
            text_embeds = self.decoder.model.model.embed_tokens(text_tokens.input_ids)

            targets = text_tokens.input_ids.masked_fill(
                text_tokens.input_ids == self.tokenizer.pad_token_id, -100
            )
            empty_targets = (
                torch.ones([input_mask.shape[0], input_mask.shape[1]], dtype=torch.long)
                .to(audio_embeds.device)
                .fill_(-100)
            )
            decoder_targets = torch.cat([empty_targets, targets], dim=1)

            input_embeds = torch.cat([input_embeds, text_embeds], dim=1)
            input_mask = torch.cat([input_mask, text_tokens.attention_mask], dim=1)
        

        return input_embeds, input_mask, decoder_targets, audio_embeds, text_embeds


    def get_prompt_embeds(self, prompt, audios):
        prompt_left = []
        prompt_right = []
        for i, p in enumerate(prompt):
            l, r = p.split("<AcousticTokens>")
            prompt_left.append(self.tokenizer.bos_token + l)
            prompt_right.append(r)

        prompt_left_tokens = self.tokenizer(
            prompt_left, 
            add_special_tokens=False, 
            padding="longest",
            return_tensors="pt"
        ).to(audios.device)
        prompt_left_embeds = self.decoder.model.model.embed_tokens(
            prompt_left_tokens.input_ids
        )

        prompt_right_tokens = self.tokenizer(
            prompt_right,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        ).to(audios.device)
        prompt_right_embeds = self.decoder.model.model.embed_tokens(
            prompt_right_tokens.input_ids
        )

        prompt_embeds = torch.cat([prompt_left_embeds, prompt_right_embeds], dim=1)
        return prompt_embeds        


    def forward_encoder(self, audios, prompt=None):
        audio_embeds = self.encoder(audios)
        
        inputs = torch.chunk(audios, chunks=audios.shape[0], dim=0)
        inputs = [item.squeeze(0).cpu().numpy() for item in inputs]
        mels = self.whisper_feature_extractor(inputs, sampling_rate=16000, return_tensors='pt')['input_features'].to(audios.device)
        whisper_embeds = self.whisper.model.encoder(mels)['last_hidden_state']

        whisper_embeds = F.interpolate(whisper_embeds.transpose(1, 2), size=audio_embeds.shape[1], mode='linear').transpose(1, 2)

        masks = torch.ones(audios.shape).to(audios.device)
        inputs = {
            "input_values": audios,
            "attention_mask": masks
        }
        outputs = self.mert(**inputs, output_hidden_states=True)
        mert_embeds = torch.stack(outputs['hidden_states']).permute(1, 0, 2, 3).squeeze()
        mert_embeds = self.conv(mert_embeds)
        if mert_embeds.ndim == 4:
            mert_embeds = mert_embeds.squeeze(1)
        mert_embeds = F.interpolate(mert_embeds.transpose(1, 2), size=audio_embeds.shape[1], mode='linear').transpose(1, 2)

        audio_embeds = torch.cat([audio_embeds, whisper_embeds, mert_embeds], dim=-1)

        # Qformer
        batch, tokens, dim = audio_embeds.shape
        kernel = (1, 17)  # for ced 714ms/per frame (ced 10s: 252 frame), we reduce to about 1.4 frames/second
        audio_embeds_new = F.unfold(
            audio_embeds.transpose(1, 2).unsqueeze(2), kernel_size=kernel, stride=kernel
        )
        audio_embeds_new = audio_embeds_new.view(batch, dim, kernel[1], -1)
        audio_embeds_new = torch.permute(audio_embeds_new, [0, 3, 2, 1])
        audio_embeds = audio_embeds_new.reshape(-1, kernel[1], dim)

        speech_atts = torch.ones(
            audio_embeds.size()[:-1], dtype=torch.long, device=audio_embeds.device
        )
        query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)

        audio_embeds = self.speech_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )["last_hidden_state"]

        # MLP
        encoder_hidden_states = []
        for mlp in self.enc_to_dec_proj:
            output = mlp(audio_embeds)
            output = output.view(batch, -1, output.size(2)).contiguous()
            encoder_hidden_states.append(output)

        prompt = self.fc(prompt)
        T_a = audio_embeds.shape[0] // batch
        T_t = prompt.shape[1]
        if T_t > T_a:
            prompt = prompt[:,:T_a]
        else:
            prompt = F.pad(prompt, (0,0,0,T_a-T_t))
        prompt = torch.reshape(prompt, (-1, 1, prompt.size(-1))).contiguous()        

        router_embeds = torch.cat([audio_embeds, prompt], dim=-1)
        weights = self.router(router_embeds)
        weights = weights.view(batch, -1, weights.size(2)).contiguous()
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)

        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=2)
        encoder_hidden_states = torch.sum(encoder_hidden_states * weights, dim=1)

        encoder_atts = torch.ones(
            encoder_hidden_states.size()[:-1], dtype=torch.long
        ).to(encoder_hidden_states.device)
        return encoder_hidden_states, encoder_atts

    def forward(self, samples):
        audios = samples["audios"]
        text = samples["text"]
        if "task" in samples:
            task = samples["task"]
        else:
            task = ["AAC"] * audios.shape[0]
        prompt = [self.prompt[t] for t in task]
        
        prompt_embeds = self.get_prompt_embeds(prompt, audios)

        # encoder
        encoder_hidden_states, encoder_atts = self.forward_encoder(audios, prompt_embeds)

        input_embeds, input_mask, decoder_targets, audio_embeds, text_embeds = (
            self.prepare_inputs_labels_for_multimodal(
                encoder_hidden_states, encoder_atts, prompt, text
            )
        )

        T_a, T_t = audio_embeds.shape[1], text_embeds.shape[1]
        if T_a > T_t:
            text_embeds = F.interpolate(
                text_embeds.transpose(1, 2), size=T_a, mode="linear"
            ).transpose(1, 2)
        elif T_a < T_t:
            audio_embeds = F.interpolate(
                audio_embeds.transpose(1, 2), size=T_t, mode="linear"
            ).transpose(1, 2)
   

        significance_scores = self.lps_module(audio_embeds, text_embeds)
        decision_matrix = (significance_scores > 0.5).float()
        calibrated_patches = self.spc_module(audio_embeds, decision_matrix)
        alignment_score = self.spa_module(calibrated_patches, text_embeds)

        positive_score = alignment_score

        negative_text_features = text_embeds[torch.randperm(text_embeds.size(0))]
        negative_score = self.spa_module(calibrated_patches, negative_text_features)


        lpa_loss = self.lps_loss_fn(alignment_score, positive_score, negative_score, significance_scores)


        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            labels=decoder_targets,
            return_dict=True,
        )

        ce_loss = decoder_output.loss        

        total_loss = lpa_loss + ce_loss

        return total_loss, decoder_output.logits

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=2,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        audios = samples["audios"].to(self.device)

        if "task" in samples:
            task = samples["task"]
        else:
            task = ["AAC"] * audios.shape[0]
        prompt = [self.prompt[t] for t in task]

        prompt_embeds = self.get_prompt_embeds(prompt, audios)
        
        encoder_hidden_states, encoder_atts = self.forward_encoder(audios, prompt_embeds)
        input_embeds, input_mask, decoder_targets, audio_embeds, text_embeds = (
            self.prepare_inputs_labels_for_multimodal(
                encoder_hidden_states, encoder_atts, prompt
            )
        )

        outputs = self.decoder.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=1.0,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        captions = self.tokenizer.batch_decode(outputs, add_special_tokens=False)
        return captions




if __name__ == "__main__":
    config = {
        "encoder_conf": {
            "encoder_strategy" : "lora"
        },
        "decoder_conf": {
            "decoder_strategy" : "lora"
        }
    }
    model = USAM(config)
    inputs = torch.rand(4, 16000*30)
    sample = {
    "audios": inputs,
    "text": "testing testing testing testing testing testing testing testing testing testing testing testing"
    }
    emb, att = model.forward(sample)
    print(f'emb: {emb.shape}')
    print(f'att: {att.shape}')
