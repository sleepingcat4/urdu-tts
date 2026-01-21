Have spent 2 hours on this rubbish. I have trained and made a nice repo to train MS's VibeVoice using an urdu dataset and written helper script to make a dataset. 

Create the dataset using `data.py` file and install below lib

#### Data
Please find the data here: https://huggingface.co/datasets/sleeping-ai/exp-urdu

#### Fine-tuning for children
```bash
git clone https://github.com/vibevoice-community/VibeVoice.git
cd VibeVoice/

pip install -e .
```

**Now run below command to train the model, had some problems with multi-gpu but ran with single 1xH100s. 

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m vibevoice.finetune.train_vibevoice \
    --model_name_or_path vibevoice/VibeVoice-1.5B \
    --dataset_name sleeping-ai/exp-urdu \
    --text_column_name text \
    --audio_column_name audio \
    --output_dir finetune_vibevoice_urdu \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2.5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --report_to wandb \
    --remove_unused_columns False \
    --bf16 True \
    --do_train \
    --gradient_clipping \
    --gradient_checkpointing False \
    --ddpm_batch_mul 4 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8
````

#### Acknowledgement 
A Huge shout-out to [Tensorpool](https://tensorpool.dev/) for providing the GPUs to Sleeping AI.  
