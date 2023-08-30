set -e

# inputs config
config='13B'
model='llama'
llama_version='llama_2'
weights_dir='./weights_2'

# inputs paths
checkpoint_dir="./$weights_dir/$config/$config/"
tokenizer_file="./$weights_dir/$config/tokenizer.model"
koala_diff_file="./$weights_dir/koala_diffs/koala_7b_diff_v2"


# derived vars
checkpoint_file="./$weights_dir/$config/$config.pt"
koala_output_file="./$weights_dir/$config/koala_$config.pt"

port=0
if [[ $model == "koala" ]]; then
    port=$((port + 2000))
    load_checkpoint_file=$koala_output_file
elif [[ $model == "llama" ]]; then
    if [[ $llama_version == "llama_2" ]]; then
        port=$((port + 3000))
        load_checkpoint_file=$checkpoint_file
    elif [[ $llama_version == "llama_2_chat" ]]; then
        port=$((port + 3500))
        load_checkpoint_file=$checkpoint_file
    else
        port=$((port + 4000))
        load_checkpoint_file=$checkpoint_file
    fi
fi

if [[ $config == "7B" ]]; then
    port=$((port + 7))
    config_lower='7b'
elif [[ $config == "13B" ]]; then
    port=$((port + 13))
    config_lower='13b'
elif [[ $config == "30B" ]]; then
    port=$((port + 30))
    config_lower='30b'
elif [[ $config == "65B" ]]; then
    port=$((port + 65))
    config_lower='65b'
fi


# process checkpoints
# python -m EasyLM.models.llama.convert_torch_to_easylm \
#    --checkpoint_dir=$checkpoint_dir \
#    --output_file=$checkpoint_file \
#    --streaming=True

# rm -r $checkpoint_dir
    
# python -m EasyLM.scripts.diff_checkpoint \
#    --recover_diff=True \
#    --load_base_checkpoint="params::$checkpoint_file" \
#    --load_target_checkpoint="params::$koala_diff_file" \
#    --output_file=$koala_output_file \
#    --streaming=True

# serve
python -m EasyLM.models.llama.llama_serve \
    --load_llama_config=$config_lower \
    --load_checkpoint="params::$load_checkpoint_file" \
    --tokenizer.vocab_file=$tokenizer_file \
    --mesh_dim='1,1,-1' \
    --dtype='bf16' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --lm_server.batch_size=1 \
    --lm_server.port=$port \
    --lm_server.pre_compile='loglikelihood' \
    --lm_server.chat_prepend_text='BEGINNING OF CONVERSATION: ' \
    --lm_server.chat_lm_prefix='GPT:' \
    --lm_server.chat_lm_suffix='</s>' \
    --lm_server.chat_user_prefix='USER: ' \
    --lm_server.chat_user_suffix=' '

