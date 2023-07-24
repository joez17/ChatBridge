
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node 1 --master_port=1234 \
train_custom.py \
--cfg-path instructiontuning_configs/instructtuning_imgsingle_resumecheck.yaml

CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node 4 --master_port=1734 \
evaluate.py \
--cfg-path eval_configs/qa/okvqa_eval_isosrcnew.yaml

torchrun --nproc_per_node 1 --master_port=1434 \
train_custom.py \
--cfg-path instructiontuning_configs/instructtuning_all_newweight_womm.yaml


torchrun --nproc_per_node 8 --master_port=1234 \
train_custom.py \
--cfg-path train_configs/minigpt4_stage1_fromtwomodal_doubledata_newpt.yaml

torchrun --nproc_per_node 8 --master_port=1231 \
evaluate.py \
--cfg-path eval_configs/caption/clothocap_eval.yaml \
--options model.ckpt=/mnt/bn/vlpopt/zhaozijia/LLM/minigpt4/minigpt4_stage1_pretrain_fromaudioonly_ftqformer_audio/20230511000/checkpoint_9.pth \
run.output_dir="$output_dir$type";

torchrun --nproc_per_node 8 --master_port=1231 \
evaluate.py \
--cfg-path eval_configs/qa/okvqa_eval.yaml \
--options model.ckpt=/mnt/bn/vlpopt/zhaozijia/LLM/minigpt4/minigpt4_stage1_fromblipflan_cclaion/20230510202/checkpoint_7.pth

template="###Human: Given following image: <query>,  Answer the following question with a one-word answer based on the content of the given image, <question>###Assistant:"
echo $template
torchrun --nproc_per_node 8 --master_port=1231 \
evaluate.py \
--cfg-path eval_configs/qa/vqa_eval.yaml \
--options model.ckpt=/mnt/bn/vlpopt/zhaozijia/LLM/minigpt4/minigpt4_stage1_fromblipflan_cclaion/20230510202/checkpoint_7.pth \

torchrun --nproc_per_node 8 --master_port=1231 \
evaluate.py \
--cfg-path eval_configs/caption/asvd/avsd_eval_video.yaml 



CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port=1325 \
evaluate.py \
--cfg-path eval_configs/qa/msrvtt_avqa_eval_4f.yaml 

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=1239 \
evaluate.py \
--cfg-path eval_configs/caption/clothocap_eval.yaml

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port=1236 \
train_custom.py \
--cfg-path instructiontuning_configs/instructtuning_imgsingle.yaml

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port=1235 \
train_custom.py \
--cfg-path instructiontuning_configs/instructtuning_imgsingle_resumecheck.yaml


type=msvdqa 
torchrun --nproc_per_node 8 --master_port=1231 \
evaluate.py \
--cfg-path eval_configs/qa_new/msvd_videoqa_eval_4f.yaml \
--options model.ckpt=$path \
run.output_dir="$output_dir$type";

torchrun --nproc_per_node 1 --master_port=1231 \
evaluate.py \
--cfg-path eval_configs/caption/valorcaps_eval.yaml \
--options model.ckpt=$path \
run.output_dir="$output_dir$type";

