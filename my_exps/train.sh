#! /bin/bash
env=yolox_inst
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${env}
echo -e "\033[34m*******************************\033[0m"
echo -e "\033[31mactivate env ${env}\033[0m"
echo -e "\033[34m*******************************\033[0m"
cd ~/rcf/main_scripts/YOLOX_INST
echo -e "\033[34mCurrent dir is ${PWD}\033[0m"

exp_root=exps/example/yolox_coco
config_name=yolox_condinst_s
config=${exp_root}/${config_name}.py
cuda=$1
ckpt=$2
work_dir=YOLOX_outputs/${config_name}
echo -e "\033[33mconfig is ${config}\033[0m"
echo -e "\033[33mwork_dir is ${work_dir}\033[0m"
echo -e "\033[33mdevice is cuda: ${cuda}\033[0m"
sleep 3s
if [ -d ${work_dir} ]; then
    read -n1 -p "find ${work_dir}, do you want to del(y or n):"
    echo 
    if [ ${REPLY}x = yx ]; then  
		rm -rf ${work_dir}
		echo -e "\033[31mAlready del ${work_dir}\033[0m"
    else
		find ${work_dir} -name "events.out*"
		find ${work_dir} -name "train_log.txt"
		read -n1 -p "do you want to del log(y or n):"
		echo
		if [ ${REPLY}x = yx ]; then
			rm -rf *log*
			echo -e "\033]31mAlready del log files\033[0m"
		fi
    fi
fi
echo -e "\033[34m*******************************\033[0m"
CUDA_VISIBLE_DEVICES=${cuda} python tools/train.py -f ${config} -d 2 -b 16 -c ${ckpt} \
  --cache --fp16
    
