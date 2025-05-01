readme:
Please follow my steps to reproduce my results:
1. Installation of Llama-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

2. Data Preparation of DISC-LAW
running trans.py for processing all the datasets(DISC-Law-SFT-Triplet-released.json，DISC-Law-SFT-Triplet-QA-released.json，DISC-Law-SFT-Pair-QA-released.json，DISC-Law-SFT-Pair.json) in Alpaca format
Move all this four data files in ./data folder, and add info in dataset_info.json:
  "DISC-LAW-Triplet":{
    "file_name":"DISC-Law-SFT-Triplet-released_alpaca.json",
    "columns":{
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "DISC-LAW-Triplet-QA":{
    "file_name":"DISC-Law-SFT-Triplet-QA-released_alpaca.json",
    "columns":{
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "DISC-LAW-Pair-QA":{
    "file_name":"DISC-Law-SFT-Pair-QA-released_alpaca.json",
    "columns":{
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  } ,
  "DISC-LAW-Pair":{
    "file_name":"DISC-Law-SFT-Pair_alpaca.json",
    "columns":{
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }

3. Download the qwen2.5-3B-Instruct model
Download the qwen2.5-3B-Instruct.

4. Fine-tune the model
cd LLaMA-Factory
Create the lora yaml file in examples/train_lora/, named qwen2.5_3b_lora_sft.yaml:
set the downloaded model path: model_name_or_path: /home/yaodong/codes/deepLearning/partb/Qwen2.5-3B-Instruct
set the dataset: DISC-LAW-Pair,DISC-LAW-Pair-QA,DISC-LAW-Triplet,DISC-LAW-Triplet-QA
set the output path: ../saves/qwen2.5-3B-Instruct/lora/sft/
run this command in terminal: CUDA_VISIBLE_DEVICES=7 llamafactory-cli train examples/train_lora/qwen2.5_3b_lora_sft.yaml

5. Inference and test
cd LLaMA-Factory
Create the lora yaml file in examples/inference/, named qwen2.5_3b_lora_sft.yaml:
set the downloaded model path: model_name_or_path: /home/yaodong/codes/deepLearning/partb/Qwen2.5-3B-Instruct
set the lora result: adapter_name_or_path: /home/yaodong/codes/deepLearning/partb/saves/qwen2.5-3B-Instruct/lora/sft/
run this command in terminal: CUDA_VISIBLE_DEVICES=7 llamafactory-cli chat examples/inference/qwen_lora_sft.yaml
