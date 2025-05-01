import json

# 输入文件路径和输出文件路径
input_file = "./DISC-Law-SFT/DISC-Law-SFT-Pair.jsonl"  # 输入文件
output_file = "./DISC-Law-SFT/DISC-Law-SFT-Pair_alpaca.json"  # 输出文件

# 打开输入文件读取数据
with open(input_file, 'r', encoding='utf-8') as f_in:
    lines = f_in.readlines()

# 处理数据，转换为 Alpaca 格式
alpaca_data = []
for line in lines:
    data = json.loads(line)
    alpaca_format = {
        "instruction": data["input"],  # 将 "input" 转换为 "instruction"
        "input": "",  # 如果没有特定输入，则为空字符串
        "output": data["output"]  # 将 "output" 转换为 "response"
    }
    alpaca_data.append(alpaca_format)

# 将处理后的数据写入输出文件
with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(alpaca_data, f_out, ensure_ascii=False, indent=2)

print("数据已转换并保存为 Alpaca 格式。")
