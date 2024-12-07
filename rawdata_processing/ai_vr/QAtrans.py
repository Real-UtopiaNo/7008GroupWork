import csv

# 处理QA.txt格式：Q: 和 A: 分隔
def convert_QA_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = infile.readlines()  # 读取所有行
        writer = csv.writer(outfile)
        writer.writerow(['Question', 'Answer'])  # 写入标题行
        
        question = None  # 用于存储当前的问问题
        answer = None  # 用于存储当前的答案

        for line in reader:
            line = line.strip()
            if line.startswith('Q:'):  # 如果是问题
                if question and answer:  # 如果有未写入的问答对
                    writer.writerow([question, answer])  # 写入问答对
                question = line[2:].strip()  # 获取问题内容
                answer = None  # 重置答案
            elif line.startswith('A:'):  # 如果是答案
                answer = line[2:].strip()  # 获取答案内容
        
        # 最后一个问答对写入
        if question and answer:
            writer.writerow([question, answer])

# 处理QA2.txt格式：空行分隔问答对
def convert_QA2_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = infile.read().strip().split('\n\n')  # 通过空行分隔问答对
        writer = csv.writer(outfile)
        writer.writerow(['Question', 'Answer'])  # 写入标题行

        for block in reader:
            lines = block.strip().split('\n')  # 每对问答的块按行分开
            if len(lines) == 2:  # 确保每块只有问题和答案
                question = lines[0].strip()
                answer = lines[1].strip()
                writer.writerow([question, answer])  # 写入问答对

# 调用函数进行转换
convert_QA_to_csv('QA.txt', 'QA.csv')
convert_QA2_to_csv('QA2.txt', 'QA2.csv')
