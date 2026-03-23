import tensorflow_datasets as tfds
import os

# 屏蔽底层 C++ 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 指向你的数据集目录
dataset_path = "/root/autodl-tmp/roboticAttack/openvla-main/dataset/libero_goal_no_noops/1.0.0"

print(f"正在加载 RLDS 数据集: {dataset_path} ...")
builder = tfds.builder_from_directory(dataset_path)
dataset = builder.as_dataset(split='train')

# 遍历数据集的前 150 条完整轨迹 (Episode)
print("\n=== 数据集前 150 条轨迹的任务分布 ===")
for i, episode in enumerate(dataset.take(150)):
    # 取出这条轨迹的第一步(Frame)
    step = next(iter(episode['steps']))
    
    # 提取文本语言指令 (判断属于哪个任务)
    # 注意：RLDS 格式中语言指令通常在 'language_instruction' 或 'observation' 里
    if 'language_instruction' in step['observation']:
        task_text = step['observation']['language_instruction'].numpy().decode('utf-8')
    elif 'language_instruction' in step:
        task_text = step['language_instruction'].numpy().decode('utf-8')
    else:
        task_text = "未找到指令"
        
    print(f"轨迹 (Episode) {i:03d} -> 任务指令: {task_text}")
    
    # 我们每隔 10 条轨迹打印一次，方便你看清数据分布规律
    if i % 10 != 0 and i != 0:
        continue