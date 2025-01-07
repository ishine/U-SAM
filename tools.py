import pandas as pd
import os
import soundfile as sf
import json
from tqdm import tqdm
import glob
import io
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import tarfile
import random
import string
import torchaudio
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count

'''
key  -->   wav     -->    label -->  --> dur    --> sr   --> channels  --> description --> tag/category
唯一标识    音频路径        caption         时长      采样率     通道数              附加信息...
'''


def process_audiocaps(d):
    scp_d = os.path.join(d, 'scp')
    if not os.path.exists(scp_d):
        os.makedirs(scp_d)

    wavs_train = os.listdir(os.path.join(d, 'train'))
    wavs_test = os.listdir(os.path.join(d, 'test'))
    wavs_eval = os.listdir(os.path.join(d, 'val'))
    
    df_train = pd.read_csv(os.path.join(d, 'train.csv'))
    df_val = pd.read_csv(os.path.join(d, 'val.csv'))
    df_test = pd.read_csv(os.path.join(d, 'test.csv'))

    # df_list = [df_train, df_val, df_test]
    df_list = [df_val]


    for i, df in enumerate(df_list):
        for _, row in tqdm(df.iterrows()):
            key = row['youtube_id']
            label = row['caption']
            name = str(row['audiocap_id']) + '.wav'

            if name in wavs_train:
                wav_path = os.path.join(d, 'train', name)
            elif name in wavs_test:
                wav_path = os.path.join(d, 'test', name)
            elif name in wavs_eval:
                wav_path = os.path.join(d, 'val', name)    
            else:
                print(f'{name} not found in train or test')
                continue
            wav, sr = sf.read(wav_path)
            entry = {
                "key": key,
                'wav': wav_path,
                "label": label,
                "dur": len(wav) / sr,
                "sr": sr,
                "channels": wav.shape[1] if wav.ndim == 2 else 1,
            }

            val_scp_path = os.path.join(scp_d, 'val.scp')
            with open(val_scp_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            # if i == 0:
            #     train_scp_path = os.path.join(scp_d, 'train.scp')
            #     with open(train_scp_path, 'a') as f:
            #         f.write(json.dumps(entry) + '\n')
            # elif i == 1:
            #     val_scp_path = os.path.join(scp_d, 'val.scp')
            #     with open(val_scp_path, 'a') as f:
            #         f.write(json.dumps(entry) + '\n')
            # else:
            #     test_scp_path = os.path.join(scp_d, 'test.scp')
            #     with open(test_scp_path, 'a') as f:
            #         f.write(json.dumps(entry) + '\n')




def process_clotho(d):
    scp_d = os.path.join(d, 'scp')
    if not os.path.exists(scp_d):
        os.makedirs(scp_d)    
    train_scp_path = os.path.join(scp_d, 'train.scp')
    val_scp_path = os.path.join(scp_d, 'val.scp')
    test_scp_path = os.path.join(scp_d, 'test.scp')

    csvs = glob.glob(os.path.join(d, 'csv/*.csv'))
    for csv in csvs:
        df = pd.read_csv(csv)
        if csv.count('development') > 0:
            wav_d = os.path.join(d, 'development')
        elif csv.count('evaluation') > 0:
            wav_d = os.path.join(d, 'evaluation')
        elif csv.count('validation') > 0:
            wav_d = os.path.join(d, 'validation')
        else:
            raise RuntimeError(f'non exists: {csv}')
        
        for _, row in tqdm(df.iterrows()):
            wav_path = os.path.join(wav_d, row['file_name'])
            try:
                wav, sr = sf.read(wav_path)
            except Exception as e:
                print(f'reading wav error: {wav_path}')
                continue
            
            for i in range(1, 6):
                label = row[f'caption_{i}']

                entry = {
                    "key": row["file_name"],
                    "wav": wav_path,
                    "label": label,
                    "dur": len(wav) / sr,
                    "sr": sr,
                    "channels": wav.shape[1] if wav.ndim == 2 else 1
                }

                if csv.count('development') > 0: 
                    with open(train_scp_path, 'a') as f:
                        f.write(json.dumps(entry) + '\n')
                elif csv.count('evaluation') > 0:
                    with open(test_scp_path, 'a') as f:
                        f.write(json.dumps(entry) + '\n')
                elif csv.count('validation') > 0:
                    with open(val_scp_path, 'a') as f:
                        f.write(json.dumps(entry) + '\n')
                


def process_sample(sample, wav_d_sub, subset):
    wav_filename = sample['id'] + '.flac' if subset != 'AudioSet_SL' else sample['id'].replace('wav', 'flac')
    wav_path = os.path.join(wav_d_sub, wav_filename)
    
    try:
        # Only fetch audio metadata
        info = sf.info(wav_path)
        d = {
            "key": sample['id'],
            'wav': wav_path,
            "label": sample['caption'],
            "dur": info.duration,
            "sr": info.samplerate,
            "channels": info.channels,
        }
        
        # Add optional fields if present
        for key in ['description', 'tag', 'category']:
            if key in sample:
                d[key] = sample[key]
                
        return json.dumps(d)
    
    except Exception as e:
        print(f'Error reading {wav_path}: {e}')
        return None

def process_json_file(jf, wav_d_sub, subset):
    with open(jf) as f:
        data = json.load(f)

    for sample in tqdm(data['data'], desc=f'Processing samples in {os.path.basename(jf)}', leave=False):
        result = process_sample(sample, wav_d_sub, subset)
        if result:
            yield result

def process_wavcaps(d):
    scp_d = os.path.join(d, 'scp')
    os.makedirs(scp_d, exist_ok=True)

    json_d = os.path.join(d, 'json_files') 
    wav_d = os.path.join(d, 'Zip_files')
    subsets = os.listdir(wav_d)

    for subset in subsets:
        if subset != 'FreeSound':
            continue

        print(f'Processing subset: {subset}')
        json_d_sub = os.path.join(json_d, subset)
        wav_d_sub = os.path.join(wav_d, subset, 'waveforms')
        json_files = glob.glob(os.path.join(json_d_sub, '*.json'))

        output_path = os.path.join(scp_d, f'{subset}.scp')
        with open(output_path, 'w') as f:
            for jf in json_files:
                for result in process_json_file(jf, wav_d_sub, subset):
                    f.write(result + '\n')
        
    print("Processing complete.")



def merge_scp():
    audiocaps = '/mnt/bn/wangziqian-nas/audiocaps/scp/train.scp'
    clotho = '/mnt/bn/wangziqian-nas/clotho/scp/train.scp'
    wavcaps = '/mnt/bn/wangziqian-nas/wavcaps/scp/wavcap_train.scp'

    data = [audiocaps, clotho, wavcaps]
    with open('/mnt/bn/wangziqian-nas/LOAE/data/train_all.scp', 'w') as f:
        for d in tqdm(data):
            with open(d, 'r') as f_in:
                for line in f_in:
                    f.write(line)            
    

def generate_random_string(length=5):
    letters = string.ascii_letters 
    return ''.join(random.choice(letters) for _ in range(length))

# def parse_json_to_tar(json_file, tar_file):
#     seen = set()
#     with tarfile.open(tar_file, 'w') as tar:
#         with open(json_file, 'r') as f:
#             for i, line in tqdm(enumerate(f)):
#                 # 解析 JSON 对象
#                 obj = json.loads(line.strip())
                
#                 # 保存 JSON 文件
#                 json_str = json.dumps(obj)
                
#                 key = obj['key']
#                 if key not in seen:
#                     json_file = f"{key}.json"
#                 else:
#                     new_key = key+generate_random_string()
#                     obj['key'] = new_key
#                     json_file = f"{new_key}.json"

#                 seen.add(key)
                
#                 # 写入 tar 文件
#                 with open(json_file, 'w') as json_f:
#                     json_f.write(json_str)
#                 tar.add(json_file, arcname=json_file)
#                 os.remove(json_file)  # 删除中间文件                


def process_line(line):
    seen = set()
    obj = json.loads(line.strip())
    key = obj['key']

    # 如果 key 已存在，生成新 key
    if key in seen:
        key += generate_random_string()
        obj['key'] = key
    seen.add(key)

    # 准备 JSON 字符串
    # json_str = json.dumps(obj).encode('utf-8')
    json_str = json.dumps(obj, ensure_ascii=False).encode('utf-8')
    json_file = f"{key}.json"
    
    return json_file, json_str

def parse_json_to_tar(json_file, tar_file):
    # 使用 tarfile 打开文件并准备进程池
    seen = set()
    with tarfile.open(tar_file, 'w') as tar:
        with open(json_file, 'r') as f, ProcessPoolExecutor() as executor:
            futures = []
            for line in f:
                futures.append(executor.submit(process_line, line))

            # 使用 tqdm 进行进度跟踪
            for future in tqdm(as_completed(futures), total=len(futures)):
                json_file, json_str = future.result()
                
                # 创建 TarInfo 并添加到 tar 文件
                json_info = tarfile.TarInfo(name=json_file)
                json_info.size = len(json_str)
                
                with io.BytesIO(json_str) as json_bytes:
                    tar.addfile(json_info, json_bytes)



def check_tarfile(f):
    with tarfile.open(f, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.json'):
                # 提取文件并读取内容
                f = tar.extractfile(member)
                if f is not None:  # 确保文件成功提取
                    content = f.read()
                    # 尝试解析为 JSON
                    try:
                        json_data = json.loads(content)
                        print(f"Contents of {member.name}:")
                        print(json.dumps(json_data, indent=2))  # 美化打印 JSON
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {member.name}: {e}")


def split_tar(input_tar, output_dir, files_per_shard):
    """
    将一个大的 tar 文件分割成多个较小的 tar 文件。
    
    参数：
    - input_tar: 原始的 tar 文件路径
    - output_dir: 分割后小 tar 文件的保存目录
    - files_per_shard: 每个分片文件中包含的文件数量
    """
    # 创建保存分片文件的目录
    os.makedirs(output_dir, exist_ok=True)
    
    with tarfile.open(input_tar, 'r') as src_tar:
        # 计数器
        file_counter = 0
        shard_counter = 0
        current_shard = None
        
        for member in src_tar.getmembers():
            # 如果当前分片文件为空或已达到最大文件数，则创建新分片文件
            if file_counter % files_per_shard == 0:
                if current_shard:
                    current_shard.close()
                
                shard_name = f"dataset-{shard_counter:03d}.tar"
                current_shard_path = os.path.join(output_dir, shard_name)
                current_shard = tarfile.open(current_shard_path, 'w')
                
                print(f"正在创建新的 shard: {shard_name}")
                shard_counter += 1
            
            # 添加文件到当前 shard
            file = src_tar.extractfile(member)
            if file:
                current_shard.addfile(member, file)
                file_counter += 1
        
        # 关闭最后一个 shard
        if current_shard:
            current_shard.close()


def check_and_rename_duplicates(tar_path, output_tar_path):
    # 用于跟踪文件名和重复计数
    file_count = {}

    # 创建一个新的 tar 文件，来存放重命名后的文件
    with tarfile.open(output_tar_path, 'w') as out_tar:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                # 获取文件名（去除目录路径）
                file_name = os.path.basename(member.name)
                
                # 检查是否存在重复文件名
                if file_name in file_count:
                    # 文件名重复，更新计数并重命名文件
                    file_count[file_name] += 1
                    new_file_name = f"{os.path.splitext(file_name)[0]}_{file_count[file_name]}{os.path.splitext(file_name)[1]}"
                    print(f"Duplicate file found: {file_name}. Renaming to {new_file_name}.")
                else:
                    # 文件名唯一，初始化计数
                    file_count[file_name] = 1
                    new_file_name = file_name
                
                # 提取文件内容并重命名添加到新的 tar 文件中
                extracted_file = tar.extractfile(member)
                new_member = tarfile.TarInfo(name=new_file_name)
                new_member.size = member.size
                out_tar.addfile(new_member, extracted_file)
    
    print(f"Done! New tar file created at: {output_tar_path}")





def check_and_shorten_audio(tar_path, output_tar_path, max_duration=10.0):
    print(f"Processing {tar_path}")
    with tarfile.open(tar_path, 'r') as tar, tarfile.open(output_tar_path, 'w') as out_tar:
        for member in tar.getmembers():
            file_name = os.path.basename(member.name)
            
            # 针对 JSON 文件
            if file_name.endswith('.json'):
                extracted_file = tar.extractfile(member)
                sample = json.load(extracted_file)

                # 检查音频路径
                wav_file = sample.get("wav")
                if wav_file and os.path.isfile(wav_file):
                    # 获取音频信息
                    info = torchaudio.info(wav_file)
                    duration = info.num_frames / info.sample_rate

                    # 如果音频过长，进行截短
                    if duration > max_duration:
                        new_wav_file = os.path.splitext(wav_file)[0] + "_shortened.wav"
                        # 读取并截取音频
                        waveform, sample_rate = torchaudio.load(wav_file)
                        max_samples = int(max_duration * sample_rate)
                        truncated_waveform = waveform[:, :max_samples]
                        
                        # 保存截短的音频文件
                        torchaudio.save(new_wav_file, truncated_waveform, sample_rate)
                        
                        # 更新JSON中的音频文件路径和持续时间
                        sample["wav"] = new_wav_file
                        sample["dur"] = max_duration

                        # 将更新后的JSON重新写入tar文件
                        json_bytes = json.dumps(sample).encode('utf-8')
                        json_info = tarfile.TarInfo(name=member.name)
                        json_info.size = len(json_bytes)
                        out_tar.addfile(json_info, io.BytesIO(json_bytes))

                        # # 处理新的音频文件
                        # audio_info = tarfile.TarInfo(name=os.path.basename(new_wav_file))
                        # audio_info.size = os.path.getsize(new_wav_file)
                        # with open(new_wav_file, 'rb') as audio_file:
                        #     out_tar.addfile(audio_info, audio_file)

                        # # 删除临时生成的音频文件
                        # os.remove(new_wav_file)
                    else:
                        # 如果音频时长符合要求，则直接复制JSON和音频文件
                        out_tar.addfile(member, tar.extractfile(member))

                        # 将音频文件也添加到新的tar文件中
                        # audio_member = tar.getmember(wav_file)
                        # out_tar.addfile(audio_member, tar.extractfile(audio_member))

def process_all_shards(input_dir, output_dir, max_duration=10.0):
    # for shard in os.listdir(input_dir):
    #     if shard.endswith(".tar"):
    #         input_tar_path = os.path.join(input_dir, shard)
    #         output_tar_path = os.path.join(output_dir, shard)
    #         print(f"Processing {input_tar_path}")
    #         check_and_shorten_audio(input_tar_path, output_tar_path, max_duration)

    shard_files = [f for f in os.listdir(input_dir) if f.endswith('.tar') and f.count('checked') > 0]
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(check_and_shorten_audio,
                            os.path.join(input_dir, shard),
                            os.path.join(output_dir, shard),
                            max_duration)
            for shard in shard_files
        ]
        for future in as_completed(futures):
            future.result()



def make_scp(d, path):
    with open(path, 'w') as f:
        wavs = glob.glob(os.path.join(d, '*.wav'))
        for p in tqdm(wavs):
            info = sf.info(p)
            entry = {
                "key": os.path.basename(p),
                'wav': p,
                "label": "not exists",
                "dur": info.duration,
                "sr": info.samplerate,
                "channels": info.channels,
            }
            f.write(json.dumps(entry) + '\n')


def process_covost_old(tsv, wavd, scp):
    meta = pd.read_csv(tsv, sep='\t', on_bad_lines='skip')
    with open(scp, 'w') as f:
        for id, row in tqdm(meta.iterrows()):
            try:
                name = row['path']
                transcription = row['sentence']
                label = row['translation']
                client_id = row['client_id']

                wav_path = os.path.join(wavd, name)

                # info = sf.info(wav_path)
                audio = AudioSegment.from_file(wav_path)

                # 获取采样率（frame_rate）、时长（duration_seconds）和通道数（channels）
                sample_rate = audio.frame_rate
                duration = len(audio) / 1000  # pydub 以毫秒为单位，除以1000转换为秒
                channels = audio.channels

                d = {
                    "key": client_id,
                    'wav': wav_path,
                    "label": label,
                    "transcription": transcription,
                    "dur": duration,
                    "sr": sample_rate,
                    "channels": channels,
                    }
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f'error processing: {id} - error: {e}')



# 定义处理每一行的函数
def process_row(row, wavd):
    try:
        name = row.path
        transcription = row.sentence
        label = row.translation
        client_id = row.client_id

        wav_path = os.path.join(wavd, name)
        
        # # 使用 pydub 读取音频
        # audio = AudioSegment.from_file(wav_path)

        # # 获取采样率、时长、通道数
        # sample_rate = audio.frame_rate
        # duration = len(audio) / 1000  # 转换为秒
        # channels = audio.channels

        # 构建字典
        return {
            "key": client_id,
            "wav": wav_path,
            "label": label,
            "transcription": transcription,
            #"dur": duration,
            #"sr": sample_rate,
            #"channels": channels,
        }
    except Exception as e:
        return f'Error processing file {row.Index}: {e}'

# 处理数据集的主函数
def process_covost(tsv, wavd, scp, num_workers=100, batch_size=1000):
    # 读取元数据
    meta = pd.read_csv(tsv, sep='\t', on_bad_lines='skip')
    
    # 用 ThreadPoolExecutor 进行并行处理
    with open(scp, 'w') as f, ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        # 批量处理数据，加快文件 I/O
        for i in tqdm(range(0, len(meta), batch_size)):
            batch = meta[i:i+batch_size]

            # 使用 itertuples()，比 iterrows() 快
            for row in batch.itertuples():
                futures.append(executor.submit(process_row, row, wavd))
        
        # 处理并写入文件
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if isinstance(result, dict):
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                print(result)




# if __name__ == '__main__':
#     d = '/mnt/bn/xianjun-lf/wangziqian/audioLM/commonvoice4/cv-corpus-19.0-2024-09-13/en/clips'
#     tsv = ['/mnt/bn/xianjun-lf/wangziqian/audioLM/covost2/covost_v2.en_zh-CN.train.tsv', '/mnt/bn/xianjun-lf/wangziqian/audioLM/covost2/covost_v2.en_zh-CN.dev.tsv', '/mnt/bn/xianjun-lf/wangziqian/audioLM/covost2/covost_v2.en_zh-CN.test.tsv']
    
#     for t in tsv:
#         scp = t.replace('tsv', 'scp')
#         process_covost(t, d, scp)



    # make_scp('/mnt/bn/wangziqian-nas/dy_testset_for_aac', '/mnt/bn/wangziqian-nas/LOAE/data/dy_testset_for_aac.scp')

    # process_audiocaps('/mnt/bn/wangziqian-nas/audiocaps')
    # process_clotho('/mnt/bn/wangziqian-nas/clotho')
    # process_wavcaps('/mnt/bn/wangziqian-nas/wavcaps')
    # merge_scp()

    # scp_files = glob.glob('/mnt/bn/xianjun-lf/wangziqian/audioLM/covost2/*.scp')

    # 使用 ThreadPoolExecutor 或 ProcessPoolExecutor
    # with ThreadPoolExecutor() as executor:  # 或使用 ProcessPoolExecutor()
    #     # 提交任务
    #     futures = {executor.submit(parse_json_to_tar, scp, scp.replace('scp', 'tar')): scp for scp in scp_files}

    #     # 等待所有任务完成
    #     for future in as_completed(futures):
    #         scp = futures[future]
    #         try:
    #             future.result()  # 获取结果或捕获异常
    #         except Exception as e:
    #             print(f"Error processing {scp}: {e}")

    # all_data = '/mnt/bn/wangziqian-nas/LOAE/data/train_all.scp'
    # parse_json_to_tar(all_data, all_data.replace('scp', 'tar'))

    # scp_f = 'test.scp'
    # parse_json_to_tar(scp_f, scp_f.replace('scp', 'tar'))

    # check_tarfile('/mnt/bn/wangziqian-nas/LOAE/data/eval_audiocaps.tar')

    # d = '/mnt/bn/xianjun-lf/wangziqian/audioLM/LOAE/data/s2tt/train_shards'
    # subdirs = glob.glob(d+'/dataset-0*')
    # for sd in subdirs:
    #     check_and_rename_duplicates(sd, sd.replace('dataset', 'dataset-checked'))

    # split_tar('/mnt/bn/wangziqian-nas/covost2/covost_v2.en_zh-CN.train.tar', '/mnt/bn/wangziqian-nas/LOAE/data/s2t/train_shards', 10000)


        # 设置参数并运行
    # input_dir = "/mnt/bn/wangziqian-nas/LOAE/data/train_all_shards"  # 输入shard的路径
    # output_dir = "/mnt/bn/wangziqian-nas/LOAE/data/train_all_shards_shorten"  # 处理后shard保存路径
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # max_duration = 10.0  # 设置音频的最大时长（秒）
    # process_all_shards(input_dir, output_dir, max_duration)



def merge_tar():
    # 定义目录和任务名称的映射
    data_dirs = {
        "/mnt/bn/wangziqian-nas/data/aac": "AAC",
        "/mnt/bn/wangziqian-nas/data/asr": "ASR",
        "/mnt/bn/wangziqian-nas/data/s2tt": "S2TT",
        "/mnt/bn/wangziqian-nas/data/mc": "MC"
    }

    # 设置输出目录
    output_dir = "/mnt/bn/wangziqian-nas/data/all_data/train"
    os.makedirs(output_dir, exist_ok=True)
    tar_file_counter = 0  # tar 文件的顺序计数器
    samples_per_tar = 10000  # 每个 tar 文件包含的样本数量
    all_samples = []  # 用于存储待打包的样本
    sample_counter = 0  # 每个 JSON 文件的唯一命名计数器

    # 处理每个目录
    for dir_name, task_name in data_dirs.items():
        base_path = f"{dir_name}/train_all_shards"

        # 遍历目录下所有 tar 文件
        for tar_file_name in os.listdir(base_path):
            tar_file_path = os.path.join(base_path, tar_file_name)
            
            # 打开 tar 文件并处理每个 JSON 文件
            with tarfile.open(tar_file_path, "r") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                with tqdm(total=len(members), desc=f"Processing {task_name} - {tar_file_name}", unit=" files") as pbar:
                    for member in members:
                        f = tar.extractfile(member)
                        if f is not None:
                            lines = f.read().decode("utf-8").splitlines()

                            # 为每个 JSON 对象创建独立的 JSON 文件
                            for line in lines:
                                data = json.loads(line)
                                data["task"] = task_name
                                all_samples.append(data)

                                # 当达到指定数量时，将数据写入 tar 文件
                                if len(all_samples) >= samples_per_tar:
                                    output_tar_file_name = f"dataset-{tar_file_counter:03d}.tar"
                                    output_tar_file_path = os.path.join(output_dir, output_tar_file_name)
                                    
                                    with tarfile.open(output_tar_file_path, "w") as new_tar:
                                        for sample in all_samples:
                                            sample_json = json.dumps(sample)
                                            sample_filename = f"{sample_counter:06d}.json"
                                            temp_json_file_path = os.path.join(output_dir, sample_filename)
                                            
                                            # 将 JSON 对象写入临时文件
                                            with open(temp_json_file_path, "w") as temp_file:
                                                temp_file.write(sample_json)

                                            # 添加到 tar 文件中
                                            new_tar.add(temp_json_file_path, arcname=sample_filename)
                                            os.remove(temp_json_file_path)  # 删除临时文件

                                            sample_counter += 1  # 更新计数器

                                    # 清空样本缓存，更新 tar 文件计数器
                                    all_samples = []
                                    tar_file_counter += 1

                        # 更新进度条
                        pbar.update(1)

    # 处理剩余不足 10000 个的样本
    if all_samples:
        output_tar_file_name = f"dataset-{tar_file_counter:03d}.tar"
        output_tar_file_path = os.path.join(output_dir, output_tar_file_name)

        with tarfile.open(output_tar_file_path, "w") as new_tar:
            for sample in all_samples:
                sample_json = json.dumps(sample)
                sample_filename = f"{sample_counter:06d}.json"
                temp_json_file_path = os.path.join(output_dir, sample_filename)
                
                with open(temp_json_file_path, "w") as temp_file:
                    temp_file.write(sample_json)

                new_tar.add(temp_json_file_path, arcname=sample_filename)
                os.remove(temp_json_file_path)

                sample_counter += 1

    print("所有数据已合并并顺序命名完成！")



def merge_scp():
    # 定义目录和任务名称的映射
    data_dirs = {
        "/mnt/bn/wangziqian-nas/data/aac": "AAC",
        "/mnt/bn/wangziqian-nas/data/asr": "ASR",
        "/mnt/bn/wangziqian-nas/data/s2tt": "S2TT",
        "/mnt/bn/wangziqian-nas/data/mc": "MC"
    }

    # 输出目录
    output_dir = "/mnt/bn/wangziqian-nas/data/all_data/eval"
    os.makedirs(output_dir, exist_ok=True)
    tar_file_counter = 0  # tar 文件的顺序计数器
    samples_per_tar = 10000  # 每个 tar 文件包含的样本数量
    all_samples = []  # 临时存储所有样本
    sample_counter = 0  # 每个样本的唯一命名计数器

    # 处理每个目录下的 eval.scp 文件
    for dir_name, task_name in data_dirs.items():
        if task_name == "S2TT":
            scp_files = [os.path.join(dir_name, "covost_v2.en_zh-CN.dev.scp")]
        elif task_name == "MC":
            scp_files = [os.path.join(dir_name, "mc_eval.scp")]
        elif task_name == "AAC":
            scp_files = [os.path.join(dir_name, "eval_audiocaps.scp"), os.path.join(dir_name, "eval_clotho.scp")]
        elif task_name == "ASR":
            scp_files = [os.path.join(dir_name, "dev-merge.scp")]
        else:
            continue

        for scp_file in scp_files:
            with open(scp_file, "r") as f:
                for line in tqdm(f, desc=f"Processing {task_name}", unit=" samples"):
                    data = json.loads(line.strip())
                    data["task"] = task_name
                    all_samples.append(data)

                    # 当达到指定数量时，将数据写入 tar 文件
                    if len(all_samples) >= samples_per_tar:
                        output_tar_file_name = f"dataset-{tar_file_counter:03d}.tar"
                        output_tar_file_path = os.path.join(output_dir, output_tar_file_name)
                        
                        with tarfile.open(output_tar_file_path, "w") as tar:
                            for sample in all_samples:
                                sample_json = json.dumps(sample)
                                sample_filename = f"{sample_counter:06d}.json"
                                temp_json_file_path = os.path.join(output_dir, sample_filename)

                                with open(temp_json_file_path, "w") as temp_file:
                                    temp_file.write(sample_json)

                                tar.add(temp_json_file_path, arcname=sample_filename)
                                os.remove(temp_json_file_path)

                                sample_counter += 1
                        
                        all_samples = []  # 清空样本缓冲区
                        tar_file_counter += 1

    # 若有剩余样本不足 1000 个，也将其写入新 tar 文件
    if all_samples:
        output_tar_file_name = f"dataset-{tar_file_counter:03d}.tar"
        output_tar_file_path = os.path.join(output_dir, output_tar_file_name)

        with tarfile.open(output_tar_file_path, "w") as tar:
            for sample in all_samples:
                sample_json = json.dumps(sample)
                sample_filename = f"{sample_counter:06d}.json"
                temp_json_file_path = os.path.join(output_dir, sample_filename)

                with open(temp_json_file_path, "w") as temp_file:
                    temp_file.write(sample_json)

                tar.add(temp_json_file_path, arcname=sample_filename)
                os.remove(temp_json_file_path)

                sample_counter += 1

    print("所有数据已合并、分批并压缩完成！")
    
merge_tar()
merge_scp()

def inspect_tar_file(tar_file_path):
    print(f"Inspecting {tar_file_path}...")
    try:
        with tarfile.open(tar_file_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    print(f"\nReading file: {member.name}")
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read().decode("utf-8")
                        try:
                            data = json.loads(content)
                            print("JSON content:", json.dumps(data, indent=4))
                        except json.JSONDecodeError as e:
                            print("JSON decode error:", e)
                            print("Raw content:", content)
    except Exception as e:
        print(f"Error opening tar file: {e}")


# inspect_tar_file('/mnt/bn/wangziqian-nas/USAM/data/all_data/train/dataset-000.tar')