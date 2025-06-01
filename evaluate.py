import os
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
import torch.cuda
from discrete_speech_metrics import MCD, LogF0RMSE, UTMOS, SpeechBERTScore

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"SoundFile version: {sf.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

def load_audio_pairs(gt_folder, gen_folder):
    """Load audio pairs from ground truth and generated folders"""
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.wav')])
    print(f"Tìm thấy {len(gt_files)} file WAV trong thư mục ground truth")
    
    pairs = []
    skipped_files = []
    
    for gt_file in gt_files:
        gen_path = os.path.join(gen_folder, gt_file)
        gt_path = os.path.join(gt_folder, gt_file)
        
        if not os.path.exists(gen_path):
            skipped_files.append((gt_file, "File không tồn tại trong thư mục generated"))
            continue
            
        try:
            # Đọc audio files
            gt_wav, sr_gt = sf.read(gt_path)
            gen_wav, sr_gen = sf.read(gen_path)
            
            # Kiểm tra sample rate
            if sr_gt != sr_gen:
                skipped_files.append((gt_file, f"Sample rate không khớp: {sr_gt} vs {sr_gen}"))
                continue
                
            # Kiểm tra độ dài audio
            if len(gt_wav) == 0 or len(gen_wav) == 0:
                skipped_files.append((gt_file, "File audio rỗng"))
                continue
            
            pairs.append((gt_wav, gen_wav, gt_file))
            
        except Exception as e:
            skipped_files.append((gt_file, f"Lỗi khi đọc file: {str(e)}"))
            continue
    
    print(f"Đã load thành công {len(pairs)} cặp audio")
    if skipped_files:
        print("\nCác file bị bỏ qua:")
        for file, reason in skipped_files:
            print(f"- {file}: {reason}")
    
    return pairs

def evaluate_folder(gt_folder, gen_folder):
    """Đánh giá toàn bộ audio trong folders"""
    print("Loading audio pairs...")
    audio_pairs = load_audio_pairs(gt_folder, gen_folder)
    
    if not audio_pairs:
        print("Không tìm thấy cặp audio nào để so sánh!")
        return None, None
    
    print(f"Tìm thấy {len(audio_pairs)} cặp audio để đánh giá")
    
    # Khởi tạo metrics
    print("Khởi tạo các metrics...")
    try:
        print("Khởi tạo MCD...")
        mcd = MCD(sr=16000)
        
        print("Khởi tạo LogF0RMSE...")
        f0_rmse = LogF0RMSE(sr=16000)
        
        print("Khởi tạo UTMOS...")
        utmos = UTMOS(sr=16000)
        
        print("Khởi tạo SpeechBERTScore...")
        bertscore = SpeechBERTScore(sr=16000, model_type="wavlm-base", layer=12, use_gpu=torch.cuda.is_available())
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo metrics: {str(e)}")
        return None, None
    
    results = {
        'mcd': [],
        'f0_rmse': [],
        'utmos': [],
        'filenames': [],
        'bert_precision': [],
        'bert_recall': [],
        'bert_f1': []
    }
    
    # Đánh giá từng cặp
    print("Bắt đầu đánh giá...")
    for gt_wav, gen_wav, filename in tqdm(audio_pairs):
        try:
            # Tính các metrics
            mcd_score = mcd.score(gt_wav, gen_wav)
            f0_score = f0_rmse.score(gt_wav, gen_wav)
            mos_score = utmos.score(gen_wav)
            prec, rec, f1 = bertscore.score(gt_wav, gen_wav)
            
            # Lưu kết quả
            results['mcd'].append(mcd_score)
            results['f0_rmse'].append(f0_score)
            results['utmos'].append(mos_score)
            results['filenames'].append(filename)
            results['bert_precision'].append(prec)
            results['bert_recall'].append(rec)
            results['bert_f1'].append(f1)
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {str(e)}")
    
    # Tính các thống kê
    stats = {}
    for key in results.keys():
        if key != 'filenames':
            values = np.array(results[key])
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Lưu kết quả chi tiết
    save_results(results, stats)
    
    return results, stats

def save_results(results, stats):
    """Lưu kết quả ra file"""
    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Lưu kết quả chi tiết
    with open('evaluation_results/detailed_results.txt', 'w', encoding='utf-8') as f:
        f.write("Kết quả đánh giá chi tiết:\n\n")
        for i, filename in enumerate(results['filenames']):
            f.write(f"File: {filename}\n")
            for metric in results.keys():
                if metric != 'filenames':
                    f.write(f"{metric}: {results[metric][i]:.4f}\n")
            f.write("\n")
    
    # Lưu thống kê tổng hợp
    with open('evaluation_results/summary_stats.txt', 'w', encoding='utf-8') as f:
        f.write("Thống kê tổng hợp:\n\n")
        for metric, values in stats.items():
            f.write(f"\n{metric}:\n")
            for stat_name, stat_value in values.items():
                f.write(f"  {stat_name}: {stat_value:.4f}\n")

if __name__ == "__main__":
    # Đường dẫn tới các folder
    gt_folder = "ground_truth_audio"
    gen_folder = "result/500000"
    
    print(f"Bắt đầu đánh giá...")
    print(f"Ground truth folder: {gt_folder}")
    print(f"Generated audio folder: {gen_folder}")
    
    results, stats = evaluate_folder(gt_folder, gen_folder)
    
    if results and stats:
        print("\nĐánh giá hoàn tất!")
        print("Kết quả đã được lưu trong thư mục 'evaluation_results'")
    else:
        print("\nĐánh giá không thành công!")