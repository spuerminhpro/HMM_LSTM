import argparse
import pandas as pd
import numpy as np
import os
import re

# Import các thành phần cần thiết
try:
    # Cần lớp HybridModel để có thể tải mô hình vào
    from model.hmm_lstm_model import HybridModel 
    # Mặc dù không gọi trực tiếp, hàm evaluate_and_plot bên trong model sẽ cần nó
except ImportError as e:
    print(f"LỖI: Không thể import các module cần thiết: {e}")
    print("Hãy đảm bảo các file 'model.py' và 'evaluate.py' tồn tại trong cùng thư mục.")
    exit()

def clean_col_names(df):
    """Chuẩn hóa tên cột để nhất quán với lúc huấn luyện."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.lower()
        new_col = re.sub(r'[^a-z0-9]+', '_', new_col).strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df



# Thêm hàm này vào đầu file test_hybrid_models.py

def save_results_to_csv(results_dict, model_name, filename="evaluation_summary.csv"):
    """
    Lưu một từ điển kết quả vào file CSV với định dạng cụ thể.

    Args:
        results_dict (dict): Dictionary chứa kết quả, ví dụ: {'All_MA': {'MAE': 0.1, ...}, ...}
        model_name (str): Tên của mô hình chính, ví dụ: 'HMM & LSTM'.
        filename (str): Tên file CSV để lưu.
    """
    if not results_dict:
        print("Không có kết quả để lưu vào CSV.")
        return

    # Chuyển đổi dictionary thành list các bản ghi để dễ tạo DataFrame
    records = []
    for feature_set, metrics in results_dict.items():
        # Tạo một dictionary mới cho mỗi hàng
        row = {'Feature': feature_set}
        row.update(metrics)
        records.append(row)

    # Tạo DataFrame
    df = pd.DataFrame(records)

    # Thêm cột 'Model' với hiệu ứng "merged cell"
    df['Model'] = ''
    df.loc[0, 'Model'] = model_name

    # Sắp xếp lại các cột theo đúng thứ tự mong muốn
    metric_cols = ['MAE', 'MAPE', 'sMAPE', 'MPE', 'MSE', 'RMSE', 'RMSPE']
    # Đảm bảo tất cả các cột metric đều tồn tại, nếu không thì tạo cột NaN
    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    final_cols = ['Model', 'Feature'] + metric_cols
    df = df[final_cols]

    # Lưu ra file CSV
    try:
        # float_format='%.6f' để lưu số thực với 6 chữ số thập phân
        df.to_csv(filename, index=False, float_format='%.6f')
        print(f"\nBảng kết quả đã được lưu thành công vào file: '{filename}'")
    except Exception as e:
        print(f"\nLỖI: Không thể lưu file CSV. Lỗi: {e}")

def run_evaluation(symbol, timeframe, output_dir, model_dir, csv_path):
    """
    Hàm chính để tải và đánh giá một loạt các mô hình Hybrid đã được huấn luyện.
    """
    print(f"--- Bắt đầu quá trình đánh giá các mô hình Hybrid cho: {symbol} ---")
    
    # ----------------------------------------------------
    # Bước 1: Tải dữ liệu đã có đặc trưng
    # ----------------------------------------------------
    print(f"\nBước 1: Tải dữ liệu từ CSV: {csv_path}")
    try:
        all_features_df = pd.read_csv(csv_path)
        all_features_df = clean_col_names(all_features_df)
        
        if 'time' in all_features_df.columns:
            all_features_df['time'] = pd.to_datetime(all_features_df['time'])
            all_features_df.set_index('time', inplace=True)
        else:
            print("LỖI: File CSV phải có cột 'time'."); return
            
        all_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        all_features_df.dropna(inplace=True)
        
    except Exception as e:
        print(f"LỖI: {e}"); return

    if all_features_df.empty:
        print("Không có dữ liệu hợp lệ."); return
    print(f"-> Tải thành công {len(all_features_df.columns)} đặc trưng.")
    
    # ----------------------------------------------------
    # Bước 2: Định nghĩa các bộ đặc trưng cần đánh giá
    # **QUAN TRỌNG**: Dictionary này phải giống hệt với file training.
    # ----------------------------------------------------
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    
    all_cols = all_features_df.columns.tolist()
    ma_cols = [col for col in all_cols if 'ma_' in col and 'ema' not in col]
    ema_cols = [col for col in all_cols if 'ema_' in col]
    std_cols = [col for col in all_cols if 'std_' in col]
    macd_cols = [col for col in all_cols if 'macd0' in col]
    bb_cols = [col for col in all_cols if 'bollinger' in col]

    feature_sets = {
        "MA": ma_cols, "EMA": ema_cols, "STD": std_cols,
        "MACD": macd_cols, "BB": bb_cols,
        "STD+EMA+MACD+BB": std_cols + ema_cols + macd_cols + bb_cols,
        "STD+EMA+MACD+BB+MA": std_cols + ema_cols + macd_cols + bb_cols + ma_cols,
        "MA+STD+EMA+MACD": ma_cols + std_cols + ema_cols + macd_cols
    }

    all_results = {}
    
    # Tính toán các chỉ số phân chia dữ liệu một lần
    n = len(all_features_df)
    train_end_idx = int(n * 0.7)
    val_end_idx = int(n * 0.85)

    # ----------------------------------------------------
    # Bước 3: Vòng lặp Tải và Đánh giá từng mô hình
    # ----------------------------------------------------
    for set_name, feature_list in feature_sets.items():
        print(f"\n\n{'='*20} Đánh giá mô hình cho bộ đặc trưng: {set_name} {'='*20}")
        
        # 1. Tải mô hình
        try:
            model = HybridModel() # Tạo một instance rỗng
            model_name_suffix = set_name.replace('+', '_').lower()
            
            print(f"-> Đang tải mô hình từ thư mục: {model_dir}")
            model.load_model(model_dir, f"{symbol}_{model_name_suffix}", timeframe)
        
        except FileNotFoundError:
            print(f"!!! LỖI: Không tìm thấy mô hình cho bộ '{set_name}'. Bỏ qua.")
            continue
        except Exception as e:
            print(f"!!! LỖI: Không thể tải mô hình cho bộ '{set_name}': {e}. Bỏ qua.")
            continue

        # 2. Chuẩn bị dữ liệu con cho mô hình này
        try:
            current_feature_cols = base_cols + feature_list
            features_subset_df = all_features_df[current_feature_cols]
        except KeyError as e:
            print(f"!!! LỖI: Dữ liệu CSV thiếu cột cần thiết cho bộ '{set_name}': {e}. Bỏ qua.")
            continue

        # 3. Chạy đánh giá
        print(f"-> Chạy đánh giá walk-forward...")
        evaluation_results = model.evaluate_and_plot_hmm_lstm(
            output_dir=output_dir,
            all_features_df=features_subset_df,
            train_end_idx=train_end_idx,
            val_end_idx=val_end_idx, 
            feature_set_name=set_name
        )
        
        if evaluation_results:
            all_results[set_name] = evaluation_results

    # ----------------------------------------------------
    # Bước 4: In Bảng Tổng kết
    # ----------------------------------------------------
    print(f"\n\n{'='*25} BẢNG KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP {'='*25}")
    if all_results:
        results_df = pd.DataFrame(all_results).T # Chuyển vị để hàng là tên model
        # Định dạng lại các cột phần trăm
        for col in ['MAPE', 'sMAPE', 'MPE', 'RMSPE']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.4f}%")
        print(results_df.to_markdown())
        csv_filename = os.path.join(output_dir, f"results_{symbol}_{timeframe}_hmm_lstm.csv")
        save_results_to_csv(all_results, model_name='HMM & LSTM', filename=csv_filename)
    else:
        print("Không có kết quả nào được ghi nhận. Hãy kiểm tra lại đường dẫn và tên file.")
    print('='*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tải và đánh giá các mô hình Hybrid HMM-LSTM đã được huấn luyện.")
    
    parser.add_argument('--symbol', type=str, default='AAPL', help='Mã định danh của mô hình (ví dụ: BTCUSD).')
    parser.add_argument('--timeframe', type=str, default='D1', help='Khung thời gian của dữ liệu.')
    parser.add_argument('--output_dir', type=str, default='./models_hybrid_aapl', help='Thư mục lưu các mô hình.')
    parser.add_argument('--model_dir', type=str, default='./models_hybrid_aapl', help='Thư mục chứa các mô hình đã lưu.')
    parser.add_argument('--csv_path', type=str, default=r'D:\Python_project\LLM_META_LEARNING\HMM_LLM\AAPL.csv', help='Đường dẫn đến file CSV chứa đầy đủ đặc trưng.')
    
    args = parser.parse_args()
    
    run_evaluation(args.symbol, args.timeframe, args.output_dir, args.model_dir, args.csv_path)