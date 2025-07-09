import argparse
import pandas as pd
import numpy as np
import os

# Import các thành phần cần thiết
try:
    from model.hmm_model import HMMForecaster
except ImportError as e:
    print(f"LỖI: Không thể import các module cần thiết: {e}")
    exit()

def save_results_to_csv(results_dict, model_name, filename="hmm_evaluation_summary.csv"):
    if not results_dict:
        print("Không có kết quả để lưu vào CSV.")
        return
    records = []
    for feature_set, metrics in results_dict.items():
        row = {'Feature': feature_set}
        row.update(metrics)
        records.append(row)
    df = pd.DataFrame(records)
    df['Model'] = ''
    df.loc[0, 'Model'] = model_name
    metric_cols = ['MAE', 'MAPE', 'sMAPE', 'MPE', 'MSE', 'RMSE', 'RMSPE']
    for col in metric_cols:
        if col not in df.columns: df[col] = np.nan
    final_cols = ['Model', 'Feature'] + metric_cols
    df = df[final_cols]
    try:
        df.to_csv(filename, index=False, float_format='%.6f')
        print(f"\nBảng kết quả đã được lưu thành công vào file: '{filename}'")
    except Exception as e:
        print(f"\nLỖI: Không thể lưu file CSV. Lỗi: {e}")

def run_hmm_training(symbol, timeframe, forecast_horizon, output_dir, csv_path, states_to_test, skip_evaluation=False):
    """
    Hàm chính để huấn luyện và đánh giá các mô hình HMM với số trạng thái khác nhau.
    """
    print(f"--- Bắt đầu quá trình thử nghiệm cho mô hình HMM: {symbol} ---")
    print(f"--- Khung thời gian: {timeframe} | Dự báo cho: {forecast_horizon} phiên tới ---")
    
    # ----------------------------------------------------
    # Bước 1: Tải và chuẩn bị dữ liệu
    # ----------------------------------------------------
    print(f"\nBước 1: Tải dữ liệu từ file CSV: {csv_path}")
    try:
        raw_data = pd.read_csv(csv_path)
        if 'time' in raw_data.columns:
            raw_data['time'] = pd.to_datetime(raw_data['time'])
            raw_data.set_index('time', inplace=True)
        else:
            print("LỖI: File CSV phải có cột 'time'."); return
        features_df = raw_data[['Close', 'Volume']].copy()
        features_df.dropna(inplace=True)
    except Exception as e:
        print(f"LỖI: {e}"); return
    print(f"-> Tải và chuẩn bị thành công {len(features_df)} phiên dữ liệu.")
    
    # ----------------------------------------------------
    # Bước 2: Phân chia Dữ liệu
    # ----------------------------------------------------
    print("\nBước 2: Phân chia Dữ liệu...")
    n = len(features_df)
    train_end_idx = int(n * 0.7)
    val_end_idx = int(n * 0.85)
    train_df = features_df.iloc[:train_end_idx]
    if len(train_df) < 100: print("Lỗi: Dữ liệu không đủ."); return
    print(f"-> Tổng số mẫu: {n} | Tập Huấn luyện: {len(train_df)} mẫu")

    # ----------------------------------------------------
    # Bước 3: Vòng lặp Huấn luyện và Đánh giá (nếu có)
    # ----------------------------------------------------
    all_results = {}

    for n_states in states_to_test:
        print(f"\n{'='*20} Bắt đầu thử nghiệm với {n_states} trạng thái HMM {'='*20}")
        
        # --- Huấn luyện ---
        print(f"Bước 3.1: Huấn luyện Mô hình HMM với {n_states} trạng thái...")
        model = HMMForecaster(n_hidden_states=n_states)
        model.train(train_df)

        # --- Lưu mô hình ---
        print(f"Bước 3.3: Lưu mô hình HMM ({n_states} states)...")
        model.save_model(output_dir, symbol, timeframe)
        

        evaluation_results = model.evaluate_and_plot_hmm(
                output_dir=output_dir,
                all_features_df=features_df,
                train_end_idx=train_end_idx,
                val_end_idx=val_end_idx,
                n_forecast_horizon=forecast_horizon,
            )
            
        if evaluation_results:
            all_results[f"{n_states} states"] = evaluation_results

    print(f"\n\n{'='*25} BẢNG KẾT QUẢ ĐÁNH GIÁ HMM {'='*25}")
    if all_results:
        results_df = pd.DataFrame(all_results).T
        results_df.index.name = "Feature"
        results_df.reset_index(inplace=True)
        for col in ['MAPE', 'sMAPE', 'MPE', 'RMSPE']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}%")
        print(results_df.to_markdown(index=False))
        csv_filename = os.path.join(output_dir, f"results_{symbol}_{timeframe}_hmm.csv")
        save_results_to_csv(all_results, model_name='HMM', filename=csv_filename)
    else:
        print("Không có kết quả nào được ghi nhận.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá các mô hình HMM với số trạng thái khác nhau.")
    
    # ... (các đối số khác giữ nguyên) ...
    parser.add_argument('--symbol', type=str, default='ETHUSD', help='Mã định danh cho mô hình.')
    parser.add_argument('--timeframe', type=str, default='D1', help='Khung thời gian của dữ liệu.')
    parser.add_argument('--horizon', type=int, default=1, help='Số phiên tương lai dự đoán.')
    parser.add_argument('--states', nargs='+', type=int, default=[8,10, 15, 20], help='Danh sách số trạng thái HMM cần thử.')
    parser.add_argument('--output_dir', type=str, default='./models_hmm_eth', help='Thư mục lưu mô hình HMM.')
    parser.add_argument('--csv_path', type=str, default=r'D:\Python_project\LLM_META_LEARNING\HMM_LLM\ETHUSD.csv', help='Đường dẫn đến file CSV.')


    
    args = parser.parse_args()
    
    # <<< THAY ĐỔI: Truyền giá trị của cờ vào hàm chính >>>
    run_hmm_training(args.symbol, args.timeframe, args.horizon, args.output_dir, args.csv_path, args.states)