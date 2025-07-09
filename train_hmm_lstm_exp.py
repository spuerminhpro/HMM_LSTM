import argparse
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import re

# Import các thành phần cần thiết
try:
    from model.hmm_lstm_model import HybridModel
except ImportError as e:
    print(f"LỖI: Không thể import các module cần thiết: {e}")
    exit()

def find_optimal_hmm_states(data):
    """Tìm số trạng thái HMM tối ưu bằng BIC."""
    lowest_bic = np.inf
    best_n_components = 0
    n_components_range = range(2, 21)
    print("-> Đang tìm số trạng thái HMM tối ưu bằng BIC (từ 2 đến 20)...")
    
    if len(data) > 10000:
        data_sample = data[np.random.choice(data.shape[0], 10000, replace=False), :]
    else:
        data_sample = data
        
    for n_components in n_components_range:
        try:
            hmm = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
            hmm.fit(data_sample)
            bic = hmm.bic(data_sample)
            if bic < lowest_bic:
                lowest_bic = bic
                best_n_components = n_components
        except ValueError:
            continue
            
    if best_n_components == 0:
        print("-> Không tìm thấy trạng thái tối ưu, mặc định là 8.")
        return 8
        
    print(f"-> Số trạng thái HMM tối ưu được tìm thấy là: {best_n_components}")
    return best_n_components

def clean_col_names(df):
    """Chuẩn hóa tên cột: chữ thường, thay thế ký tự đặc biệt bằng gạch dưới."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.lower()
        new_col = re.sub(r'[^a-z0-9]+', '_', new_col).strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def run_hybrid_experiments(symbol, timeframe, forecast_horizon, output_dir, epochs, csv_path):
    """
    Hàm chính để chạy thử nghiệm trên mô hình Hybrid từ file CSV đã có sẵn đặc trưng.
    """
    print(f"--- Bắt đầu chuỗi thử nghiệm cho mô hình Hybrid: {symbol} ---")
    
    # ----------------------------------------------------
    # Bước 1: Tải dữ liệu đã có đặc trưng
    # ----------------------------------------------------
    print(f"\nBước 1: Tải dữ liệu đã có đặc trưng từ CSV: {csv_path}")
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
        print("Không có dữ liệu hợp lệ sau khi làm sạch."); return
    print(f"-> Tải và làm sạch thành công {len(all_features_df.columns)} đặc trưng cho {len(all_features_df)} phiên.")
    
    # ----------------------------------------------------
    # Bước 2: Phân chia dữ liệu & Tìm trạng thái HMM tối ưu
    # ----------------------------------------------------
    print("\nBước 2: Phân chia dữ liệu và tìm trạng thái HMM tối ưu...")
    n = len(all_features_df)
    train_end_idx = int(n * 0.7)
    val_end_idx = int(n * 0.85)
    train_df_full_features = all_features_df.iloc[:train_end_idx]
    
    try:
        hmm_fit_data = train_df_full_features[['close', 'volume']].values
    except KeyError:
        print("LỖI: Dữ liệu phải có cột 'close' và 'volume'."); return
    
    optimal_states = find_optimal_hmm_states(hmm_fit_data)
    
    # ----------------------------------------------------
    # Bước 3: Định nghĩa các bộ đặc trưng và chạy thử nghiệm
    # ----------------------------------------------------
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    
    #BTC a=5 b=50
    all_cols = all_features_df.columns.tolist()
    ma_cols = [col for col in all_cols if 'ma_' in col and 'ema' not in col]
    ema_cols = [col for col in all_cols if 'ema_' in col]
    std_cols = [col for col in all_cols if 'std_' in col]
    macd_cols = [col for col in all_cols if 'macd0' in col]
    bb_cols = [col for col in all_cols if 'bollinger' in col]

    print("\nCác nhóm đặc trưng được phát hiện:")
    print(f"MA cols: {ma_cols}")
    
    feature_sets = {
        "MA": ma_cols, "EMA": ema_cols, "STD": std_cols,
        "MACD": macd_cols, "BB": bb_cols,
        "STD+EMA+MACD+BB": std_cols + ema_cols + macd_cols + bb_cols,
        "STD+EMA+MACD+BB+MA": std_cols + ema_cols + macd_cols + bb_cols + ma_cols,
        "MA+STD+EMA+MACD": ma_cols + std_cols + ema_cols + macd_cols
    }

    all_results = {}

    for set_name, feature_list in feature_sets.items():
        print(f"\n\n{'='*20} Bắt đầu thử nghiệm với bộ đặc trưng: {set_name} {'='*20}")
        
        try:
            current_feature_cols = base_cols + feature_list
            features_subset_df = all_features_df[current_feature_cols]
        except KeyError as e:
            print(f"Bỏ qua bộ '{set_name}' do không tìm thấy cột đặc trưng: {e}"); continue

        train_df = features_subset_df.iloc[:train_end_idx]
        val_df = features_subset_df.iloc[train_end_idx:val_end_idx]
        
        if len(train_df) < 60 or len(val_df) < 60:
            print(f"Bỏ qua bộ '{set_name}' do không đủ dữ liệu."); continue
        
        # --- HUẤN LUYỆN ---
        print(f"Huấn luyện mô hình Hybrid với {len(current_feature_cols)} đặc trưng...")
        model = HybridModel(n_steps_in=60, n_hidden_states=optimal_states, epochs=epochs)
        model.train(train_df, val_df, forecast_horizon)
        
        # --- LƯU MÔ HÌNH ---
        model_name_suffix = set_name.replace('+', '_').lower()
        model.save_model(output_dir, f"{symbol}_{model_name_suffix}", timeframe)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chạy thử nghiệm trên mô hình Hybrid từ file CSV đã có sẵn đặc trưng.")
    
    parser.add_argument('--symbol', type=str, default='AAPL', help='Mã định danh cho mô hình.')
    parser.add_argument('--timeframe', type=str, default='D1', help='Khung thời gian của dữ liệu.')
    parser.add_argument('--horizon', type=int, default=1, help='Số phiên tương lai dự đoán.')
    parser.add_argument('--epochs', type=int, default=100, help='Số epoch tối đa.')
    parser.add_argument('--output_dir', type=str, default='./models_hybrid_aapl', help='Thư mục lưu các mô hình.')
    parser.add_argument('--csv_path', type=str, default=r'D:\Python_project\LLM_META_LEARNING\HMM_LLM\AAPL.csv', help='Đường dẫn đến file CSV ĐÃ CÓ ĐẶC TRƯNG.')

    
    args = parser.parse_args()
    
    run_hybrid_experiments(args.symbol, args.timeframe, args.horizon, args.output_dir, args.epochs, args.csv_path)