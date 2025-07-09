import argparse
import pandas as pd
import numpy as np

# Import các thành phần cần thiết
try:
    from model.lstm_model import LSTMModel
except ImportError as e:
    print(f"LỖI: Không thể import các module cần thiết: {e}")
    exit()

def run_lstm_training(symbol, timeframe, forecast_horizon, output_dir, epochs, csv_path):
    """
    Hàm chính để thực hiện quá trình huấn luyện mô hình LSTM độc lập từ file CSV.
    """
    print(f"--- Bắt đầu quá trình huấn luyện cho mô hình LSTM: {symbol} ---")
    print(f"--- Khung thời gian: {timeframe} | Dự báo cho: {forecast_horizon} phiên tới ---")
    
    # ----------------------------------------------------
    # Bước 1: Tải dữ liệu thô từ CSV
    # ----------------------------------------------------
    print(f"\nBước 1: Tải dữ liệu từ file CSV: {csv_path}")
    try:
        raw_data = pd.read_csv(csv_path)
        
        if 'time' in raw_data.columns:
            raw_data['time'] = pd.to_datetime(raw_data['time'])
            raw_data.set_index('time', inplace=True)
        else:
            print("LỖI: File CSV phải có cột 'time'.")
            return
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file tại đường dẫn: {csv_path}")
        return
    except Exception as e:
        print(f"LỖI: Có lỗi xảy ra khi đọc file CSV: {e}")
        return

    if raw_data.empty:
        print("Dữ liệu trong file CSV trống. Huấn luyện bị hủy.")
        return
    print(f"-> Tải thành công {len(raw_data)} phiên dữ liệu thô.")

    features_df = raw_data[['Close']]
    # ----------------------------------------------------
    # Bước 3: Phân chia Dữ liệu
    # ----------------------------------------------------
    print("\nBước 3: Phân chia Dữ liệu...")
    n = len(features_df)
    train_end_idx = int(n * 0.7)
    val_end_idx = int(n * 0.85)
    
    train_df = features_df.iloc[:train_end_idx]
    val_df = features_df.iloc[train_end_idx:val_end_idx]
    test_df = features_df.iloc[val_end_idx:]
    
    if len(train_df) < 60 or len(val_df) < 60 or len(test_df) < forecast_horizon:
        print("Lỗi: Dữ liệu không đủ để chia thành các tập Train/Validation/Test hợp lệ.")
        print(f"Cần ít nhất ~{int((60+forecast_horizon)/0.15)} phiên dữ liệu sau khi trích xuất đặc trưng.")
        return
        
    print(f"-> Tổng số mẫu: {n}")
    print(f"-> Tập Huấn luyện: {len(train_df)} mẫu")
    print(f"-> Tập Thẩm định: {len(val_df)} mẫu")
    print(f"-> Tập Kiểm tra:  {len(test_df)} mẫu")

    # ----------------------------------------------------
    # Bước 4: Huấn luyện Mô hình LSTM
    # ----------------------------------------------------
    print("\nBước 4: Chuẩn bị và Huấn luyện Mô hình LSTM...")
    
    model = LSTMModel(n_steps_in=60, epochs=epochs)
    
    print(f"\n-> Bắt đầu huấn luyện mô hình LSTM để dự đoán {forecast_horizon} phiên tới...")
    model.train(train_df, val_df, forecast_horizon)
    
    model.evaluate_and_plot_lstm(output_dir, features_df, train_end_idx, val_end_idx, forecast_horizon)
    
    #Lưu mô hình
    print("\nBước 6: Lưu mô hình LSTM tốt nhất...")
    model.save_model(output_dir, symbol, timeframe)
    
    print(f"\n--- Quá trình huấn luyện LSTM cho {symbol} đã hoàn tất! ---")
    print(f"--- Mô hình đã được lưu vào '{output_dir}' và sẵn sàng để sử dụng. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình LSTM Multi-Output từ file CSV.")
    
    parser.add_argument('--symbol', 
                        type=str, 
                        default='AAPL', 
                        help='Mã định danh cho mô hình (ví dụ: BTCUSD, GOLD, VNM).')
                        
    parser.add_argument('--timeframe', 
                        type=str, 
                        default='D1', 
                        choices=['M1', 'M5', 'M15', 'M30', 'H1', 'D1', 'W1', 'MN1'],
                        help='Khung thời gian của dữ liệu (dùng để đặt tên model).')
                        
    parser.add_argument('--horizon', 
                        type=int, 
                        default=1, 
                        help='Số phiên tương lai mà mô hình được huấn luyện để dự đoán.')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Số epoch tối đa để huấn luyện mô hình.')
                        
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='./models_lstm_aapl',
                        help='Thư mục để lưu trữ mô hình LSTM đã huấn luyện.')
                        
    parser.add_argument('--csv_path', 
                        type=str, 
                        default=r'D:\Python_project\LLM_META_LEARNING\HMM_LLM\AAPL.csv', 
                        help='Đường dẫn BẮT BUỘC đến file CSV chứa dữ liệu OHLCV thô.')
    
    args = parser.parse_args()
    
    run_lstm_training(args.symbol, args.timeframe, args.horizon, args.output_dir, args.epochs, args.csv_path)