import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
# <<< THAY ĐỔI: Import thêm EarlyStopping >>>
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import matplotlib.pyplot as plt
from eval_metric import calculate_all_metrics
class HybridModel:
    def __init__(self, n_steps_in=60, n_hidden_states=8, epochs=200):
        self.n_steps_in = n_steps_in
        self.n_hidden_states = n_hidden_states
        self.n_features = None
        self.hmm = None
        self.lstm_model = None
        self.scaler = None
        self.epochs = epochs

    def _create_multi_output_dataset(self, dataset, n_steps_out):
        """
        Tạo dataset với input là một chuỗi và output là một chuỗi.
        Ví dụ: Input là 60 ngày, Output là giá Close của 10 ngày tiếp theo.
        """
        X, y = [], []
        for i in range(len(dataset)):
            # Tìm điểm kết thúc của chuỗi output
            end_ix = i + self.n_steps_in + n_steps_out -1
            # Kiểm tra xem có vượt ra ngoài dataset không
            if end_ix > len(dataset)-1:
                break
            # Lấy dữ liệu input (X) và output (y)
            seq_x = dataset[i : i + self.n_steps_in, :]
            seq_y = dataset[i + self.n_steps_in : i + self.n_steps_in + n_steps_out, 0] # Chỉ lấy giá Close
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train(self, train_features_df, val_features_df, n_forecast_horizon):
        """
        Huấn luyện mô hình Multi-Output.
        n_forecast_horizon: số phiên tương lai cần dự đoán (ví dụ: 10).
        """
        print("Huấn luyện HMM và fit Scaler chỉ trên tập Train...")
        self.hmm = GaussianHMM(n_components=self.n_hidden_states, covariance_type="full", n_iter=100)
        hmm_train_features = train_features_df[['close', 'volume']].values
        self.hmm.fit(hmm_train_features)
        
        train_hidden_states = self.hmm.predict(hmm_train_features)
        val_hidden_states = self.hmm.predict(val_features_df[['close', 'volume']].values)

        train_input_df = train_features_df.copy().iloc[-len(train_hidden_states):]
        train_input_df['HMM_State'] = train_hidden_states
        val_input_df = val_features_df.copy().iloc[-len(val_hidden_states):]
        val_input_df['HMM_State'] = val_hidden_states

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = self.scaler.fit_transform(train_input_df)
        scaled_val_data = self.scaler.transform(val_input_df)
        self.n_features = scaled_train_data.shape[1]

        # --- Chuẩn bị dữ liệu Multi-Output ---
        X_train, y_train = self._create_multi_output_dataset(scaled_train_data, n_forecast_horizon)
        X_val, y_val = self._create_multi_output_dataset(scaled_val_data, n_forecast_horizon)
        
        # --- Xây dựng và Huấn luyện LSTM Multi-Output ---
        print("Bắt đầu huấn luyện LSTM Multi-Output với Early Stopping...")
        self.lstm_model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=(self.n_steps_in, self.n_features)),
            Dropout(0.2),
            LSTM(units=50), 
            Dropout(0.2),
            Dense(units=n_forecast_horizon) 
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mean_absolute_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        
        self.lstm_model.fit(X_train, y_train, 
                            epochs=self.epochs, 
                            batch_size=100,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=1)

    def forecast(self, features_df):
        """
        Dự báo một chuỗi tương lai bằng mô hình Multi-Output.
        Độ dài của chuỗi dự báo được quyết định bởi kiến trúc của mô hình đã được huấn luyện.
        """
        if self.lstm_model is None:
            raise Exception("Mô hình chưa được tải hoặc huấn luyện.")
            
        # Lấy số lượng nơ-ron output để biết mô hình dự báo được bao nhiêu phiên
        n_forecast_horizon = self.lstm_model.output_shape[1]

        # Lấy dữ liệu đầu vào cuối cùng
        last_n_steps_data = features_df.tail(self.n_steps_in)
        
        # Gán nhãn HMM
        hmm_features = last_n_steps_data[['close', 'volume']].values
        hidden_states = self.hmm.predict(hmm_features)
        
        last_n_steps_with_hmm = last_n_steps_data.copy()
        last_n_steps_with_hmm['HMM_State'] = hidden_states

        # Scale dữ liệu đầu vào
        scaled_input = self.scaler.transform(last_n_steps_with_hmm)
        input_batch = scaled_input.reshape(1, self.n_steps_in, self.n_features)

        # Dự đoán ra cả chuỗi output
        predicted_scaled_sequence = self.lstm_model.predict(input_batch, verbose=0)[0]
        
        # Kiểm tra lại độ dài có khớp không
        if len(predicted_scaled_sequence) != n_forecast_horizon:
             raise ValueError("Độ dài chuỗi dự đoán từ mô hình không khớp với kiến trúc output.")

        # Inverse transform kết quả
        dummy_array = np.zeros((n_forecast_horizon, self.n_features))
        dummy_array[:, 0] = predicted_scaled_sequence
        unscaled_forecast = self.scaler.inverse_transform(dummy_array)
        
        return unscaled_forecast[:, 0]

    def save_model(self, model_dir, symbol, timeframe):
        """Lưu các thành phần của mô hình vào thư mục."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        joblib.dump(self.hmm, os.path.join(model_dir, f'{symbol}_{timeframe}_hmm.pkl'))
        self.lstm_model.save(os.path.join(model_dir, f'{symbol}_{timeframe}_lstm.h5'))
        joblib.dump(self.scaler, os.path.join(model_dir, f'{symbol}_{timeframe}_scaler.pkl'))
        print(f"Mô hình đã được lưu vào thư mục '{model_dir}'")

    def load_model(self, model_dir, symbol, timeframe):
        """Tải các thành phần của mô hình từ thư mục."""
        hmm_path = os.path.join(model_dir, f'{symbol}_{timeframe}_hmm.pkl')
        lstm_path = os.path.join(model_dir, f'{symbol}_{timeframe}_lstm.h5')
        scaler_path = os.path.join(model_dir, f'{symbol}_{timeframe}_scaler.pkl')

        if not all([os.path.exists(p) for p in [hmm_path, lstm_path, scaler_path]]):
            raise FileNotFoundError("Không tìm thấy file mô hình. Vui lòng chạy `train.py` trước.")

        self.hmm = joblib.load(hmm_path)
        self.lstm_model = load_model(lstm_path)
        self.scaler = joblib.load(scaler_path)
        
        # Lấy lại n_features từ scaler đã tải
        self.n_features = self.scaler.n_features_in_
        print("Mô hình đã được tải thành công.")




    def evaluate_and_plot_hmm_lstm(self, output_dir, all_features_df, train_end_idx, val_end_idx, 
                          feature_set_name="Unknown_Features"): # Loại bỏ n_forecast_horizon
        """
        Đánh giá mô hình, tự động suy ra horizon từ kiến trúc model.
        """
        print(f"\n--- Bắt đầu Đánh giá cho bộ đặc trưng: '{feature_set_name}' ---")
        
        try:
            # Lấy số nơ-ron ở lớp output của mô hình LSTM
            n_forecast_horizon = self.lstm_model.output_shape[1]
            print(f"-> Horizon được suy ra từ mô hình: {n_forecast_horizon} phiên")
        except:
            print("LỖI: Không thể suy ra horizon từ kiến trúc mô hình. Mặc định là 1.")
            n_forecast_horizon = 1
        
        # --- Phần Walk-Forward Validation (giữ nguyên) ---
        all_predictions = []
        all_dates = []
        current_idx = val_end_idx 
        while current_idx < len(all_features_df):
            start_input_idx = current_idx - self.n_steps_in
            if start_input_idx < 0:
                break
            
            input_features = all_features_df.iloc[start_input_idx : current_idx]
            predicted_sequence = self.forecast(input_features)
            
            end_pred_idx = current_idx + len(predicted_sequence)
            actual_dates_for_preds = all_features_df.index[current_idx:end_pred_idx]
            
            predicted_sequence = predicted_sequence[:len(actual_dates_for_preds)]

            all_predictions.extend(predicted_sequence)
            all_dates.extend(actual_dates_for_preds)
            
            if len(predicted_sequence) > 0:
                current_idx += len(predicted_sequence)
            else:
                break


        if not all_dates:
            print("!!! CẢNH BÁO: Không có dự báo nào được tạo ra. Bỏ qua đánh giá.")
            return None

        prediction_series = pd.Series(all_predictions, index=all_dates)
        comparison_df = pd.DataFrame({'Actual': all_features_df['close'], 'Predicted': prediction_series})
        comparison_df.dropna(inplace=True)
        
        if comparison_df.empty:
            print("!!! CẢNH BÁO: Không có điểm dữ liệu chung. Bỏ qua đánh giá.")
            return None
            
        evaluation_results = calculate_all_metrics(comparison_df['Actual'], comparison_df['Predicted'])
    
        print(f"\n--- Kết quả Đánh giá cho '{feature_set_name}' ---")
        for key, value in evaluation_results.items():
            unit = '%' if key in ['MAPE', 'sMAPE', 'MPE', 'RMSPE'] else ''
            print(f"{key:<6}: {value:.4f}{unit}")
        print("---------------------------------------------------")
    
        
        # --- Phần Vẽ biểu đồ (được cập nhật) ---
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(20, 10))
        
        ax.plot(all_features_df.index, all_features_df['close'], label='Giá Thực tế (Toàn bộ)', color='royalblue', linewidth=1.5)
        ax.plot(prediction_series.index, prediction_series.values, label=f'Dự đoán (Bộ: {feature_set_name})', color='darkorange', linewidth=2, linestyle='--')
        
        ax.axvspan(all_features_df.index[0], all_features_df.index[train_end_idx], color='green', alpha=0.1, label='Tập Huấn luyện')
        ax.axvspan(all_features_df.index[train_end_idx], all_features_df.index[val_end_idx], color='yellow', alpha=0.1, label='Tập Thẩm định')
        ax.axvspan(all_features_df.index[val_end_idx], all_features_df.index[-1], color='red', alpha=0.1, label='Tập Kiểm tra')

        ax.set_title(f'So sánh Dự đoán và Thực tế - Bộ đặc trưng: {feature_set_name}', fontsize=18)
        
        ax.set_xlabel('Ngày', fontsize=14)
        ax.set_ylabel('Giá Đóng cửa', fontsize=14)
        ax.legend(fontsize=12)
        plt.grid(True)
        
        # Làm sạch tên bộ đặc trưng để dùng làm tên file
        safe_filename = feature_set_name.replace('+', '_').replace(' ', '').lower()
        plot_filename = f'evaluation_plot_{safe_filename}.png'
        
        # Lưu vào một thư mục con để gọn gàng
        plot_dir = os.path.join(output_dir, 'evaluation_plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        full_plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(full_plot_path)
        print(f"Biểu đồ đánh giá đã được lưu vào file '{full_plot_path}'")
        
        # Đóng figure để giải phóng bộ nhớ, quan trọng khi chạy trong vòng lặp
        plt.close(fig) 
        
        # Trả về kết quả để script chính có thể tổng hợp
        return evaluation_results