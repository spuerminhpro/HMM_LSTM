import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from eval_metric import calculate_all_metrics
class LSTMModel:
    def __init__(self, n_steps_in=60, epochs=200):
        self.n_steps_in = n_steps_in
        self.n_features = None
        self.lstm_model = None
        self.scaler = None
        self.epochs = epochs

    def _create_multi_output_dataset(self, dataset, n_steps_out):
        X, y = [], []
        for i in range(len(dataset)):
            end_ix = i + self.n_steps_in + n_steps_out -1
            if end_ix > len(dataset)-1:
                break
            seq_x = dataset[i : i + self.n_steps_in, :]
            seq_y = dataset[i + self.n_steps_in : i + self.n_steps_in + n_steps_out, 0] # Chỉ lấy giá Close
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train(self, train_features_df, val_features_df, n_forecast_horizon):
        """Huấn luyện mô hình LSTM Multi-Output (không có HMM)."""
        print("Fit Scaler chỉ trên tập Train...")
        
        # <<< THAY ĐỔI: Không còn logic HMM >>>
        train_input_df = train_features_df.copy()
        val_input_df = val_features_df.copy()

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = self.scaler.fit_transform(train_input_df)
        scaled_val_data = self.scaler.transform(val_input_df)
        self.n_features = scaled_train_data.shape[1]

        X_train, y_train = self._create_multi_output_dataset(scaled_train_data, n_forecast_horizon)
        X_val, y_val = self._create_multi_output_dataset(scaled_val_data, n_forecast_horizon)
        
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
        """Dự báo một chuỗi tương lai."""
        if self.lstm_model is None:
            raise Exception("Mô hình chưa được tải hoặc huấn luyện.")
            
        n_forecast_horizon = self.lstm_model.output_shape[1]
        last_n_steps_data = features_df.tail(self.n_steps_in)
        
        # <<< THAY ĐỔI: Không còn logic HMM >>>
        scaled_input = self.scaler.transform(last_n_steps_data)
        input_batch = scaled_input.reshape(1, self.n_steps_in, self.n_features)

        predicted_scaled_sequence = self.lstm_model.predict(input_batch, verbose=0)[0]
        
        dummy_array = np.zeros((n_forecast_horizon, self.n_features))
        dummy_array[:, 0] = predicted_scaled_sequence
        unscaled_forecast = self.scaler.inverse_transform(dummy_array)
        
        return unscaled_forecast[:, 0]

    def save_model(self, model_dir, symbol, timeframe):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.lstm_model.save(os.path.join(model_dir, f'{symbol}_{timeframe}_lstm_only.h5'))
        joblib.dump(self.scaler, os.path.join(model_dir, f'{symbol}_{timeframe}_scaler_only.pkl'))
        print(f"Mô hình LSTM đã được lưu vào thư mục '{model_dir}'")

    def load_model(self, model_dir, symbol, timeframe):
        # <<< THAY ĐỔI: Chỉ tải LSTM và Scaler >>>
        lstm_path = os.path.join(model_dir, f'{symbol}_{timeframe}_lstm_only.h5')
        scaler_path = os.path.join(model_dir, f'{symbol}_{timeframe}_scaler_only.pkl')

        if not all([os.path.exists(p) for p in [lstm_path, scaler_path]]):
            raise FileNotFoundError("Không tìm thấy file mô hình LSTM. Vui lòng chạy training trước.")

        self.lstm_model = load_model(lstm_path)
        self.scaler = joblib.load(scaler_path)
        self.n_features = self.scaler.n_features_in_
        print("Mô hình LSTM đã được tải thành công.")
    
    def evaluate_and_plot_lstm(self,output_dir, all_features_df, train_end_idx, val_end_idx, n_forecast_horizon):
        """
        Đánh giá mô hình Multi-Output trên tập Test bằng phương pháp Walk-Forward
        và vẽ biểu đồ kết quả.
        """
        print("\n--- Bắt đầu Đánh giá Mô hình Multi-Output trên tập Test ---")
        
        test_df = all_features_df.iloc[val_end_idx:]
        
        all_predictions = []
        all_dates = []

        current_idx = val_end_idx 
        while current_idx < len(all_features_df):
            start_input_idx = current_idx - self.n_steps_in
            if start_input_idx < 0:
                break
            
            input_features = all_features_df.iloc[start_input_idx : current_idx]
            
            # <<< THAY ĐỔI Ở ĐÂY: Bỏ tham số thứ hai khi gọi forecast >>>
            predicted_sequence = self.forecast(input_features)
            
            # Logic còn lại giữ nguyên
            end_pred_idx = current_idx + len(predicted_sequence) # Dùng độ dài thực tế của kết quả
            actual_dates_for_preds = all_features_df.index[current_idx:end_pred_idx]
            
            predicted_sequence = predicted_sequence[:len(actual_dates_for_preds)]

            all_predictions.extend(predicted_sequence)
            all_dates.extend(actual_dates_for_preds)
            
            # Trượt cửa sổ đi một khoảng bằng độ dài chuỗi dự báo THỰC TẾ
            current_idx += len(predicted_sequence)

        prediction_series = pd.Series(all_predictions, index=all_dates)
        
        # 3. Tính toán các chỉ số lỗi
        # So khớp dữ liệu dự đoán và thực tế theo index
        comparison_df = pd.DataFrame({'Actual': all_features_df['Close'], 'Predicted': prediction_series})
        comparison_df.dropna(inplace=True) # Chỉ giữ lại các điểm có cả giá trị thực và dự đoán
        
        evaluation_results = calculate_all_metrics(comparison_df['Actual'], comparison_df['Predicted'])
    
        print("\n--- Kết quả Đánh giá Multi-Output trên Tập Test ---")
        print(f"MAE:   {evaluation_results['MAE']:.4f}")
        print(f"MAPE:  {evaluation_results['MAPE']:.4f}%")
        print(f"sMAPE: {evaluation_results['sMAPE']:.4f}%")
        print(f"MPE:   {evaluation_results['MPE']:.4f}%")
        print(f"MSE:   {evaluation_results['MSE']:.4f}")
        print(f"RMSE:  {evaluation_results['RMSE']:.4f}")
        print(f"RMSPE: {evaluation_results['RMSPE']:.4f}%")
        print("---------------------------------------------------")

        # 4. Lưu kết quả đánh giá vào file
        os.makedirs(output_dir, exist_ok=True)  # Đảm bảo thư mục tồn tại

        evaluation_results_filename = os.path.join(output_dir, 'evaluation_results.txt')
        with open(evaluation_results_filename, 'w') as f:
            for metric, value in evaluation_results.items():
                f.write(f"{metric}: {value:.4f}\n")
        print(f"Kết quả đánh giá đã được lưu vào file '{evaluation_results_filename}'")
        
        # 4. Vẽ biểu đồ
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(20, 10))
        
        ax.plot(all_features_df.index, all_features_df['Close'], label='Giá Thực tế (Toàn bộ)', color='royalblue', linewidth=1.5)
        ax.plot(prediction_series.index, prediction_series.values, label='Giá Dự đoán (trên tập Test)', color='darkorange', linewidth=2, linestyle='--')
        
        ax.axvspan(all_features_df.index[0], all_features_df.index[train_end_idx], color='green', alpha=0.1, label='Tập Huấn luyện')
        ax.axvspan(all_features_df.index[train_end_idx], all_features_df.index[val_end_idx], color='yellow', alpha=0.1, label='Tập Thẩm định')
        ax.axvspan(all_features_df.index[val_end_idx], all_features_df.index[-1], color='red', alpha=0.1, label='Tập Kiểm tra')

        ax.set_title('LSTM evaluation', fontsize=18)
        ax.set_xlabel('Ngày', fontsize=14)
        ax.set_ylabel('Giá Đóng cửa', fontsize=14)
        ax.legend(fontsize=12)
        plt.grid(True)
        
        plot_filename = os.path.join(output_dir, 'evaluation_plot_multi_output.png')
        plt.savefig(plot_filename)
        print(f"Biểu đồ đánh giá đã được lưu vào file '{plot_filename}'")
        plt.show()