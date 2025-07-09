import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
import matplotlib.pyplot as plt

# Giả sử bạn đã import hàm này
from eval_metric import calculate_all_metrics 

class HMMForecaster:
    def __init__(self, n_hidden_states=8):
        """
        Khởi tạo HMMForecaster với số lượng trạng thái ẩn.
        Args:
            n_hidden_states (int): Số trạng thái ẩn của HMM.
        """
        self.n_hidden_states = n_hidden_states
        self.hmm = GaussianHMM(n_components=self.n_hidden_states, covariance_type="full", n_iter=100)
        self.avg_state_changes = None

    def train(self, train_features_df):
        """
        Huấn luyện HMM và tính toán các đặc tính của trạng thái.
        Args:
            train_features_df (pd.DataFrame): Dữ liệu huấn luyện có cột 'Close' và 'Volume'.
        """
        print(f"Huấn luyện HMM với {self.n_hidden_states} trạng thái...")
        
        train_data = train_features_df[['Close', 'Volume']].copy()
        train_data['Price_Change'] = train_data['Close'].diff()
        train_data.dropna(inplace=True)
        
        hmm_train_features = train_data[['Price_Change', 'Volume']].values
        
        jitter = 1e-4 * np.std(hmm_train_features, axis=0) 
        hmm_train_features_jittered = hmm_train_features + np.random.randn(*hmm_train_features.shape) * jitter

        # Huấn luyện HMM trên dữ liệu đã thêm nhiễu
        self.hmm.fit(hmm_train_features_jittered)
        
        # Các bước tính toán avg_state_changes vẫn dùng dữ liệu gốc để không bị ảnh hưởng
        hidden_states = self.hmm.predict(hmm_train_features_jittered) # Dự đoán trên dữ liệu có nhiễu
        self.avg_state_changes = np.zeros(self.n_hidden_states)
        for i in range(self.n_hidden_states):
            state_indices = np.where(hidden_states == i)[0]
            if len(state_indices) > 0:
                # Tính giá trị trung bình trên dữ liệu gốc
                self.avg_state_changes[i] = np.mean(train_data['Price_Change'].iloc[state_indices])
            else:
                 self.avg_state_changes[i] = 0

    def forecast(self, features_df, n_forecast_horizon):
        """
        Dự báo giá trị tương lai dựa trên trạng thái ẩn hiện tại.
        Args:
            features_df (pd.DataFrame): Dữ liệu đầu vào có cột 'Close' và 'Volume'.
            n_forecast_horizon (int): Số bước dự báo.
        Returns:
            np.ndarray: Chuỗi giá trị dự báo.
        """
        if self.avg_state_changes is None:
            raise Exception("Mô hình chưa được huấn luyện.")
        last_data = features_df[['Close', 'Volume']].tail(1).copy()
        last_data['Price_Change'] = features_df['Close'].diff().iloc[-1]
        last_state_features = last_data[['Price_Change', 'Volume']].values
        current_state = self.hmm.predict(last_state_features)[0]
        predicted_change = self.avg_state_changes[current_state]
        last_close_price = features_df['Close'].iloc[-1]
        forecast_sequence = []
        current_price = last_close_price
        for _ in range(n_forecast_horizon):
            current_price += predicted_change
            forecast_sequence.append(current_price)
        return np.array(forecast_sequence)
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
        
        try:
            model = joblib.load(model_path)
            print(f"Mô hình HMM Forecaster đã được tải thành công từ '{model_path}'")
            # Kiểm tra xem đối tượng được tải có đúng là HMMForecaster không
            if not isinstance(model, HMMForecaster):
                print("CẢNH BÁO: File được tải không phải là một đối tượng HMMForecaster.")
            return model
        except Exception as e:
            print(f"LỖI: Có lỗi xảy ra khi tải mô hình từ '{model_path}'. Lỗi: {e}")
            raise # Ném lại lỗi để chương trình gọi biết và xử lý


    def save_model(self, model_dir, symbol, timeframe):
        """
        Lưu mô hình HMM đã huấn luyện vào file.
        Args:
            model_dir (str): Thư mục lưu mô hình.
            symbol (str): Mã tài sản.
            timeframe (str): Khung thời gian.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f'{symbol}_{timeframe}_hmm_forecaster_{self.n_hidden_states}states.pkl')
        joblib.dump(self, model_path)
        print(f"Mô hình HMM Forecaster đã được lưu tại '{model_path}'")

    def evaluate_and_plot_hmm(self,output_dir, all_features_df, train_end_idx, val_end_idx, n_forecast_horizon):
        """
        Đánh giá mô hình HMM trên tập Test bằng phương pháp Walk-Forward và vẽ biểu đồ so sánh.
        Args:
            all_features_df (pd.DataFrame): Dữ liệu đầy đủ.
            train_end_idx (int): Chỉ số kết thúc tập huấn luyện.
            val_end_idx (int): Chỉ số kết thúc tập thẩm định.
            n_forecast_horizon (int): Số bước dự báo mỗi lần.
        Returns:
            dict: Kết quả đánh giá các chỉ số lỗi.
        """
        print(f"\n--- Bắt đầu Đánh giá HMM Forecaster ({self.n_hidden_states} states) ---")
        
        all_predictions = []
        all_dates = []


        input_window_size = 60 

        current_idx = val_end_idx 
        while current_idx < len(all_features_df):
            start_input_idx = current_idx - input_window_size
            if start_input_idx < 0:
                print("Cảnh báo: Không đủ dữ liệu ở đầu tập test để bắt đầu đánh giá walk-forward.")
                break
            
            # Lấy dữ liệu cửa sổ hiện tại
            input_features = all_features_df.iloc[start_input_idx : current_idx]
            
            predicted_sequence = self.forecast(input_features, n_forecast_horizon)
            
            # Logic xử lý kết quả dự báo
            end_pred_idx = current_idx + len(predicted_sequence)
            actual_dates_for_preds = all_features_df.index[current_idx:end_pred_idx]
            predicted_sequence = predicted_sequence[:len(actual_dates_for_preds)]

            all_predictions.extend(predicted_sequence)
            all_dates.extend(actual_dates_for_preds)
            
            if len(predicted_sequence) == 0:
                print("Dừng đánh giá vì không thể tạo thêm dự báo.")
                break
            current_idx += len(predicted_sequence)

        if not all_dates:
            print("Không có đủ dữ liệu trong tập test để đánh giá.")
            return None # Trả về None nếu không có kết quả

        # Tính toán các metrics
        prediction_series = pd.Series(all_predictions, index=all_dates)
        comparison_df = pd.DataFrame({'Actual': all_features_df['Close'], 'Predicted': prediction_series})
        comparison_df.dropna(inplace=True)
        
        evaluation_results = calculate_all_metrics(comparison_df['Actual'], comparison_df['Predicted'])
        
        print(f"\n--- Kết quả Đánh giá HMM ({self.n_hidden_states} states) ---")
        for key, value in evaluation_results.items():
            print(f"{key}: {value:.4f}")
        print("---------------------------------------------------")
        
        # Vẽ biểu đồ
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(20, 10))
        
        ax.plot(all_features_df.index, all_features_df['Close'], label='Giá Thực tế', color='royalblue', linewidth=1.5)
        ax.plot(prediction_series.index, prediction_series.values, label=f'Dự đoán HMM ({self.n_hidden_states} states)', color='darkorange', linewidth=2, linestyle='--')
        
        ax.axvspan(all_features_df.index[0], all_features_df.index[train_end_idx], color='green', alpha=0.1, label='Tập Huấn luyện')
        ax.axvspan(all_features_df.index[train_end_idx], all_features_df.index[val_end_idx], color='yellow', alpha=0.1, label='Tập Thẩm định')
        ax.axvspan(all_features_df.index[val_end_idx], all_features_df.index[-1], color='red', alpha=0.1, label='Tập Kiểm tra')

        ax.set_title(f'So sánh Giá Thực tế và Dự đoán HMM ({self.n_hidden_states} states)', fontsize=18)
        ax.set_xlabel('Ngày', fontsize=14)
        ax.set_ylabel('Giá Đóng cửa', fontsize=14)
        ax.legend(fontsize=12)
        plt.grid(True)
        
        plot_filename = os.path.join(output_dir, f'evaluation_plot_hmm_{self.n_hidden_states}_states.png')
        plt.savefig(plot_filename)
        print(f"Biểu đồ đánh giá đã được lưu vào file '{plot_filename}'")
        plt.close(fig)
        
        return evaluation_results