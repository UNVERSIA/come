import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
import os

class CarbonLSTMPredictor:
    def __init__(self, lookback=30, n_features=10):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback = lookback
        self.n_features = n_features
        self.feature_columns = [
            '处理水量(m³)', '电耗(kWh)', 'PAC投加量(kg)', 'PAM投加量(kg)', 
            '次氯酸钠投加量(kg)', '进水COD(mg/L)', '出水COD(mg/L)', 
            '进水TN(mg/L)', '出水TN(mg/L)', 'total_CO2eq'
        ]

    def _create_dataset(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X[i:(i + time_steps)]
            Xs.append(v)
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    def _convert_to_monthly(self, df):
        """将日度数据转换为月度数据"""
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        
        # 按月聚合
        monthly_df = df.resample('M').agg({
            '处理水量(m³)': 'sum',
            '电耗(kWh)': 'sum',
            'PAC投加量(kg)': 'sum',
            'PAM投加量(kg)': 'sum',
            '次氯酸钠投加量(kg)': 'sum',
            '进水COD(mg/L)': 'mean',
            '出水COD(mg/L)': 'mean',
            '进水TN(mg/L)': 'mean',
            '出水TN(mg/L)': 'mean',
            'total_CO2eq': 'sum'
        }).reset_index()
        
        # 添加年月标识
        monthly_df['年月'] = monthly_df['日期'].dt.strftime('%Y年%m月')
        return monthly_df

    def prepare_data(self, df, target_column='total_CO2eq'):
        """准备训练数据"""
        # 确保所有必需的列都存在
        for col in self.feature_columns:
            if col not in df.columns:
                if col == 'total_CO2eq':
                    # 计算碳排放
                    from carbon_calculator import CarbonCalculator
                    calculator = CarbonCalculator()
                    df = calculator.calculate_direct_emissions(df)
                    df = calculator.calculate_indirect_emissions(df)
                    df = calculator.calculate_unit_emissions(df)
                else:
                    # 添加缺失列并用0填充
                    df[col] = 0
        
        # 选择特征列
        features = df[self.feature_columns].copy()
        
        # 处理缺失值
        features = features.fillna(0)
        
        # 归一化
        scaled_data = self.scaler.fit_transform(features)
        
        # 创建时间序列数据集
        X = scaled_data[:, :-1]  # 所有特征除了目标列
        y = scaled_data[:, -1]   # 目标列
        
        X_train, y_train = self._create_dataset(X, y, self.lookback)
        
        return X_train, y_train, features

    def build_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse',
                     metrics=['mae'])
        
        return model

    def train(self, df, target_column='total_CO2eq', epochs=100, 
              validation_split=0.2, batch_size=32, save_path=None):
        """训练LSTM模型"""
        # 准备数据
        X_train, y_train, features = self.prepare_data(df, target_column)
        
        # 构建模型
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # 保存模型
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            # 保存scaler
            scaler_path = save_path.replace('.keras', '_scaler.save')
            joblib.dump(self.scaler, scaler_path)
        
        return history

    def load_model(self, model_path):
        """加载预训练模型"""
        try:
            self.model = load_model(model_path)
            # 加载scaler
            scaler_path = model_path.replace('.keras', '_scaler.save')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def predict(self, df, target_column='total_CO2eq', steps=12):
        """使用训练好的模型进行预测"""
        if self.model is None:
            raise ValueError("模型未训练或加载，请先训练或加载模型")
        
        # 准备数据
        X_train, y_train, features = self.prepare_data(df, target_column)
        
        # 使用最后lookback天的数据作为初始输入
        last_sequence = X_train[-1:]
        
        predictions = []
        confidence_intervals = []
        
        # 进行多步预测
        for _ in range(steps):
            # 预测下一步
            pred = self.model.predict(last_sequence, verbose=0)[0, 0]
            predictions.append(pred)
            
            # 更新序列 (使用预测值)
            new_sequence = np.roll(last_sequence[0], -1, axis=0)
            new_features = np.zeros((1, self.n_features - 1))
            
            # 创建新的输入序列
            new_sequence[-1, :] = new_features
            new_sequence[-1, -1] = pred  # 将预测值放在目标列位置
            
            last_sequence = np.expand_dims(new_sequence, axis=0)
            
            # 计算置信区间 (简单方法)
            confidence_intervals.append(pred * 0.1)  # 假设10%的误差
        
        # 反归一化预测结果
        dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_array[:, -1] = predictions
        predictions_rescaled = self.scaler.inverse_transform(dummy_array)[:, -1]
        
        # 反归一化置信区间
        dummy_array[:, -1] = confidence_intervals
        confidence_rescaled = self.scaler.inverse_transform(dummy_array)[:, -1]
        
        # 生成预测日期
        last_date = pd.to_datetime(df['日期'].iloc[-1])
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='M'
        )
        
        # 创建预测结果DataFrame
        result_df = pd.DataFrame({
            '日期': future_dates,
            'predicted_CO2eq': predictions_rescaled,
            'lower_bound': predictions_rescaled - confidence_rescaled,
            'upper_bound': predictions_rescaled + confidence_rescaled
        })
        
        return result_df

    def evaluate(self, df, target_column='total_CO2eq'):
        """评估模型性能"""
        X_train, y_train, features = self.prepare_data(df, target_column)
        
        # 预测
        y_pred = self.model.predict(X_train)
        
        # 反归一化
        dummy_array = np.zeros((len(y_pred), len(self.feature_columns)))
        dummy_array[:, -1] = y_pred.flatten()
        y_pred_rescaled = self.scaler.inverse_transform(dummy_array)[:, -1]
        
        dummy_array[:, -1] = y_train
        y_true_rescaled = self.scaler.inverse_transform(dummy_array)[:, -1]
        
        # 计算指标
        mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
        mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred_rescaled,
            'actuals': y_true_rescaled
        }
