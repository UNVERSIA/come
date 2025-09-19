# lstm_predictor.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import timedelta
import warnings
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

from carbon_calculator import CarbonCalculator


class CarbonLSTMPredictor:
    def __init__(self, sequence_length=12, forecast_months=12):  # 改为12个月序列长度，预测12个月
        self.sequence_length = sequence_length  # 使用12个月的历史数据
        self.forecast_months = forecast_months  # 预测未来12个月
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.target_scaler = MinMaxScaler()
        self.feature_columns = [
            '处理水量(m³)', '电耗(kWh)', 'PAC投加量(kg)',
            'PAM投加量(kg)', '次氯酸钠投加量(kg)',
            '进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)'
        ]
        self.start_date = pd.Timestamp('2018-01-01')
        self.end_date = pd.Timestamp('2024-12-31')

    def load_monthly_data(self, file_path="data/simulated_data_monthly.csv"):
        """加载月度数据"""
        try:
            monthly_data = pd.read_csv(file_path)
            monthly_data['日期'] = pd.to_datetime(monthly_data['日期'])
            return monthly_data
        except FileNotFoundError:
            print(f"月度数据文件 {file_path} 未找到，将尝试生成")
            return None

    def build_model(self, input_shape):
        """构建LSTM模型 - 针对月度数据优化"""
        if input_shape is None:
            input_shape = (self.sequence_length, len(self.feature_columns))

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),  # 增加神经元数量
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),  # 添加激活函数
            Dense(1)  # 输出层不需要激活函数
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, df, target_column='total_CO2eq', epochs=100, batch_size=16,
              validation_split=0.2, save_path='models/carbon_lstm_model.keras'):
        """训练模型 - 针对月度数据"""
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 检查是否为月度数据，如果不是则转换
        if '年月' not in df.columns:
            print("输入数据不是月度数据，正在转换...")
            df = self._convert_to_monthly(df)

        # 准备训练数据
        X, y = self.prepare_training_data(df, target_column)

        if len(X) == 0:
            raise ValueError("没有足够的数据来训练模型")

        print(f"训练数据形状: X={X.shape}, y={y.shape}")

        # 构建并训练模型
        self.model = self.build_model((X.shape[1], X.shape[2]))

        # 使用更多的训练轮次，因为月度数据较少
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=True
        )

        # 保存模型和缩放器
        self.model.save(save_path)

        # 保存元数据
        serializable_scalers = {}
        for col, scaler in self.feature_scalers.items():
            serializable_scalers[col] = {
                'min_': scaler.min_,
                'scale_': scaler.scale_,
                'data_min_': scaler.data_min_,
                'data_max_': scaler.data_max_,
                'data_range_': scaler.data_range_
            }

        joblib.dump({
            'feature_scalers': serializable_scalers,
            'sequence_length': self.sequence_length,
            'forecast_months': self.forecast_months,
            'feature_columns': self.feature_columns,
            'target_scaler': {
                'min_': self.target_scaler.min_,
                'scale_': self.target_scaler.scale_,
                'data_min_': self.target_scaler.data_min_,
                'data_max_': self.target_scaler.data_max_,
                'data_range_': self.target_scaler.data_range_
            } if hasattr(self.target_scaler, 'min_') else None
        }, save_path.replace('.keras', '_metadata.pkl'))

        return history

    def _convert_to_monthly(self, daily_df):
        """将日度数据转换为月度数据"""
        df = daily_df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)

        # 按月聚合
        monthly_df = df.resample('M').agg({
            '处理水量(m³)': 'mean',
            '电耗(kWh)': 'mean',
            'PAC投加量(kg)': 'mean',
            'PAM投加量(kg)': 'mean',
            '次氯酸钠投加量(kg)': 'mean',
            '进水COD(mg/L)': 'mean',
            '出水COD(mg/L)': 'mean',
            '进水TN(mg/L)': 'mean',
            '出水TN(mg/L)': 'mean',
            'total_CO2eq': 'mean'
        }).reset_index()

        monthly_df['年月'] = monthly_df['日期'].dt.strftime('%Y年%m月')
        return monthly_df

    def prepare_training_data(self, df, target_column):
        """准备月度训练数据"""
        # 确保数据按日期排序
        df = df.sort_values('日期').reset_index(drop=True)

        # 检查目标列是否存在且有有效数据
        if target_column not in df.columns or df[target_column].isna().all():
            raise ValueError(f"目标列 '{target_column}' 不存在或全部为NaN值")

        # 检查是否有足够的数据（至少需要序列长度+1个月的数据）
        if len(df) < self.sequence_length + 1:
            raise ValueError(f"需要至少 {self.sequence_length + 1} 个月的记录进行训练，当前只有 {len(df)} 个月")

        # 确保所有必需的特征列都存在
        for col in self.feature_columns:
            if col not in df.columns:
                print(f"警告: 特征列 '{col}' 不存在，将使用默认值填充")
                df[col] = self._get_default_value(col)

        # 填充NaN值 - 修复pandas版本兼容性问题
        df = df.ffill().bfill().fillna(0)

        # 初始化缩放器
        self.feature_scalers = {}
        for col in self.feature_columns:
            self.feature_scalers[col] = MinMaxScaler()
            self.feature_scalers[col].fit(df[col].values.reshape(-1, 1))

        # 目标变量缩放器
        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(df[target_column].values.reshape(-1, 1))

        # 创建序列数据
        X, y = [], []

        for i in range(self.sequence_length, len(df)):
            # 提取特征序列
            sequence_features = []
            for col in self.feature_columns:
                col_data = df[col].iloc[i - self.sequence_length:i].values
                scaled_data = self.feature_scalers[col].transform(col_data.reshape(-1, 1)).flatten()
                sequence_features.append(scaled_data)

            # 堆叠特征序列
            stacked_sequence = np.stack(sequence_features, axis=1)

            # 缩放目标值
            target = df[target_column].iloc[i]
            scaled_target = self.target_scaler.transform([[target]])[0][0]

            X.append(stacked_sequence)
            y.append(scaled_target)

        print(f"成功创建 {len(X)} 个月度训练序列")
        return np.array(X), np.array(y)

    def predict(self, df, target_column='total_CO2eq', steps=12):
        """月度预测方法"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载或训练模型")

        # 转换为月度数据（如果需要）
        if '年月' not in df.columns:
            df = self._convert_to_monthly(df)

        # 确保数据已排序
        df = df.sort_values('日期').reset_index(drop=True)

        # 确保所有必需的特征列都存在
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = self._get_default_value(col)

        # 填充NaN值 - 修复pandas兼容性问题
        df = df.ffill().bfill().fillna(0)

        # 使用最后12个月数据作为输入序列
        if len(df) < self.sequence_length:
            raise ValueError(f"需要至少 {self.sequence_length} 个月的历史数据进行预测")

        # 准备特征数据
        X = self._prepare_features_for_prediction(df.tail(self.sequence_length))

        if X is None or len(X) == 0:
            raise ValueError("无法准备特征数据进行预测")

        # 进行预测
        predictions = []
        lower_bounds = []
        upper_bounds = []

        # 使用最后一段序列作为初始输入
        current_sequence = X[-1:]

        # 获取历史统计信息用于置信区间计算
        historical_values = df[target_column].values
        historical_mean = np.mean(historical_values)
        historical_std = np.std(historical_values)

        # 确保目标缩放器已拟合 - 修复sklearn版本兼容性问题
        scaler_fitted = False
        try:
            # 检查不同版本sklearn的拟合状态
            if hasattr(self.target_scaler, 'n_samples_seen_'):
                scaler_fitted = self.target_scaler.n_samples_seen_ > 0
            elif hasattr(self.target_scaler, 'scale_'):
                scaler_fitted = self.target_scaler.scale_ is not None
            elif hasattr(self.target_scaler, 'data_min_'):
                scaler_fitted = self.target_scaler.data_min_ is not None
        except:
            scaler_fitted = False

        if not scaler_fitted:
            target_values = df[target_column].dropna().values.reshape(-1, 1)
            if len(target_values) > 0:
                self.target_scaler.fit(target_values)
            else:
                # 如果没有目标数据，使用默认范围
                self.target_scaler.fit([[0], [2000]])  # 假设碳排放在0-2000范围内

        # 获取最后一个月的日期作为基准
        last_date = df['日期'].max()

        # 进行递归预测，添加趋势稳定化控制
        trend_damping = 0.95  # 趋势衰减因子，防止预测过度发散
        base_prediction = historical_mean  # 基准预测值

        for i in range(steps):
            # 预测下一步
            try:
                # 验证输入序列形状
                if current_sequence.shape[1:] != self.model.input_shape[1:]:
                    print(f"警告: 输入形状不匹配，期望 {self.model.input_shape[1:]}，实际 {current_sequence.shape[1:]}")
                    # 尝试重新构建序列
                    current_sequence = self._prepare_features_for_prediction(df.tail(self.sequence_length))
                    if current_sequence is None:
                        raise ValueError("无法构建有效的输入序列")

                pred_scaled = self.model.predict(current_sequence, verbose=0)[0][0]
                # 添加适度随机变化
                pred_scaled += np.random.normal(0, 0.002)
            except Exception as e:
                print(f"模型预测错误: {e}")
                # 如果预测失败，使用历史均值
                try:
                    pred_scaled = self.target_scaler.transform([[historical_mean]])[0][0]
                except:
                    pred_scaled = 0.5  # 使用中间值作为fallback

            # 逆变换预测值
            try:
                pred = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
            except Exception as e:
                print(f"逆变换失败: {e}")
                pred = historical_mean

            # 添加趋势稳定化：随着预测步数增加，逐渐回归历史均值
            stabilization_factor = (trend_damping ** i)
            pred = pred * stabilization_factor + base_prediction * (1 - stabilization_factor)

            # 确保预测值在合理范围内（比原来更严格的控制）
            pred = max(0, pred)
            # 限制预测变化幅度不超过±30%
            pred = np.clip(pred, historical_mean * 0.7, historical_mean * 1.3)

            predictions.append(pred)

            # 计算更保守的置信区间
            error_estimate = historical_std * 0.15  # 减小置信区间
            lower_bounds.append(max(0, pred - error_estimate))
            upper_bounds.append(pred + error_estimate)

            # 简化版序列更新 - 保持当前序列不变以避免累积误差
            # 在生产环境中，应该根据预测结果智能更新特征序列

            # 获取最后一个月的日期作为基准
            last_date = df['日期'].max()

            # 进行递归预测，添加趋势稳定化控制
            trend_damping = 0.95  # 趋势衰减因子，防止预测过度发散
            base_prediction = historical_mean  # 基准预测值

            for i in range(steps):
                # 预测下一步
                try:
                    pred_scaled = self.model.predict(current_sequence, verbose=0)[0][0]
                    # 添加适度随机变化
                    pred_scaled += np.random.normal(0, 0.002)
                except Exception as e:
                    print(f"模型预测错误: {e}")
                    # 如果预测失败，使用历史平均值
                    pred_scaled = self.target_scaler.transform([[historical_mean]])[0][0]

                # 逆变换预测值
                try:
                    pred = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
                except Exception as e:
                    print(f"逆变换失败: {e}")
                    pred = historical_mean

                # 添加趋势稳定化：随着预测步数增加，逐渐回归历史均值
                stabilization_factor = (trend_damping ** i)
                pred = pred * stabilization_factor + base_prediction * (1 - stabilization_factor)

                # 确保预测值在合理范围内（比原来更严格的控制）
                pred = max(0, pred)
                # 限制预测变化幅度不超过±30%
                pred = np.clip(pred, historical_mean * 0.7, historical_mean * 1.3)

                predictions.append(pred)

                # 计算更保守的置信区间
                error_estimate = historical_std * 0.15  # 减小置信区间
                lower_bounds.append(max(0, pred - error_estimate))
                upper_bounds.append(pred + error_estimate)

            # 生成2025年12个月的正确日期
            prediction_dates = []
            for i in range(1, steps + 1):
                # 使用月份的第一天作为标识，更清晰地表示月度数据
                next_date = pd.Timestamp(year=2025, month=i, day=1)
                prediction_dates.append(next_date)

            # 创建结果DataFrame
            result_df = pd.DataFrame({
                '日期': prediction_dates,
                'predicted_CO2eq': predictions,
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds
            })

            # 添加清晰的月度标识
            result_df['年月'] = result_df['日期'].dt.strftime('%Y年%m月')
            result_df['月度标识'] = result_df['日期'].dt.strftime('%Y-%m')

            return result_df

    def _prepare_features_for_prediction(self, df):
        """为预测准备特征数据 - 修复版本"""
        if df is None or df.empty:
            return None

        if len(df) < self.sequence_length:
            raise ValueError(f"需要至少 {self.sequence_length} 个月的数据进行预测")

        # 复制数据避免修改原始数据
        df = df.copy()

        # 确保所有特征列都存在且有有效数据
        for col in self.feature_columns:
            if col not in df.columns or df[col].isna().all():
                df[col] = self._get_default_value(col)
            elif df[col].isna().any():
                col_mean = df[col].mean()
                if pd.isna(col_mean):
                    col_mean = self._get_default_value(col)
                df[col] = df[col].fillna(col_mean)

        # 确保所有特征都有缩放器
        for col in self.feature_columns:
            if col not in self.feature_scalers:
                self.feature_scalers[col] = MinMaxScaler()
                col_values = df[col].values.reshape(-1, 1)
                self.feature_scalers[col].fit(col_values)

        # 获取最后sequence_length行的数据
        last_data = df.iloc[-self.sequence_length:].copy()

        # 强制确保数据长度
        if len(last_data) < self.sequence_length:
            # 如果数据不足，重复最后一行来填充
            last_row = last_data.iloc[-1:] if not last_data.empty else pd.DataFrame()
            while len(last_data) < self.sequence_length:
                last_data = pd.concat([last_data, last_row], ignore_index=True)

        # 只取前sequence_length行
        last_data = last_data.iloc[:self.sequence_length]

        # 创建特征矩阵 - 统一处理所有特征
        feature_matrix = np.zeros((self.sequence_length, len(self.feature_columns)), dtype=np.float32)

        for j, col in enumerate(self.feature_columns):
            col_data = last_data[col].values

            # 确保长度一致
            if len(col_data) != self.sequence_length:
                col_data = np.resize(col_data, self.sequence_length)

            # 处理NaN值
            if np.isnan(col_data).any():
                col_mean = self._get_default_value(col)
                col_data = np.where(np.isnan(col_data), col_mean, col_data)

            # 缩放数据
            try:
                scaled_data = self.feature_scalers[col].transform(col_data.reshape(-1, 1)).flatten()
                feature_matrix[:, j] = scaled_data
            except Exception as e:
                print(f"缩放特征 {col} 时出错: {e}")
                feature_matrix[:, j] = 0.0

        # 验证形状
        expected_shape = (self.sequence_length, len(self.feature_columns))
        if feature_matrix.shape != expected_shape:
            print(f"错误: 特征矩阵形状 {feature_matrix.shape} 不匹配期望形状 {expected_shape}")
            return None

        return np.array([feature_matrix], dtype=np.float32)

    def load_model(self, model_path=None):
        """加载预训练模型 - 兼容性改进版"""
        # 如果没有提供模型路径，使用默认路径
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, "models")
            model_path = os.path.join(models_dir, "carbon_lstm_model.keras")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 构建所有可能的文件路径
        possible_model_paths = [
            model_path,
            model_path.replace('.keras', '.h5'),
            'models/carbon_lstm_model.h5',
            'models/carbon_lstm.h5',
            'models/carbon_lstm_model.weights.h5'
        ]

        possible_meta_paths = [
            model_path.replace('.keras', '_metadata.pkl').replace('.h5', '_metadata.pkl'),
            'models/carbon_lstm_metadata.pkl',
            model_path.replace('.keras', '.pkl').replace('.h5', '.pkl')
        ]

        # 查找模型文件
        found_model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                found_model_path = path
                break

        if not found_model_path:
            logger.warning("未找到预训练模型文件，模型将保持未加载状态")
            self.model = None
            return False

        # 查找并加载元数据
        metadata_path = None
        for path in possible_meta_paths:
            if os.path.exists(path):
                metadata_path = path
                break

        # 加载元数据
        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                serializable_scalers = metadata.get('feature_scalers', {})

                # 重建特征缩放器
                self.feature_scalers = {}
                for col, scaler_params in serializable_scalers.items():
                    new_scaler = MinMaxScaler()
                    if scaler_params is not None:
                        new_scaler.min_ = scaler_params['min_']
                        new_scaler.scale_ = scaler_params['scale_']
                        new_scaler.data_min_ = scaler_params['data_min_']
                        new_scaler.data_max_ = scaler_params['data_max_']
                        new_scaler.data_range_ = scaler_params['data_range_']
                    self.feature_scalers[col] = new_scaler

                # 重建目标缩放器
                target_scaler_params = metadata.get('target_scaler')
                self.target_scaler = MinMaxScaler()
                if target_scaler_params is not None:
                    self.target_scaler.min_ = target_scaler_params['min_']
                    self.target_scaler.scale_ = target_scaler_params['scale_']
                    self.target_scaler.data_min_ = target_scaler_params['data_min_']
                    self.target_scaler.data_max_ = target_scaler_params['data_max_']
                    self.target_scaler.data_range_ = target_scaler_params['data_range_']

                self.sequence_length = metadata.get('sequence_length', 12)
                self.forecast_months = metadata.get('forecast_months', 12)
                self.feature_columns = metadata.get('feature_columns', self.feature_columns)
            except Exception as e:
                logger.warning(f"加载元数据失败: {str(e)}")

        # 尝试直接加载模型
        try:
            self.model = load_model(found_model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info("模型直接加载成功")
            return True
        except Exception as e:
            logger.error(f"所有加载方式均失败: {str(e)}")
            self.model = None
            return False

    def _get_default_value(self, col_name):
        """获取特征的典型默认值"""
        defaults = {
            '处理水量(m³)': 10000.0,
            '电耗(kWh)': 3000.0,
            'PAC投加量(kg)': 0.0,
            'PAM投加量(kg)': 0.0,
            '次氯酸钠投加量(kg)': 0.0,
            '进水COD(mg/L)': 200.0,
            '出水COD(mg/L)': 50.0,
            '进水TN(mg/L)': 40.0,
            '出水TN(mg/L)': 15.0,
            'total_CO2eq': 1000.0
        }
        return defaults.get(col_name, 0.0)


# 使用示例
if __name__ == "__main__":
    # 加载月度数据
    predictor = CarbonLSTMPredictor()

    # 如果有月度数据文件则加载，否则生成
    try:
        monthly_data = pd.read_csv("data/simulated_data_monthly.csv")
        monthly_data['日期'] = pd.to_datetime(monthly_data['日期'])
    except FileNotFoundError:
        print("未找到月度数据文件，正在生成...")
        from data_simulator import DataSimulator

        simulator = DataSimulator()
        daily_data = simulator.generate_simulated_data()
        monthly_data = predictor._convert_to_monthly(daily_data)
        monthly_data.to_csv("data/simulated_data_monthly.csv", index=False)

    # 计算总碳排放（如果尚未计算）
    if 'total_CO2eq' not in monthly_data.columns:
        calculator = CarbonCalculator()
        monthly_data = calculator.calculate_direct_emissions(monthly_data)
        monthly_data = calculator.calculate_indirect_emissions(monthly_data)
        monthly_data = calculator.calculate_unit_emissions(monthly_data)

    # 训练预测模型
    history = predictor.train(monthly_data, 'total_CO2eq', epochs=50)

    # 进行预测
    predictions = predictor.predict(monthly_data, 'total_CO2eq', steps=12)
    print("月度模型训练完成并进行预测")
