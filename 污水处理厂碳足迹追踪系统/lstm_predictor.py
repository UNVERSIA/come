# lstm_predictor.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
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
    def __init__(self, sequence_length=30, forecast_days=7):
        self.sequence_length = sequence_length
        self.forecast_days = forecast_days
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.feature_columns = [
            '处理水量(m³)', '电耗(kWh)', 'PAC投加量(kg)',
            'PAM投加量(kg)', '次氯酸钠投加量(kg)',
            '进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)'
        ]

    def prepare_training_data(self, df, target_column='total_CO2eq'):
        """准备训练数据"""
        # 确保数据按日期排序
        df = df.sort_values('日期').reset_index(drop=True)

        # 单独缩放每个特征
        scaled_features = {}
        for col in self.feature_columns + [target_column]:
            if col not in df.columns:
                # 如果缺少某些列，使用0填充
                df[col] = 0.0

            self.feature_scalers[col] = MinMaxScaler()
            scaled_features[col] = self.feature_scalers[col].fit_transform(df[[col]])

        # 创建序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(df) - self.forecast_days):
            # 特征序列
            seq_features = []
            for col in self.feature_columns:
                seq_features.append(scaled_features[col][i - self.sequence_length:i])

            # 目标值（未来forecast_d天的平均值）
            target_values = scaled_features[target_column][i:i + self.forecast_days]
            seq_target = np.mean(target_values)

            X.append(np.hstack(seq_features))
            y.append(seq_target)

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """构建LSTM模型 - 使用更兼容的方式"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, df, target_column='total_CO2eq', epochs=50, batch_size=32,
              save_path='models/carbon_lstm.keras'):
        """训练模型"""
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 准备数据
        X, y = self.prepare_training_data(df, target_column)

        if len(X) == 0:
            raise ValueError("没有足够的数据来训练模型")

        # 构建并训练模型
        self.model = self.build_model((X.shape[1], X.shape[2]))
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                 validation_split=0.2, verbose=1)

        # 保存模型和缩放器 - 使用新的Keras格式
        self.model.save(save_path)

        # 同时保存权重和架构以便兼容性
        self.model.save_weights(save_path.replace('.keras', '.weights.h5'))
        model_json = self.model.to_json()
        with open(save_path.replace('.keras', '_architecture.json'), 'w') as json_file:
            json_file.write(model_json)

        joblib.dump({
            'feature_scalers': self.feature_scalers,
            'sequence_length': self.sequence_length,
            'forecast_days': self.forecast_days,
            'feature_columns': self.feature_columns
        }, save_path.replace('.keras', '_metadata.pkl'))

        return history

    # lstm_predictor.py
    # 修改load_model方法

    def load_model(self, model_path='models/carbon_lstm.keras'):
        """加载预训练模型"""
        # 添加路径前缀
        if not model_path.startswith('./污水处理厂碳足迹追踪系统/'):
            model_path = f'./污水处理厂碳足迹追踪系统/{model_path}'

        # 检查文件是否存在
        if not os.path.exists(model_path):
            # 尝试不带前缀的路径
            alt_path = model_path.replace('./污水处理厂碳足迹追踪系统/', '')
            if os.path.exists(alt_path):
                model_path = alt_path

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试加载.h5格式的旧模型（兼容性）
            h5_path = model_path.replace('.keras', '.h5')
            if os.path.exists(h5_path):
                model_path = h5_path
            else:
                # 如果都没有找到，尝试其他可能的路径
                possible_paths = [
                    model_path,
                    h5_path,
                    model_path.replace('.keras', '.weights.h5'),
                    'models/carbon_lstm.h5',
                    'models/carbon_lstm.weights.h5'
                ]

                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        found = True
                        break

                if not found:
                    raise FileNotFoundError(f"模型文件不存在，尝试了以下路径: {possible_paths}")

        metadata_path = model_path.replace('.keras', '_metadata.pkl').replace('.h5', '_metadata.pkl')
        if not os.path.exists(metadata_path):
            # 尝试其他可能的元数据路径
            possible_meta_paths = [
                metadata_path,
                'models/carbon_lstm_metadata.pkl',
                model_path.replace('.keras', '.pkl').replace('.h5', '.pkl')
            ]

            found_meta = False
            for path in possible_meta_paths:
                if os.path.exists(path):
                    metadata_path = path
                    found_meta = True
                    break

            if not found_meta:
                raise FileNotFoundError(f"元数据文件不存在，尝试了以下路径: {possible_meta_paths}")

        # 先加载元数据
        try:
            metadata = joblib.load(metadata_path)
            self.feature_scalers = metadata['feature_scalers']
            self.sequence_length = metadata['sequence_length']
            self.forecast_days = metadata['forecast_days']
            self.feature_columns = metadata['feature_columns']
        except Exception as e:
            logger.warning(f"加载元数据失败: {str(e)}")
            # 设置默认值
            self.sequence_length = 30
            self.forecast_days = 7
            self.feature_columns = [
                '处理水量(m³)', '电耗(kWh)', 'PAC投加量(kg)',
                'PAM投加量(kg)', '次氯酸钠投加量(kg)',
                '进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)'
            ]

        try:
            # 尝试直接加载模型
            self.model = load_model(model_path)
            logger.info("模型加载成功")
        except Exception as e:
            # 如果直接加载失败，尝试使用权重和架构
            try:
                logger.warning(f"模型加载遇到兼容性问题: {str(e)}")
                logger.info("尝试使用备用加载方式...")

                # 尝试加载架构和权重
                architecture_path = model_path.replace('.keras', '_architecture.json').replace('.h5',
                                                                                               '_architecture.json')
                weights_path = model_path.replace('.keras', '.weights.h5').replace('.h5', '.weights.h5')

                # 如果权重文件不存在，尝试其他可能的路径
                if not os.path.exists(weights_path):
                    possible_weights = [
                        weights_path,
                        model_path.replace('.keras', '.h5').replace('.h5', '.h5'),
                        'models/carbon_lstm.weights.h5',
                        'models/carbon_lstm.h5'
                    ]

                    for path in possible_weights:
                        if os.path.exists(path):
                            weights_path = path
                            break

                if os.path.exists(architecture_path) and os.path.exists(weights_path):
                    # 从JSON加载模型架构
                    from tensorflow.keras.models import model_from_json
                    with open(architecture_path, 'r') as json_file:
                        model_json = json_file.read()
                    self.model = model_from_json(model_json)

                    # 加载权重
                    self.model.load_weights(weights_path)

                    # 编译模型
                    self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    logger.info("使用备用方式加载模型成功!")
                else:
                    # 重新构建模型结构
                    self.model = self.build_model((self.sequence_length, len(self.feature_columns)))

                    # 尝试加载权重
                    if os.path.exists(weights_path):
                        self.model.load_weights(weights_path)
                        logger.info("使用权重加载方式成功!")
                    else:
                        logger.warning("权重文件不存在，只能使用模型架构")
                        # 编译模型
                        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            except Exception as inner_e:
                logger.error(f"所有加载方式均失败: {str(inner_e)}")
                # 作为最后的手段，创建一个新的未训练模型
                logger.info("创建新的未训练模型")
                self.model = self.build_model((self.sequence_length, len(self.feature_columns)))
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def predict(self, df, target_column='total_CO2eq'):
        steps = 7
        """使用最近的数据进行多步预测"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载或训练模型")

        # 获取最近sequence_length天的数据
        recent_data = df.tail(self.sequence_length).copy()

        # 确保所有需要的列都存在
        for col in self.feature_columns:
            if col not in recent_data.columns:
                recent_data[col] = 0.0

        # 初始化预测结果
        predictions = []

        # 准备输入数据
        current_input = recent_data.copy()

        for step in range(steps):
            # 缩放特征
            scaled_features = []
            for col in self.feature_columns:
                if col in self.feature_scalers:
                    scaled_col = self.feature_scalers[col].transform(current_input[[col]].tail(self.sequence_length))
                    scaled_features.append(scaled_col)
                else:
                    # 如果缺少某些缩放器，使用默认值
                    default_value = np.zeros((self.sequence_length, 1))
                    scaled_features.append(default_value)

            # 准备输入数据
            X_input = np.hstack(scaled_features)
            X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])

            # 进行预测
            scaled_prediction = self.model.predict(X_input, verbose=0)[0][0]

            # 反缩放预测结果
            prediction = self.feature_scalers[target_column].inverse_transform(
                [[scaled_prediction]]
            )[0][0]

            predictions.append(prediction)

            # 更新输入数据（将预测值添加到特征中用于下一步预测）
            # 这里简化处理，实际应用中可能需要更复杂的数据更新逻辑
            new_row = current_input.iloc[-1:].copy()
            new_row[target_column] = prediction
            # 可以更新其他相关特征（如日期等）
            current_input = pd.concat([current_input, new_row]).tail(self.sequence_length)

        return predictions

    def generate_future_dates(self, last_date, days=7):
        """生成未来日期序列"""
        return [last_date + timedelta(days=i) for i in range(1, days + 1)]


# 使用示例
if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv("data/simulated_data.csv")
    data['日期'] = pd.to_datetime(data['日期'])

    # 计算总碳排放（假设已有碳核算结果）
    calculator = CarbonCalculator()
    data_with_emissions = calculator.calculate_direct_emissions(data)
    data_with_emissions = calculator.calculate_indirect_emissions(data_with_emissions)
    data_with_emissions = calculator.calculate_unit_emissions(data_with_emissions)

    # 训练预测模型
    predictor = CarbonLSTMPredictor()
    history = predictor.train(data_with_emissions, 'total_CO2eq', epochs=30)

    print("模型训练完成并保存")
