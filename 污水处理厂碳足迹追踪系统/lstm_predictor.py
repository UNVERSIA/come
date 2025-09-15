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
        self.start_date = pd.Timestamp('2020-01-01')  # 添加默认起始日期
        self.end_date = pd.Timestamp('2022-12-31')  # 添加默认结束日期

    def generate_simulated_data(self, save_path="data/simulated_data.csv"):
        """生成完整的模拟数据集"""
        # 确保生成足够长的数据（至少2年）
        if (self.end_date - self.start_date).days < 730:
            self.end_date = self.start_date + timedelta(days=730)  # 至少2年数据

        date_range = pd.date_range(self.start_date, self.end_date)
        length = len(date_range)

        # 生成各指标数据
        water_flow = self.generate_water_flow(length)
        energy_consumption = self.generate_energy_consumption(water_flow, length)
        pac_usage, pam_usage, naclo_usage = self.generate_chemical_usage(water_flow, length)
        cod_in, cod_out, tn_in, tn_out = self.generate_water_quality(length)

        # 构建DataFrame
        data = {
            "日期": date_range,
            "处理水量(m³)": np.round(water_flow),
            "电耗(kWh)": np.round(energy_consumption),
            "PAC投加量(kg)": np.round(pac_usage),
            "PAM投加量(kg)": np.round(pam_usage),
            "次氯酸钠投加量(kg)": np.round(naclo_usage),
            "进水COD(mg/L)": np.round(cod_in, 1),
            "出水COD(mg/L)": np.round(cod_out, 1),
            "进水TN(mg/L)": np.round(tn_in, 1),
            "出水TN(mg/L)": np.round(tn_out, 1)
        }

        df = pd.DataFrame(data)

        # 保存到文件
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"模拟数据已生成并保存到 {save_path}，共 {len(df)} 条记录")

        return df

    def generate_water_flow(self, length):
        """生成处理水量数据"""
        # 基础水量 + 季节性波动 + 随机噪声
        base_flow = 50000
        seasonal = 5000 * np.sin(np.arange(length) * 2 * np.pi / 365)
        noise = np.random.normal(0, 1000, length)
        return base_flow + seasonal + noise

    def generate_energy_consumption(self, water_flow, length):
        """生成电耗数据"""
        # 与处理水量正相关，但有基础能耗
        base_energy = 5000
        flow_factor = 0.1
        noise = np.random.normal(0, 200, length)
        return base_energy + flow_factor * water_flow + noise

    def generate_chemical_usage(self, water_flow, length):
        """生成化学品投加量数据"""
        # 与处理水量正相关
        pac_factor = 0.01
        pam_factor = 0.002
        naclo_factor = 0.005

        pac_usage = pac_factor * water_flow + np.random.normal(0, 10, length)
        pam_usage = pam_factor * water_flow + np.random.normal(0, 5, length)
        naclo_usage = naclo_factor * water_flow + np.random.normal(0, 8, length)

        return pac_usage, pam_usage, naclo_usage

    def generate_water_quality(self, length):
        """生成水质数据"""
        # 进水COD - 有季节性变化
        cod_in_base = 300
        cod_in_seasonal = 100 * np.sin(np.arange(length) * 2 * np.pi / 365 + np.pi / 4)
        cod_in_noise = np.random.normal(0, 20, length)
        cod_in = cod_in_base + cod_in_seasonal + cod_in_noise

        # 出水COD - 与进水相关但更稳定
        removal_efficiency = 0.85 + np.random.normal(0, 0.05, length)
        cod_out = cod_in * (1 - removal_efficiency)

        # 进水TN
        tn_in_base = 40
        tn_in_seasonal = 10 * np.sin(np.arange(length) * 2 * np.pi / 365 + np.pi / 3)
        tn_in_noise = np.random.normal(0, 5, length)
        tn_in = tn_in_base + tn_in_seasonal + tn_in_noise

        # 出水TN
        tn_removal = 0.7 + np.random.normal(0, 0.1, length)
        tn_out = tn_in * (1 - tn_removal)

        return cod_in, cod_out, tn_in, tn_out

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

    def prepare_training_data(self, df, target_column):
        """准备训练数据 - 修复版"""
        # 确保数据按日期排序
        df = df.sort_values('日期').reset_index(drop=True)

        # 检查目标列是否存在且有有效数据
        if target_column not in df.columns or df[target_column].isna().all():
            raise ValueError(f"目标列 '{target_column}' 不存在或全部为NaN值")

        # 检查是否有足够的数据
        if len(df) < self.sequence_length * 2:
            raise ValueError(f"需要至少 {self.sequence_length * 2} 条记录进行训练，当前只有 {len(df)} 条")

        # 初始化特征数据列表
        X, y = [], []

        # 为每个特征创建单独的缩放器
        self.feature_scalers = {}
        valid_features = []

        # 只处理实际存在的特征列
        for col in self.feature_columns:
            if col in df.columns and not df[col].isna().all():
                self.feature_scalers[col] = MinMaxScaler()
                # 只使用非NaN值进行拟合
                valid_values = df[col].dropna().values.reshape(-1, 1)
                if len(valid_values) > 0:
                    self.feature_scalers[col].fit(valid_values)
                    valid_features.append(col)
            else:
                print(f"警告: 特征列 '{col}' 不存在或全部为NaN值，将跳过")

        # 如果没有有效特征，抛出错误
        if not valid_features:
            raise ValueError("没有有效的特征列可用于训练")

        # 目标变量缩放器
        self.target_scaler = MinMaxScaler()
        target_values = df[target_column].dropna().values.reshape(-1, 1)
        if len(target_values) > 0:
            self.target_scaler.fit(target_values)
        else:
            raise ValueError(f"目标列 '{target_column}' 没有有效值")

        # 创建序列数据
        valid_count = 0
        for i in range(self.sequence_length, len(df)):
            # 检查目标值是否有效
            target = df[target_column].iloc[i]
            if np.isnan(target):
                continue

            # 提取特征序列
            sequence_features = []
            has_valid_data = False

            for col in valid_features:
                # 获取当前特征序列
                col_data = df[col].iloc[i - self.sequence_length:i].values

                # 检查是否有NaN值，如果有则使用均值填充
                if np.isnan(col_data).any():
                    col_mean = np.nanmean(col_data)
                    if np.isnan(col_mean):  # 如果均值也是NaN，使用0
                        col_mean = 0
                    col_data = np.where(np.isnan(col_data), col_mean, col_data)

                # 确保所有值都是有效的
                if np.isnan(col_data).any() or len(col_data) != self.sequence_length:
                    # 如果数据无效，使用0填充整个序列
                    col_data = np.zeros(self.sequence_length)
                else:
                    has_valid_data = True

                # 缩放特征数据
                try:
                    scaled_data = self.feature_scalers[col].transform(col_data.reshape(-1, 1))
                    sequence_features.append(scaled_data.flatten())
                except Exception as e:
                    print(f"缩放特征 {col} 时出错: {e}")
                    # 如果缩放失败，使用0填充
                    scaled_data = np.zeros((len(col_data), 1))
                    sequence_features.append(scaled_data.flatten())

            # 如果没有有效数据，跳过该序列
            if not has_valid_data:
                continue

            # 确保所有特征序列长度一致
            if not all(len(seq) == self.sequence_length for seq in sequence_features):
                continue

            # 缩放目标值
            try:
                scaled_target = self.target_scaler.transform([[target]])[0][0]
            except Exception as e:
                print(f"缩放目标值时出错: {e}")
                continue

            # 堆叠特征序列
            try:
                stacked_sequence = np.stack(sequence_features, axis=1)

                # 添加到数据集
                X.append(stacked_sequence)
                y.append(scaled_target)
                valid_count += 1
            except Exception as e:
                print(f"堆叠序列时出错: {e}")
                continue

        # 检查是否有有效数据
        if valid_count == 0:
            raise ValueError("没有有效的训练序列")

        print(f"成功创建 {valid_count} 个有效训练序列")

        return np.array(X), np.array(y)

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

    def predict(self, df, target_column='total_CO2eq', steps=365):
        """科学的多步预测方法 - 完全重写"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载或训练模型")

        # 0. 数据准备与校验
        df = df.sort_values('日期').reset_index(drop=True)
        if len(df) < self.sequence_length:
            raise ValueError(f"需要至少{self.sequence_length}天的历史数据，当前只有{len(df)}天")
        if '日期' not in df.columns:
            raise ValueError("数据必须包含'日期'列")

        # 1. 准备历史数据 - 使用提供的完整历史数据
        historical_data = df.copy()
        # 确保目标列已计算
        if target_column not in historical_data.columns:
            calculator = CarbonCalculator()
            historical_data = calculator.calculate_direct_emissions(historical_data)
            historical_data = calculator.calculate_indirect_emissions(historical_data)
            historical_data = calculator.calculate_unit_emissions(historical_data)

        # 2. 分析历史趋势和季节性模式
        # 使用完整的多年历史数据分析趋势和季节性
        historical_data['年份'] = historical_data['日期'].dt.year
        historical_data['月份'] = historical_data['日期'].dt.month
        historical_data['日期序号'] = (historical_data['日期'] - historical_data['日期'].min()).dt.days

        # 计算各特征的年均增长率和季节性模式
        feature_trends = {}
        seasonal_patterns = {}

        for col in self.feature_columns:
            if col in historical_data.columns:
                # 计算年均增长率（线性回归斜率）
                X = historical_data['日期序号'].values.reshape(-1, 1)
                y = historical_data[col].values

                # 使用稳健的线性回归（Huber回归）
                from sklearn.linear_model import HuberRegressor
                model = HuberRegressor()
                model.fit(X, y)
                feature_trends[col] = model.coef_[0]  # 每日增长量

                # 计算季节性模式（月度平均值）
                monthly_avg = historical_data.groupby('月份')[col].mean()
                seasonal_patterns[col] = monthly_avg.to_dict()

        # 3. 初始化预测序列：使用最后`sequence_length`天的数据
        current_sequence = historical_data.tail(self.sequence_length).copy()
        all_predictions = []
        all_dates = []

        # 4. 多步预测循环
        for step in range(steps):
            try:
                # 4.1 准备当前时间步的输入
                X_input = []
                missing_features = []

                for col in self.feature_columns:
                    if col in current_sequence.columns and col in self.feature_scalers:
                        col_data = current_sequence[col].values[-self.sequence_length:]

                        # 检查是否有NaN值，如果有则使用均值填充
                        if np.isnan(col_data).any():
                            col_mean = np.nanmean(col_data)
                            col_data = np.where(np.isnan(col_data), col_mean, col_data)

                        # 缩放特征数据
                        try:
                            scaled_data = self.feature_scalers[col].transform(col_data.reshape(-1, 1))
                            X_input.append(scaled_data.flatten())
                        except:
                            # 如果缩放失败，使用0填充
                            scaled_data = np.zeros((self.sequence_length, 1))
                            X_input.append(scaled_data.flatten())
                            missing_features.append(col)
                    else:
                        # 对于模型训练时未见过的特征，使用0填充
                        scaled_data = np.zeros((self.sequence_length, 1))
                        X_input.append(scaled_data.flatten())
                        missing_features.append(col)

                # 记录缺失的特征
                if missing_features:
                    print(f"警告：以下特征缺失或缩放失败，使用0填充: {missing_features}")

                # 确保有数据可堆叠
                if not X_input:
                    raise ValueError("没有可用的特征数据进行预测。")

                # 堆叠特征并调整形状为 [1, sequence_length, num_features]
                X_input = np.stack(X_input, axis=1)
                X_input = X_input.reshape(1, self.sequence_length, len(self.feature_columns))

                # 4.2 进行单步预测
                scaled_prediction = self.model.predict(X_input, verbose=0)[0][0]
                prediction = self.target_scaler.inverse_transform(
                    [[scaled_prediction]]
                )[0][0]
                prediction = max(0, prediction)  # 碳排放不为负
                all_predictions.append(prediction)

                # 4.3 为下一步预测创建新行（科学地更新所有特征）
                last_date = current_sequence['日期'].iloc[-1]
                next_date = last_date + pd.Timedelta(days=1)
                all_dates.append(next_date)

                # 创建一个新的DataFrame行，基于趋势和季节性预测所有特征
                new_row = current_sequence.iloc[-1:].copy()
                new_row['日期'] = next_date

                # 计算下一个日期的特征值（考虑趋势和季节性）
                next_month = next_date.month
                days_since_start = (next_date - historical_data['日期'].min()).days

                for col in self.feature_columns:
                    if col in historical_data.columns:
                        # 基础值：使用最近7天的平均值
                        base_value = np.mean(current_sequence[col].values[-7:])

                        # 添加趋势成分
                        trend_component = 0
                        if col in feature_trends:
                            trend_component = feature_trends[col] * days_since_start

                        # 添加季节性成分
                        seasonal_component = 0
                        if col in seasonal_patterns and next_month in seasonal_patterns[col]:
                            # 使用该月份的历史平均值与全年平均值的差异作为季节性
                            yearly_avg = np.mean(list(seasonal_patterns[col].values()))
                            seasonal_component = seasonal_patterns[col][next_month] - yearly_avg

                        # 添加合理的随机噪声（最大5%）
                        noise_component = np.random.normal(0, base_value * 0.05)

                        # 计算预测值
                        predicted_value = base_value + trend_component + seasonal_component + noise_component
                        predicted_value = max(0, predicted_value)  # 确保值合理

                        new_row[col] = predicted_value

                # 目标列使用模型预测值
                new_row[target_column] = prediction

                # 4.4 将新行追加到序列中，并移除最旧的一行，保持序列长度不变
                current_sequence = pd.concat([current_sequence, new_row], ignore_index=True)
                current_sequence = current_sequence.tail(self.sequence_length).reset_index(drop=True)

            except Exception as e:
                print(f"第{step + 1}步预测失败: {str(e)}")
                # 如果失败，使用更稳健的回退策略
                if all_predictions:
                    # 使用指数平滑回退
                    alpha = 0.3
                    fallback_value = alpha * all_predictions[-1] + (1 - alpha) * historical_data[target_column].mean()
                else:
                    fallback_value = historical_data[target_column].mean()

                all_predictions.append(fallback_value)
                next_date = last_date + pd.Timedelta(days=1) if 'last_date' in locals() else \
                    historical_data['日期'].iloc[-1] + pd.Timedelta(days=step + 1)
                all_dates.append(next_date)

        # 5. 生成预测结果DataFrame
        result_df = pd.DataFrame({
            '日期': all_dates,
            'predicted_CO2eq': all_predictions
        })

        # 6. 计算科学合理的置信区间
        # 使用历史预测误差的标准差，考虑趋势不确定性
        historical_errors = []
        if len(historical_data) > self.sequence_length:
            # 使用历史数据进行回测，计算预测误差
            for i in range(self.sequence_length, len(historical_data)):
                train_data = historical_data.iloc[:i]
                actual_value = historical_data.iloc[i][target_column]

                # 使用最近sequence_length天数据预测
                X_test = []
                test_sequence = train_data.tail(self.sequence_length)

                for col in self.feature_columns:
                    if col in test_sequence.columns and col in self.feature_scalers:
                        col_data = test_sequence[col].values[-self.sequence_length:]
                        scaled_data = self.feature_scalers[col].transform(col_data.reshape(-1, 1))
                        X_test.append(scaled_data.flatten())

                X_test = np.stack(X_test, axis=1)
                X_test = X_test.reshape(1, self.sequence_length, len(self.feature_columns))

                try:
                    scaled_pred = self.model.predict(X_test, verbose=0)[0][0]
                    prediction = self.target_scaler.inverse_transform([[scaled_pred]])[0][0]
                    error = abs(prediction - actual_value) / actual_value if actual_value > 0 else 0
                    historical_errors.append(error)
                except:
                    continue

        # 计算平均相对误差
        if historical_errors:
            mean_error = np.mean(historical_errors)
            std_error = np.std(historical_errors)
        else:
            # 默认误差估计
            mean_error = 0.15  # 15%的平均误差
            std_error = 0.08  # 8%的标准差

        # 置信区间考虑预测步长的增加而扩大（不确定性随时间增加）
        confidence_factors = []
        for i in range(steps):
            # 不确定性随预测步长线性增加
            uncertainty_factor = 1 + (i / steps) * 2  # 最终不确定性增加到3倍
            confidence_factors.append(uncertainty_factor)

        # 计算上下界
        result_df['lower_bound'] = [
            max(0, pred * (1 - mean_error * factor - std_error))
            for pred, factor in zip(result_df['predicted_CO2eq'], confidence_factors)
        ]
        result_df['upper_bound'] = [
            pred * (1 + mean_error * factor + std_error)
            for pred, factor in zip(result_df['predicted_CO2eq'], confidence_factors)
        ]

        return result_df

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
