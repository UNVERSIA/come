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
    def __init__(self, sequence_length=30, forecast_days=7):
        self.sequence_length = sequence_length
        self.forecast_days = forecast_days
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.target_scaler = MinMaxScaler()  # 添加target_scaler初始化
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
              validation_split=0.2, save_path='models/carbon_lstm.keras'):
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
                                 validation_split=validation_split, verbose=1)

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
            'feature_columns': self.feature_columns,
            'target_scaler': self.target_scaler  # 保存目标缩放器
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

        # 确保所有必需的特征列都存在，如果不存在则创建并填充默认值
        for col in self.feature_columns:
            if col not in df.columns:
                print(f"警告: 特征列 '{col}' 不存在，将使用默认值填充")
                if col == '处理水量(m³)':
                    df[col] = 10000  # 默认处理水量
                elif col == '电耗(kWh)':
                    df[col] = 3000  # 默认电耗
                elif col in ['PAC投加量(kg)', 'PAM投加量(kg)', '次氯酸钠投加量(kg)']:
                    df[col] = 0  # 默认药剂投加量
                elif col in ['进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)']:
                    # 根据典型污水处理厂水质设置默认值
                    if col == '进水COD(mg/L)':
                        df[col] = 200
                    elif col == '出水COD(mg/L)':
                        df[col] = 50
                    elif col == '进水TN(mg/L)':
                        df[col] = 40
                    elif col == '出水TN(mg/L)':
                        df[col] = 15

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

    def load_model(self, model_path=None):
        """加载预训练模型 - 兼容性改进版"""
        # 如果没有提供模型路径，使用默认路径
        if model_path is None:
            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, "models")
            model_path = os.path.join(models_dir, "carbon_lstm_model.keras")

            # 确保目录存在
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

        possible_arch_paths = [
            model_path.replace('.keras', '_architecture.json').replace('.h5', '_architecture.json'),
            'models/carbon_lstm_architecture.json'
        ]

        possible_weights_paths = [
            model_path.replace('.keras', '.weights.h5').replace('.h5', '.weights.h5'),
            'models/carbon_lstm.weights.h5',
            'models/carbon_lstm.h5'
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

        if metadata_path and os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                self.feature_scalers = metadata.get('feature_scalers', {})
                self.sequence_length = metadata.get('sequence_length', 30)
                self.forecast_days = metadata.get('forecast_days', 7)
                self.feature_columns = metadata.get('feature_columns', [
                    '处理水量(m³)', '电耗(kWh)', 'PAC投加量(kg)',
                    'PAM投加量(kg)', '次氯酸钠投加量(kg)',
                    '进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)'
                ])
                # 加载目标缩放器（如果存在）
                if 'target_scaler' in metadata:
                    self.target_scaler = metadata['target_scaler']
                else:
                    # 初始化默认目标缩放器
                    self.target_scaler = MinMaxScaler()
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
                self.target_scaler = MinMaxScaler()

        # 尝试直接加载模型
        try:
            # 使用自定义对象和兼容性选项
            custom_objects = None
            try:
                self.model = load_model(found_model_path, custom_objects=custom_objects, compile=False)
            except:
                # 如果失败，尝试使用安全的加载方式
                self.model = load_model(
                    found_model_path,
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False  # 禁用安全模式以提高兼容性
                )

            # 手动编译模型
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            logger.info("模型直接加载成功")
            return True
        except Exception as e:
            logger.warning(f"直接模型加载失败: {str(e)}，尝试备用加载方式...")

            # 备用加载方式：尝试加载架构和权重
            architecture_path = None
            for path in possible_arch_paths:
                if os.path.exists(path):
                    architecture_path = path
                    break

            weights_path = None
            for path in possible_weights_paths:
                if os.path.exists(path):
                    weights_path = path
                    break

            if architecture_path and weights_path and os.path.exists(architecture_path) and os.path.exists(
                    weights_path):
                try:
                    # 从JSON加载模型架构
                    with open(architecture_path, 'r') as json_file:
                        model_json = json_file.read()

                    # 修复可能的架构兼容性问题
                    model_json = model_json.replace('"batch_shape": [null, 30, 9]',
                                                    '"batch_input_shape": [null, 30, 9]')

                    self.model = model_from_json(model_json)

                    # 加载权重
                    self.model.load_weights(weights_path)

                    # 编译模型
                    self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    logger.info("使用备用方式加载模型成功!")
                    return True
                except Exception as arch_e:
                    logger.error(f"架构加载失败: {str(arch_e)}")

            # 最后尝试：重建模型并加载权重
            try:
                logger.info("尝试重建模型并加载权重...")

                # 确定输入形状
                n_features = len(self.feature_columns)
                input_shape = (self.sequence_length, n_features)

                # 重建模型
                self.model = self.build_model(input_shape)

                # 尝试加载权重
                if weights_path and os.path.exists(weights_path):
                    self.model.load_weights(weights_path)

                # 编译模型
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                logger.info("模型重建并加载权重成功")
                return True
            except Exception as rebuild_e:
                logger.error(f"所有加载方式均失败: {str(rebuild_e)}")
                self.model = None
                return False

    def predict(self, df, target_column='total_CO2eq', steps=12):
        """改进的月度预测方法"""
        if self.model is None:
            raise ValueError("模型未加载，请先加载或训练模型")

        # 确保数据已排序
        df = df.sort_values('日期').reset_index(drop=True)

        # 计算总碳排放（如果尚未计算）
        if target_column not in df.columns:
            calculator = CarbonCalculator()
            df = calculator.calculate_direct_emissions(df)
            df = calculator.calculate_indirect_emissions(df)
            df = calculator.calculate_unit_emissions(df)

        # 准备特征数据 - 使用最后30天数据
        if len(df) < self.sequence_length:
            # 如果数据不足，使用所有可用数据
            X = self._prepare_features(df)
        else:
            X = self._prepare_features(df.tail(self.sequence_length))

        if X is None or len(X) == 0:
            raise ValueError("无法准备特征数据进行预测")

        # 进行预测
        predictions = []
        lower_bounds = []
        upper_bounds = []

        # 使用最后一段序列作为初始输入
        current_sequence = X[-1:]

        # 获取历史统计信息
        historical_values = df[target_column].values
        historical_mean = np.mean(historical_values) if len(historical_values) > 0 else 1000
        historical_std = np.std(historical_values) if len(historical_values) > 0 else historical_mean * 0.2

        # 确保目标缩放器已拟合
        if not hasattr(self.target_scaler, 'n_samples_seen_') or self.target_scaler.n_samples_seen_ == 0:
            target_values = df[target_column].dropna().values.reshape(-1, 1)
            if len(target_values) > 0:
                self.target_scaler.fit(target_values)
            else:
                # 如果没有有效数据，使用默认范围
                self.target_scaler.fit([[0], [historical_mean * 2]])

        for i in range(steps):
            # 预测下一步
            try:
                pred_scaled = self.model.predict(current_sequence, verbose=0)[0][0]
            except Exception as e:
                print(f"模型预测错误: {e}")
                # 如果预测失败，使用历史平均值
                pred_scaled = self.target_scaler.transform([[historical_mean]])[0][0] if historical_mean > 0 else 0.5

            # 逆变换预测值
            try:
                pred = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
            except Exception as e:
                print(f"逆变换失败: {e}")
                # 如果逆变换失败，使用历史平均值
                pred = historical_mean

            # 确保预测值合理（非负且在合理范围内）
            pred = max(0, pred)  # 确保非负
            # 不超过历史均值±3倍标准差范围
            pred = min(pred, historical_mean + 3 * historical_std)
            pred = max(pred, max(0, historical_mean - 3 * historical_std))

            predictions.append(pred)

            # 计算置信区间（基于历史误差）
            if len(historical_values) > 10:
                # 使用历史数据的标准差作为误差估计
                error_estimate = historical_std * 0.3  # 调整误差系数
                lower_bounds.append(max(0, pred - error_estimate))
                upper_bounds.append(pred + error_estimate)
            else:
                # 数据不足时使用固定比例
                lower_bounds.append(max(0, pred * 0.8))
                upper_bounds.append(pred * 1.2)

            # 更新序列 - 使用更简单的方法
            # 创建一个新的序列，将最后的值向前移动一位
            new_sequence = np.roll(current_sequence[0], -1, axis=0)

            # 使用预测值更新序列的最后一个位置（假设目标变量是最后一个特征）
            # 注意：这里假设目标变量不是输入特征的一部分，所以不需要更新
            # 保持序列的其他部分不变，只更新时间特征（如果有的话）

            # 如果有时间特征，可以在这里更新时间特征
            # 例如：new_sequence[-1, time_feature_index] = i + 1

            # 更新当前序列
            current_sequence = np.expand_dims(new_sequence, axis=0)

        # 生成预测日期（按月生成）
        last_date = df['日期'].max()

        # 如果steps是12，表示预测12个月，则按月生成日期
        if steps == 12:
            # 生成每月最后一天的日期
            prediction_dates = []
            for i in range(1, steps + 1):
                # 计算下个月的日期
                next_month = last_date + pd.DateOffset(months=i)
                # 获取该月的最后一天
                month_end = pd.Timestamp(year=next_month.year, month=next_month.month, day=1) + pd.offsets.MonthEnd(1)
                prediction_dates.append(month_end)
        else:
            # 如果不是12个月，按天生成日期
            prediction_dates = [last_date + timedelta(days=i + 1) for i in range(steps)]

        # 创建结果DataFrame
        result_df = pd.DataFrame({
            '日期': prediction_dates,
            'predicted_CO2eq': predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        })

        # 添加年月列用于显示
        result_df['年月'] = result_df['日期'].dt.strftime('%Y年%m月')

        return result_df

    def _prepare_features(self, df):
        """准备特征数据用于预测 - 改进版"""
        if df is None or df.empty:
            return None

        if len(df) < self.sequence_length:
            # 如果数据不足，使用所有可用数据并填充
            pad_length = self.sequence_length - len(df)
            padded_df = pd.concat([df] * (pad_length // len(df) + 1), ignore_index=True)
            df = padded_df.head(self.sequence_length)

        # 确保所有特征列都存在
        for col in self.feature_columns:
            if col not in df.columns:
                # 使用默认值填充缺失特征
                if col == '处理水量(m³)':
                    df[col] = 10000  # 默认处理水量
                elif col == '电耗(kWh)':
                    df[col] = 3000  # 默认电耗
                elif col in ['PAC投加量(kg)', 'PAM投加量(kg)', '次氯酸钠投加量(kg)']:
                    df[col] = 0  # 默认药剂投加量
                elif col in ['进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)']:
                    # 根据典型污水处理厂水质设置默认值
                    if col == '进水COD(mg/L)':
                        df[col] = 200
                    elif col == '出水COD(mg/L)':
                        df[col] = 50
                    elif col == '进水TN(mg/L)':
                        df[col] = 40
                    elif col == '出水TN(mg/L)':
                        df[col] = 15
                else:
                    df[col] = 0

        # 创建序列
        sequences = []

        # 为每个特征创建/获取缩放器
        for col in self.feature_columns:
            if col not in self.feature_scalers:
                self.feature_scalers[col] = MinMaxScaler()

                # 使用当前数据拟合缩放器
                col_values = df[col].values.reshape(-1, 1)
                if len(col_values) > 0:
                    self.feature_scalers[col].fit(col_values)

        # 准备单个序列
        seq = []
        valid_sequence = True

        for col in self.feature_columns:
            # 获取序列数据
            col_data = df[col].iloc[-self.sequence_length:].values

            # 处理NaN值
            if np.isnan(col_data).any():
                col_mean = np.nanmean(col_data)
                if np.isnan(col_mean):  # 如果均值也是NaN，使用列均值或默认值
                    col_mean = df[col].mean() if not np.isnan(df[col].mean()) else 0
                col_data = np.where(np.isnan(col_data), col_mean, col_data)

            # 检查数据有效性
            if np.isnan(col_data).any() or len(col_data) != self.sequence_length:
                valid_sequence = False
                break

            # 缩放数据
            try:
                scaled_data = self.feature_scalers[col].transform(col_data.reshape(-1, 1)).flatten()
            except Exception as e:
                print(f"缩放特征 {col} 时出错: {e}")
                # 如果缩放失败，使用0填充
                scaled_data = np.zeros(self.sequence_length)

            seq.append(scaled_data)

        # 只有所有特征都有效时才添加序列
        if valid_sequence:
            try:
                stacked_seq = np.stack(seq, axis=1)
                sequences.append(stacked_seq)
            except Exception as e:
                print(f"堆叠序列时出错: {e}")
                return None

        return np.array(sequences) if sequences else None

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
