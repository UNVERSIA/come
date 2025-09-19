import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from carbon_calculator import CarbonCalculator


class DataSimulator:
    def __init__(self, random_seed=42):
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        # 设置随机种子确保数据一致性，但允许用户修改
        np.random.seed(random_seed)

    def _create_monthly_data(self, daily_df):
        """将日度数据聚合为月度数据"""
        df = daily_df.copy()
        df['日期'] = pd.to_datetime(df['日期'])

        # 设置日期为索引
        df.set_index('日期', inplace=True)

        # 按月聚合 - 使用平均值更适合预测
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
            'total_CO2eq': 'mean',
            '自来水(m³/d)': 'mean',
            '脱水污泥外运量(80%)': 'mean'
        }).reset_index()

        # 添加年月标识列
        monthly_df['年月'] = monthly_df['日期'].dt.strftime('%Y年%m月')

        return monthly_df

    def generate_seasonal_pattern(self, length, amplitude, phase=0):
        """生成季节性模式"""
        x = np.arange(length)
        seasonal = amplitude * np.sin(2 * np.pi * x / 365 + phase)
        return seasonal

    def generate_trend(self, length, slope):
        """生成趋势成分"""
        return slope * np.arange(length)

    def generate_noise(self, length, scale):
        """生成噪声成分"""
        return np.random.normal(0, scale, length)

    def generate_water_flow(self, length):
        """生成处理水量数据"""
        base = 10000  # 基础水量
        seasonal = self.generate_seasonal_pattern(length, 1500, 0)  # 减小季节波动
        trend = self.generate_trend(length, 0.2)  # 减小趋势增长
        noise = self.generate_noise(length, 200)  # 减小随机噪声
        return np.maximum(base + seasonal + trend + noise, 5000)  # 确保最小值

    def generate_energy_consumption(self, water_flow, cod_in, tn_in, length):
        """生成能耗数据（与水量、水质相关）"""
        # 确保输入参数都是数组形式
        if not isinstance(cod_in, np.ndarray):
            cod_in = np.full(length, cod_in)
        if not isinstance(tn_in, np.ndarray):
            tn_in = np.full(length, tn_in)
        if not isinstance(water_flow, np.ndarray):
            water_flow = np.full(length, water_flow)

        # 基础能耗系数随水质变化 - 基于实际运行经验
        base_ratio = 0.30  # 提高基础能耗系数 kWh/m³

        # COD负荷影响：COD浓度越高，生化处理能耗越大
        cod_factor = 1 + (cod_in - 200) / 800  # 减小COD影响系数
        cod_factor = np.clip(cod_factor, 0.85, 1.3)

        # TN负荷影响：TN浓度越高，硝化反硝化能耗越大
        tn_factor = 1 + (tn_in - 40) / 150  # 减小TN影响系数
        tn_factor = np.clip(tn_factor, 0.9, 1.2)

        # 季节性变化：冬季能耗高（低温），夏季能耗相对低
        seasonal_var = self.generate_seasonal_pattern(length, 0.05, np.pi)  # 减小季节变化

        # 随机波动
        noise = self.generate_noise(length, 0.02)  # 减小随机波动

        # 综合计算能耗系数
        ratios = base_ratio * cod_factor * tn_factor * (1 + seasonal_var + noise)
        ratios = np.clip(ratios, 0.20, 0.45)  # 限制能耗系数范围

        return water_flow * ratios

    def generate_chemical_usage(self, water_flow, length):
        """生成药剂使用量数据"""
        # PAC投加量 (与水量和季节相关)
        pac_base = 0.02  # kg/m³
        pac_seasonal = self.generate_seasonal_pattern(length, 0.005, np.pi)
        pac_ratio = pac_base + pac_seasonal + self.generate_noise(length, 0.002)
        pac_usage = water_flow * pac_ratio

        # PAM投加量
        pam_base = 0.005  # kg/m³
        pam_ratio = pam_base + self.generate_noise(length, 0.001)
        pam_usage = water_flow * pam_ratio

        # 次氯酸钠投加量
        naclo_base = 0.01  # kg/m³
        naclo_ratio = naclo_base + self.generate_seasonal_pattern(length, 0.002, np.pi / 4)
        naclo_usage = water_flow * naclo_ratio

        return pac_usage, pam_usage, naclo_usage

    def generate_water_quality(self, length):
        """生成水质数据"""
        # 进水COD - 有季节性变化和缓慢改善趋势
        cod_in_base = 250
        cod_in_seasonal = self.generate_seasonal_pattern(length, 30, np.pi / 3)
        cod_in_trend = self.generate_trend(length, -0.05)  # 逐年改善
        cod_in_noise = self.generate_noise(length, 10)
        cod_in = cod_in_base + cod_in_seasonal + cod_in_trend + cod_in_noise

        # 出水COD - 处理效果逐年改善
        removal_efficiency = 0.85 + self.generate_trend(length, 0.001)  # 效率逐年提高
        cod_out = cod_in * (1 - removal_efficiency) + self.generate_noise(length, 5)

        # 进水TN
        tn_in_base = 40
        tn_in_seasonal = self.generate_seasonal_pattern(length, 8, np.pi / 2)
        tn_in_trend = self.generate_trend(length, -0.03)
        tn_in_noise = self.generate_noise(length, 3)
        tn_in = tn_in_base + tn_in_seasonal + tn_in_trend + tn_in_noise

        # 出水TN
        tn_removal = 0.75 + self.generate_trend(length, 0.002)
        tn_out = tn_in * (1 - tn_removal) + self.generate_noise(length, 1.5)

        return cod_in, cod_out, tn_in, tn_out

    def generate_simulated_data(self, save_path="data/simulated_data.csv"):
        """生成完整的模拟数据集"""
        # 修改日期范围：2018-2024年
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2024, 12, 31)

        date_range = pd.date_range(self.start_date, self.end_date)
        length = len(date_range)

        # 生成各指标数据 - 调整顺序，先生成水质数据
        water_flow = self.generate_water_flow(length)
        cod_in, cod_out, tn_in, tn_out = self.generate_water_quality(length)
        energy_consumption = self.generate_energy_consumption(water_flow, cod_in, tn_in, length)
        pac_usage, pam_usage, naclo_usage = self.generate_chemical_usage(water_flow, length)

        # 构建DataFrame - 确保包含LSTM预测器所需的所有列
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
            "出水TN(mg/L)": np.round(tn_out, 1),
            # 添加一些可能用到的其他列
            "自来水(m³/d)": np.round(water_flow * 0.05),  # 假设自来水用量为处理水量的5%
            "脱水污泥外运量(80%)": np.round(water_flow * 0.001)  # 假设污泥产量为处理水量的0.1%
        }

        df = pd.DataFrame(data)

        # 确保没有NaN值
        df = df.fillna(0)

        # 确保所有数值都是正数
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].abs()

        # 计算碳排放数据（确保包含total_CO2eq列）
        try:
            calculator = CarbonCalculator()
            df_with_emissions = calculator.calculate_direct_emissions(df)
            df_with_emissions = calculator.calculate_indirect_emissions(df_with_emissions)
            df_with_emissions = calculator.calculate_unit_emissions(df_with_emissions)
            df = df_with_emissions
        except Exception as e:
            print(f"计算碳排放数据时出错: {e}")
            # 如果计算失败，添加一个默认的total_CO2eq列
            df['total_CO2eq'] = df['电耗(kWh)'] * 0.5 + df['处理水量(m³)'] * 0.1

        # 新增：创建月度聚合数据
        df_monthly = self._create_monthly_data(df)

        # 保存到文件
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        df_monthly.to_csv(save_path.replace('.csv', '_monthly.csv'), index=False, encoding='utf-8')

        print(f"模拟数据已生成并保存到 {save_path}，共 {len(df)} 条记录")
        print(f"月度数据已生成并保存到 {save_path.replace('.csv', '_monthly.csv')}，共 {len(df_monthly)} 条记录")

        return df


# 使用示例
if __name__ == "__main__":
    simulator = DataSimulator()
    simulated_data = simulator.generate_simulated_data()
