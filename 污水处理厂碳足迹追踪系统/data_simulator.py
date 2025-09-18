import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from carbon_calculator import CarbonCalculator


class DataSimulator:
    def __init__(self):
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2024, 12, 31)

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
        seasonal = self.generate_seasonal_pattern(length, 2000, 0)
        trend = self.generate_trend(length, 0.5)  # 缓慢上升趋势
        noise = self.generate_noise(length, 300)
        return base + seasonal + trend + noise

    def generate_energy_consumption(self, water_flow, cod_in, tn_in, length):
        """生成能耗数据（与水量、水质相关）"""
        # 基础能耗系数随水质变化 - 基于实际运行经验
        base_ratio = 0.25  # 基础能耗系数 kWh/m³

        # COD负荷影响：COD浓度越高，生化处理能耗越大
        cod_factor = 1 + (cod_in - 200) / 500  # 以200mg/L为标准
        cod_factor = np.clip(cod_factor, 0.8, 1.5)

        # TN负荷影响：TN浓度越高，硝化反硝化能耗越大
        tn_factor = 1 + (tn_in - 40) / 100  # 以40mg/L为标准
        tn_factor = np.clip(tn_factor, 0.9, 1.3)

        # 季节性变化：冬季能耗高（低温），夏季能耗相对低
        seasonal_var = self.generate_seasonal_pattern(length, 0.08, np.pi)  # 冬高夏低

        # 随机波动
        noise = self.generate_noise(length, 0.03)

        # 综合计算能耗系数
        ratios = base_ratio * cod_factor * tn_factor * (1 + seasonal_var + noise)
        ratios = np.maximum(ratios, 0.15)  # 确保最低能耗

        return water_flow * ratios

    def generate_chemical_usage(self, water_flow, cod_in, ss_in, length):
        """生成药剂使用量数据（与水质关联）"""
        # PAC投加量与悬浮物和COD相关
        pac_base = 0.015  # 基础投加量 kg/m³

        # SS浓度影响PAC用量
        ss_factor = 1 + (ss_in - 150) / 300  # 以150mg/L为标准
        ss_factor = np.clip(ss_factor, 0.7, 1.4)

        # COD影响因子
        cod_factor = 1 + (cod_in - 200) / 800  # 高COD需要更多混凝剂
        cod_factor = np.clip(cod_factor, 0.8, 1.2)

        pac_seasonal = self.generate_seasonal_pattern(length, 0.003, np.pi / 3)
        pac_noise = self.generate_noise(length, 0.002)
        pac_ratio = pac_base * ss_factor * cod_factor * (1 + pac_seasonal + pac_noise)
        pac_usage = water_flow * pac_ratio

        # PAM投加量主要与污泥产量相关
        pam_base = 0.003  # kg/m³
        sludge_factor = (cod_in / 200) * 0.8 + 0.2  # 污泥产量与COD去除相关
        pam_ratio = pam_base * sludge_factor * (1 + self.generate_noise(length, 0.0008))
        pam_usage = water_flow * pam_ratio

        # 次氯酸钠投加量与出水消毒需求相关
        naclo_base = 0.008  # kg/m³
        # 夏季细菌活跃，消毒剂用量增加
        naclo_seasonal = self.generate_seasonal_pattern(length, 0.002, 0)  # 夏高冬低
        naclo_ratio = naclo_base * (1 + naclo_seasonal + self.generate_noise(length, 0.001))
        naclo_usage = water_flow * naclo_ratio

        return pac_usage, pam_usage, naclo_usage

    def generate_water_quality(self, length):
        """生成具有相关性的水质数据"""
        # 进水COD - 考虑季节性和工业排放模式
        cod_in_base = 220  # 调整为更合理的基础值
        # 春季雨水稀释，夏季浓度高，秋冬工业排放增加
        cod_in_seasonal = (self.generate_seasonal_pattern(length, 25, np.pi / 6) +
                           self.generate_seasonal_pattern(length, 15, np.pi))
        cod_in_trend = self.generate_trend(length, -0.03)  # 逐年改善
        cod_in_noise = self.generate_noise(length, 12)
        cod_in = cod_in_base + cod_in_seasonal + cod_in_trend + cod_in_noise
        cod_in = np.maximum(cod_in, 80)  # 确保最低浓度

        # 出水COD - 基于去除效率和工艺稳定性
        base_removal = 0.88  # 基础去除率88%
        # 进水浓度高时去除率略有提升（生物活性好）
        concentration_effect = np.minimum(0.05, (cod_in - 200) / 2000)
        # 工艺改进趋势
        improvement_trend = self.generate_trend(length, 0.0008)
        # 运行稳定性波动
        stability_noise = self.generate_noise(length, 0.02)

        actual_removal = base_removal + concentration_effect + improvement_trend + stability_noise
        actual_removal = np.clip(actual_removal, 0.80, 0.95)  # 限制在合理范围

        cod_out = cod_in * (1 - actual_removal) + self.generate_noise(length, 3)
        cod_out = np.maximum(cod_out, 10)  # 确保最低出水浓度

        # 进水TN - 与COD有一定相关性
        tn_cod_ratio = 0.18 + self.generate_noise(length, 0.02)  # TN/COD比值约0.18
        tn_cod_ratio = np.clip(tn_cod_ratio, 0.12, 0.25)
        tn_in_base = cod_in * tn_cod_ratio
        tn_in_seasonal = self.generate_seasonal_pattern(length, 6, np.pi / 4)
        tn_in = tn_in_base + tn_in_seasonal + self.generate_noise(length, 2.5)
        tn_in = np.maximum(tn_in, 15)  # 确保最低浓度

        # 出水TN - 脱氮效果受温度影响显著
        base_tn_removal = 0.75
        # 温度影响：夏季脱氮效果好，冬季差
        temp_effect = self.generate_seasonal_pattern(length, 0.08, 0)  # 夏高冬低
        # 进水浓度影响
        load_effect = np.minimum(0.05, (tn_in - 40) / 400)

        tn_removal = base_tn_removal + temp_effect + load_effect + self.generate_noise(length, 0.03)
        tn_removal = np.clip(tn_removal, 0.65, 0.85)

        tn_out = tn_in * (1 - tn_removal) + self.generate_noise(length, 1.2)
        tn_out = np.maximum(tn_out, 8)  # 确保最低出水浓度

        return cod_in, cod_out, tn_in, tn_out

    def generate_simulated_data(self, save_path="data/simulated_data.csv"):
        """生成完整的模拟数据集"""
        # 修改日期范围：2018-2024年
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2024, 12, 31)

        date_range = pd.date_range(self.start_date, self.end_date)
        length = len(date_range)

        # 生成相关联的指标数据
        water_flow = self.generate_water_flow(length)
        cod_in, cod_out, tn_in, tn_out = self.generate_water_quality(length)

        # 生成SS数据（与COD相关）
        ss_in_base = cod_in * 0.7 + self.generate_noise(length, 15)  # SS通常为COD的0.7倍左右
        ss_in = np.maximum(ss_in_base, 50)
        ss_out = ss_in * 0.08 + self.generate_noise(length, 2)  # 92%去除率
        ss_out = np.maximum(ss_out, 5)

        # 基于水质生成能耗和药剂用量（体现相关性）
        energy_consumption = self.generate_energy_consumption(water_flow, cod_in, tn_in, length)
        pac_usage, pam_usage, naclo_usage = self.generate_chemical_usage(water_flow, cod_in, ss_in, length)

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
