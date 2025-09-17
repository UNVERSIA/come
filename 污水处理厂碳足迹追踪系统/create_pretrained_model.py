# create_pretrained_model.py
import pandas as pd
from carbon_calculator import CarbonCalculator
from lstm_predictor import CarbonLSTMPredictor
from data_simulator import DataSimulator
import os


def create_pretrained_model():
    """创建预训练模型"""
    # 确保目录存在
    os.makedirs("models", exist_ok=True)

    # 生成模拟数据
    simulator = DataSimulator()
    data = simulator.generate_simulated_data()

    # 计算碳排放
    calculator = CarbonCalculator()
    data_with_emissions = calculator.calculate_direct_emissions(data)
    data_with_emissions = calculator.calculate_indirect_emissions(data_with_emissions)
    data_with_emissions = calculator.calculate_unit_emissions(data_with_emissions)

    # 确保所有必需的特征列都存在
    predictor = CarbonLSTMPredictor()
    for col in predictor.feature_columns:
        if col not in data_with_emissions.columns:
            if col == '处理水量(m³)':
                data_with_emissions[col] = 10000
            elif col == '电耗(kWh)':
                data_with_emissions[col] = 3000
            elif col in ['PAC投加量(kg)', 'PAM投加量(kg)', '次氯酸钠投加量(kg)']:
                data_with_emissions[col] = 0
            elif col in ['进水COD(mg/L)', '出水COD(mg/L)', '进水TN(mg/L)', '出水TN(mg/L)']:
                if col == '进水COD(mg/L)':
                    data_with_emissions[col] = 200
                elif col == '出水COD(mg/L)':
                    data_with_emissions[col] = 50
                elif col == '进水TN(mg/L)':
                    data_with_emissions[col] = 40
                elif col == '出水TN(mg/L)':
                    data_with_emissions[col] = 15

    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    save_path = os.path.join(models_dir, "carbon_lstm_model.keras")

    # 训练模型 - 使用新的Keras格式
    predictor = CarbonLSTMPredictor()
    history = predictor.train(
        data_with_emissions,
        'total_CO2eq',
        epochs=50,
        save_path=save_path  # 使用绝对路径
    )

    print("预训练模型已创建并保存到 models/carbon_lstm_model.keras")

if __name__ == "__main__":
    create_pretrained_model()
