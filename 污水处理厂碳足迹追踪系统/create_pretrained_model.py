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

    # 训练模型 - 使用新的Keras格式
    predictor = CarbonLSTMPredictor()
    history = predictor.train(
        data_with_emissions,
        'total_CO2eq',
        epochs=50,
        save_path='models/carbon_lstm.keras'  # 使用新的Keras格式
    )

    print("预训练模型已创建并保存到 models/carbon_lstm.keras")

if __name__ == "__main__":
    create_pretrained_model()