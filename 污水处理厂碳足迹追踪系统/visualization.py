import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_heatmap_overlay(emission_data: dict) -> go.Figure:
    """工艺单元碳排热力图"""
    units = list(emission_data.keys())
    emissions = [emission_data[unit] for unit in units]

    fig = go.Figure(data=go.Heatmap(
        z=[emissions],
        x=units,
        y=['碳排放'],
        colorscale='RdBu_r',
        text=[[f"{unit}<br>{e:.1f} kgCO2eq" for e, unit in zip(emissions, units)]],
        hoverinfo="text",
        showscale=True,
        colorbar=dict(title='碳排放 (kgCO2eq)')
    ))

    annotations = []
    for i, emission in enumerate(emissions):
        annotations.append(dict(
            x=i, y=0, text=f"{emission:.1f}", showarrow=False,
            font=dict(color='black', size=12)
        ))

    fig.update_layout(
        title="工艺单元碳排放热力图",
        title_font=dict(size=24, family="Arial", color="black"),
        xaxis_title="工艺单元",
        yaxis_title="",
        font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)",
        height=400,
        annotations=annotations
    )

    fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))

    return fig


def create_sankey_diagram(df: pd.DataFrame) -> go.Figure:
    """碳流动态追踪图谱"""
    if df.empty:
        return go.Figure()

    labels = [
        "电耗", "PAC", "PAM", "次氯酸钠",
        "预处理区", "生物处理区", "深度处理区", "泥处理区", "出水区", "除臭系统",
        "N2O排放", "CH4排放", "能耗间接排放", "药耗间接排放"
    ]

    energy = df['energy_CO2eq'].sum()
    pac = df['PAC_CO2eq'].sum() if 'PAC_CO2eq' in df else 0
    pam = df['PAM_CO2eq'].sum() if 'PAM_CO2eq' in df else 0
    naclo = df['NaClO_CO2eq'].sum() if 'NaClO_CO2eq' in df else 0

    source = [0, 0, 0, 0, 0, 0, 1, 2, 3]
    target = [4, 5, 6, 7, 8, 9, 6, 6, 6]
    value = [
        energy * 0.3193, energy * 0.4453, energy * 0.1155,
        energy * 0.0507, energy * 0.0672, energy * 0.0267,
        pac, pam, naclo
    ]

    n2o_emission = df['N2O_CO2eq'].sum()
    ch4_emission = df['CH4_CO2eq'].sum()

    if n2o_emission > 0:
        source.append(5)
        target.append(10)
        value.append(n2o_emission)

    if ch4_emission > 0:
        source.append(5)
        target.append(11)
        value.append(ch4_emission)

    valid_indices = [i for i, v in enumerate(value) if v > 0]
    source = [source[i] for i in valid_indices]
    target = [target[i] for i in valid_indices]
    value = [value[i] for i in valid_indices]

    node_colors = [
        "#FFD700", "#FFA500", "#FF6347", "#FF4500",
        "#1E90FF", "#4169E1", "#4682B4", "#5F9EA0", "#1abc9c", "#1abc9c",
        "#32CD32", "#228B22", "#2E8B57", "#3CB371"
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20, thickness=30, line=dict(color="black", width=1.5),
            label=labels, color=node_colors[:len(labels)],
            hovertemplate='%{label}<br>%{value} kgCO2eq',
            hoverlabel=dict(font=dict(color='black', size=12))
        ),
        link=dict(
            source=source, target=target, value=value,
            color="rgba(150, 150, 150, 0.5)",
            hovertemplate='源: %{source.label}<br>目标: %{target.label}<br>流量: %{value} kgCO2eq'
        )
    )])

    fig.update_layout(
        title="碳流动态追踪图谱",
        title_font=dict(size=24, family="Arial", color="black"),
        height=500, font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)"
    )

    return fig


def create_efficiency_ranking(df: pd.DataFrame) -> go.Figure:
    """碳排放效率排行榜"""
    if df.empty:
        return go.Figure()

    efficiency_data = {}
    units = ["预处理区", "生物处理区", "深度处理区", "泥处理区", "出水区", "除臭系统"]
    unit_cols = ['pre_CO2eq', 'bio_CO2eq', 'depth_CO2eq', 'sludge_CO2eq', 'effluent_CO2eq', 'deodorization_CO2eq']

    total_water = df['处理水量(m³)'].sum()
    if total_water > 0:
        for unit, col in zip(units, unit_cols):
            total_emission = df[col].sum()
            efficiency_data[unit] = total_water / total_emission if total_emission > 0 else 0

    df_eff = pd.DataFrame(efficiency_data.items(), columns=["工艺单元", "效率（m³/kgCO2eq）"])
    df_eff = df_eff.sort_values("效率（m³/kgCO2eq）", ascending=False)

    fig = px.bar(
        df_eff, x="工艺单元", y="效率（m³/kgCO2eq）", title="碳排放效率排行榜",
        color="效率（m³/kgCO2eq）", color_continuous_scale="Tealgrn",
        text="效率（m³/kgCO2eq）", height=500
    )

    fig.update_traces(
        texttemplate='%{text:.2f}', textposition='outside',
        marker_line_color='rgb(8,48,107)', marker_line_width=1.5,
        textfont=dict(color="black", size=12)
    )

    fig.update_layout(
        title_font=dict(size=24, family="Arial", color="black"),
        xaxis_title="工艺单元", yaxis_title="碳排放效率 (m³/kgCO2eq)",
        font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)",
        showlegend=False,
        xaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        yaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black"))
    )

    fig.update_coloraxes(
        colorbar_title_text="效率（m³/kgCO2eq）",
        colorbar_title_font_color="black",
        colorbar_tickfont=dict(color="black")
    )

    if not df_eff.empty:
        avg_efficiency = df_eff["效率（m³/kgCO2eq）"].mean()
        fig.add_hline(
            y=avg_efficiency, line_dash="dash", line_color="red", line_width=2,
            annotation_text=f"平均效率: {avg_efficiency:.2f}",
            annotation_position="bottom right",
            annotation_font_size=14, annotation_font_color="black"
        )

    return fig


def create_carbon_trend_chart(historical_data, predicted_data=None):
    """创建碳排放趋势与预测图表"""
    fig = go.Figure()

    # 历史数据
    fig.add_trace(go.Scatter(
        x=historical_data['日期'], y=historical_data['total_CO2eq'],
        mode='lines+markers', name='历史碳排放',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))

    # 预测数据（如果有）
    if predicted_data is not None and not predicted_data.empty:
        # 确保预测数据按日期排序（解决线条异常问题）
        predicted_data = predicted_data.sort_values('日期')

        # 处理可能的列名差异
        pred_col = 'predicted_CO2eq' if 'predicted_CO2eq' in predicted_data.columns else '预测碳排放'
        lower_col = 'lower_bound' if 'lower_bound' in predicted_data.columns else '下限'
        upper_col = 'upper_bound' if 'upper_bound' in predicted_data.columns else '上限'

        # 修正预测数据显示问题
        fig.add_trace(go.Scatter(
            x=predicted_data['日期'], y=predicted_data[pred_col],
            mode='lines+markers', name='预测碳排放',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6)
        ))

        # 添加置信区间（如果有）
        if lower_col in predicted_data.columns and upper_col in predicted_data.columns:
            fig.add_trace(go.Scatter(
                x=predicted_data['日期'], y=predicted_data[upper_col],
                mode='lines', line=dict(width=0), showlegend=False,
                name='预测上限'
            ))
            fig.add_trace(go.Scatter(
                x=predicted_data['日期'], y=predicted_data[lower_col],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)', showlegend=False,
                name='预测下限'
            ))

    # 使用更灵活的布局设置
    fig.update_layout(
        title="碳排放趋势与预测",
        title_font=dict(size=24, family="Arial", color="black"),
        xaxis_title="日期", yaxis_title="碳排放 (kgCO2eq)",
        font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)",
        height=500,
        # 移除固定宽度设置，使用自动调整
        width=None,
        autosize=True,
        margin=dict(l=50, r=50, b=80, t=100, pad=10),
        xaxis=dict(
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            tickangle=-45,
            tickformat="%m-%d",
            # 确保x轴标签不会重叠
            tickmode='auto',
            nticks=20  # 设置最大刻度数
        ),
        yaxis=dict(
            tickfont=dict(color="black"),
            title_font=dict(color="black")
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.5)'
        )
    )

    # 添加响应式配置
    fig.update_layout(
        autosize=True,
        width=None  # 让图表自动填充可用空间
    )

    return fig

def create_technology_comparison(tech_data):
    """创建碳减排技术对比图"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=tech_data['技术名称'], y=tech_data['减排量_kgCO2eq'],
        name='减排量', marker_color='#2ca02c',
        text=tech_data['减排量_kgCO2eq'], textposition='auto',
        texttemplate='%{text:.0f}'
    ))

    fig.add_trace(go.Bar(
        x=tech_data['技术名称'], y=tech_data['投资成本_万元'],
        name='投资成本', marker_color='#d62728', yaxis='y2',
        text=tech_data['投资成本_万元'], textposition='auto',
        texttemplate='%{text:.0f}'
    ))

    fig.update_layout(
        title="碳减排技术对比分析",
        title_font=dict(size=24, family="Arial", color="black"),
        xaxis_title="减排技术", yaxis_title="减排量 (kgCO2eq)",
        font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)",
        height=500,
        xaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
        yaxis=dict(
            tickfont=dict(color="black"), title_font=dict(color="black"),
            title="减排量 (kgCO2eq)"
        ),
        yaxis2=dict(
            tickfont=dict(color="black"), title_font=dict(color="black"),
            title="投资成本 (万元)", overlaying='y', side='right'
        ),
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_sensitivity_analysis_chart(sensitivity_data, param_name):
    """创建参数敏感性分析图表"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sensitivity_data['adjustment_percent'],
        y=sensitivity_data['reduction_percent'],
        mode='lines+markers',
        name='减排率变化',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=sensitivity_data['adjustment_percent'],
        y=sensitivity_data['emission'],
        mode='lines',
        name='碳排放量',
        yaxis='y2',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))

    fig.update_layout(
        title=f"{param_name}参数敏感性分析",
        title_font=dict(size=24, family="Arial", color="black"),
        xaxis_title=f"{param_name}调整百分比 (%)",
        yaxis_title="减排率 (%)",
        font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)",
        height=500,
        yaxis=dict(
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            title="减排率 (%)"
        ),
        yaxis2=dict(
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            title="碳排放量 (kgCO2eq)",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_carbon_offset_chart(carbon_offset_data):
    """创建碳抵消技术贡献图"""
    technologies = list(carbon_offset_data.keys())
    values = list(carbon_offset_data.values())
    total = sum(values)

    # 计算百分比
    percentages = [v / total * 100 for v in values]

    fig = go.Figure(data=[go.Pie(
        labels=technologies,
        values=values,
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{percent} (%{value:.1f} kgCO2eq)',
        hovertemplate='<b>%{label}</b><br>贡献: %{percent}<br>减排量: %{value} kgCO2eq',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])

    fig.update_layout(
        title="碳抵消技术贡献分析",
        title_font=dict(size=24, family="Arial", color="black"),
        font=dict(size=14, color="black"),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig


def create_optimization_comparison(optimization_results):
    """创建优化方案对比图"""
    strategies = list(optimization_results.keys())
    reductions = [result['减排率_%'] for result in optimization_results.values()]
    emissions = [result['优化后排放_kgCO2eq'] for result in optimization_results.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=strategies,
        y=reductions,
        name='减排率',
        marker_color='#2ca02c',
        text=[f"{r:.1f}%" for r in reductions],
        textposition='auto'
    ))

    fig.add_trace(go.Scatter(
        x=strategies,
        y=emissions,
        name='优化后排放',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="优化方案效果对比",
        title_font=dict(size=24, family="Arial", color="black"),
        xaxis_title="优化策略",
        yaxis_title="减排率 (%)",
        font=dict(size=14, color="black"),
        plot_bgcolor="rgba(245, 245, 245, 1)",
        paper_bgcolor="rgba(245, 245, 245, 1)",
        height=500,
        yaxis=dict(
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            title="减排率 (%)"
        ),
        yaxis2=dict(
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            title="碳排放量 (kgCO2eq)",
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98)
    )

    # 添加以下新函数

    def create_real_time_trend(df, time_unit='day'):
        """创建实时碳排放负荷趋势图"""
        if time_unit == 'hour':
            df['时间'] = df['日期'] + pd.to_timedelta(df['小时'], unit='h')
            x_col = '时间'
        else:
            df['日期'] = pd.to_datetime(df['日期'])
            x_col = '日期'

        fig = px.line(df, x=x_col, y='total_CO2eq',
                      title='实时碳排放趋势')

        fig.update_layout(
            xaxis_title="时间",
            yaxis_title="碳排放 (kgCO2eq)",
            hovermode="x unified"
        )

        return fig

    def create_emission_pie_chart(emission_data):
        """创建工艺单元碳排放占比饼图"""
        labels = list(emission_data.keys())
        values = list(emission_data.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            textinfo='label+percent',
            insidetextorientation='radial'
        )])

        fig.update_layout(
            title="工艺单元碳排放占比",
            height=500
        )

        return fig

    def create_indicators_dashboard(indicators):
        """创建关键指标仪表盘"""
        fig = go.Figure()

        # 单位水量碳排放仪表
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=indicators.get('carbon_per_water', 0),
            title={'text': "单位水量碳排放 (kgCO2eq/m³)"},
            domain={'row': 0, 'column': 0},
            gauge={
                'axis': {'range': [0, 2]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "yellow"},
                    {'range': [1, 2], 'color': "red"}
                ]
            }
        ))

        # 能源中和率仪表
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=indicators.get('energy_neutrality', 0),
            number={'suffix': '%'},
            title={'text': "能源中和率"},
            domain={'row': 0, 'column': 1},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))

        fig.update_layout(
            grid={'rows': 1, 'columns': 2, 'pattern': "independent"},
            title="关键性能指标仪表盘",
            height=400
        )

        return fig

    def create_sensitivity_curve(sensitivity_data, param_name):
        """创建参数敏感性分析曲线"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sensitivity_data['adjustment_percent'],
            y=sensitivity_data['reduction_percent'],
            mode='lines+markers',
            name='减排率变化'
        ))

        fig.update_layout(
            title=f"{param_name}参数敏感性分析",
            xaxis_title=f"{param_name}调整百分比 (%)",
            yaxis_title="减排率 (%)",
            hovermode="x unified"
        )

        return fig

    return fig

