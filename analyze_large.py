#!/usr/bin/env python3
"""分析超大规模实验结果."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np

def main():
    print("🎊 超大规模ACR实验分析")
    print("=" * 50)
    
    # 加载数据
    print("📂 加载超大规模数据...")
    df = pd.read_csv('outputs/run_20250908_164640/events.csv')
    
    # 基本统计
    print(f"📊 数据形状: {df.shape}")
    print(f"💾 内存使用: ~{df.memory_usage(deep=True).sum()/1024/1024:.0f}MB")
    print(f"📅 时间跨度: {df['t'].min()}-{df['t'].max()} 月")
    print(f"👥 借款人数: {df['id'].nunique():,}")
    print(f"⚠️  违约率: {df['default'].mean():.3f}")
    print(f"💰 平均贷款额: ${df['loan'].mean():.0f}")
    print(f"📈 平均DTI: {df['dti'].mean():.3f}")
    
    # 时间序列统计
    print("\n📈 时间序列统计:")
    time_stats = df.groupby('t').agg({
        'default': 'mean',
        'dti': 'mean', 
        'loan': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'n_applications'})
    
    print("前5个月:")
    print(time_stats.head())
    print("\n最后5个月:")
    print(time_stats.tail())
    
    # 周期性分析
    print("\n🔄 周期性分析:")
    print(f"第1年平均违约率: {df[df['t'] <= 12]['default'].mean():.3f}")
    print(f"第15年平均违约率: {df[(df['t'] >= 169) & (df['t'] <= 180)]['default'].mean():.3f}")
    print(f"第30年平均违约率: {df[df['t'] >= 349]['default'].mean():.3f}")
    
    # 画像代理统计
    print("\n🎭 画像代理特征统计:")
    proxy_features = ['night_active_ratio', 'session_std', 'task_completion_ratio', 'spending_volatility']
    for feature in proxy_features:
        if feature in df.columns:
            print(f"{feature}: 均值={df[feature].mean():.3f}, 标准差={df[feature].std():.3f}")
    
    # 借款人行为统计
    print("\n👤 借款人行为统计:")
    borrower_stats = df.groupby('id').agg({
        'default': 'sum',
        't': 'count',
        'loan': 'mean'
    }).rename(columns={'t': 'n_applications', 'default': 'total_defaults'})
    
    print(f"平均申请次数: {borrower_stats['n_applications'].mean():.1f}")
    print(f"平均违约次数: {borrower_stats['total_defaults'].mean():.2f}")
    print(f"从未违约比例: {(borrower_stats['total_defaults'] == 0).mean():.1%}")
    print(f"多次违约比例: {(borrower_stats['total_defaults'] >= 2).mean():.1%}")
    
    print("\n✅ 分析完成！")
    print(f"📊 这是迄今为止最大规模的ACR仿真实验！")

if __name__ == "__main__":
    main()
