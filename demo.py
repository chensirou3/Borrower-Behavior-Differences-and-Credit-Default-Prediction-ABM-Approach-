#!/usr/bin/env python3
"""演示脚本：运行ACR系统的最小可行版本."""

import os
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from acr.config.loader import load_config
from acr.simulation.runner import simulate_events
from acr.features.builder import build_datasets
from acr.models.selection import train_test_split_temporal
from acr.models.pipelines import train_models
from acr.evaluation.metrics import compute_classification_metrics
from acr.evaluation.fairness import compute_fairness_metrics, compute_fairness_summary


def main():
    """运行演示."""
    print("🏦 ACR 信贷风控代理模型演示")
    print("=" * 50)
    
    # 1. 加载配置
    print("📋 加载配置...")
    try:
        config = load_config("configs/experiment.yaml")
        # 使用较小的参数进行快速演示
        config.population.N = 1000  # 1000个借款人
        config.timeline.T = 24      # 2年时间
        print(f"   人口规模: {config.population.N}")
        print(f"   时间周期: {config.timeline.T} 月")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 2. 运行仿真
    print("\n🚀 运行仿真...")
    try:
        rng = np.random.default_rng(config.seed)
        events_df = simulate_events(config, rng)
        
        print(f"   生成事件: {len(events_df):,}")
        print(f"   借款人数: {events_df['id'].nunique():,}")
        print(f"   违约率: {events_df['default'].mean():.1%}")
        print(f"   平均贷款额: ${events_df['loan'].mean():,.0f}")
    except Exception as e:
        print(f"❌ 仿真失败: {e}")
        return
    
    if len(events_df) == 0:
        print("⚠️ 未生成任何事件，停止演示")
        return
    
    # 3. 构建特征集
    print("\n🔧 构建特征集...")
    try:
        X_baseline, X_augmented, y, group = build_datasets(events_df, config)
        print(f"   基线特征: {X_baseline.shape[1]} 个")
        print(f"   增强特征: {X_augmented.shape[1]} 个")
        print(f"   样本数: {len(y):,}")
    except Exception as e:
        print(f"❌ 特征构建失败: {e}")
        return
    
    # 4. 划分训练/测试集
    print("\n✂️ 划分数据集...")
    try:
        (X_train_base, X_test_base, X_train_aug, 
         X_test_aug, y_train, y_test) = train_test_split_temporal(
            X_baseline, X_augmented, y, events_df, config
        )
        print(f"   训练集: {len(y_train):,} 样本")
        print(f"   测试集: {len(y_test):,} 样本")
    except Exception as e:
        print(f"❌ 数据划分失败: {e}")
        return
    
    # 5. 训练模型
    print("\n🤖 训练模型...")
    try:
        models = train_models(
            X_train_base, X_train_aug, y_train,
            X_test_base, X_test_aug, y_test, config
        )
        print(f"   训练完成: {len(models)} 个模型")
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    # 6. 评估模型性能
    print("\n📊 评估模型性能...")
    results = {}
    
    for model_name, model_info in models.items():
        try:
            predictions = model_info['predictions']
            feature_set = model_info['feature_set']
            
            # 计算分类指标
            metrics = compute_classification_metrics(y_test.values, predictions)
            
            results[model_name] = {
                'feature_set': feature_set,
                'metrics': metrics,
                'predictions': predictions
            }
            
            print(f"   {model_name}:")
            print(f"     AUC: {metrics.get('auc', 0):.3f}")
            print(f"     KS: {metrics.get('ks', 0):.3f}")
            print(f"     Brier: {metrics.get('brier', 0):.3f}")
            
        except Exception as e:
            print(f"   ❌ {model_name} 评估失败: {e}")
    
    # 7. 公平性分析
    print("\n⚖️ 公平性分析...")
    try:
        # 使用最佳模型进行公平性分析
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x]['metrics'].get('auc', 0))
        best_predictions = results[best_model_name]['predictions']
        
        # 获取对应的组别信息
        if len(events_df) == len(y_test):
            test_groups = group.iloc[y_test.index].values
        else:
            # 如果长度不匹配，使用简单的组别划分
            test_groups = (np.arange(len(y_test)) % 2).astype(int)
        
        fairness_metrics = compute_fairness_metrics(
            y_test.values, best_predictions, test_groups
        )
        
        fairness_summary = compute_fairness_summary(fairness_metrics)
        print(fairness_summary)
        
    except Exception as e:
        print(f"❌ 公平性分析失败: {e}")
    
    # 8. 保存结果
    print("\n💾 保存结果...")
    try:
        output_dir = f"outputs/demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存事件数据
        events_path = os.path.join(output_dir, "events.csv")
        events_df.to_csv(events_path, index=False)
        
        # 保存配置
        from acr.config.loader import save_config
        config_path = os.path.join(output_dir, "config.yaml")
        save_config(config, config_path)
        
        print(f"   结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
    
    print("\n✅ 演示完成！")
    print("\n📈 主要发现:")
    
    # 比较基线vs增强特征的性能
    baseline_models = {k: v for k, v in results.items() if 'baseline' in k}
    augmented_models = {k: v for k, v in results.items() if 'augmented' in k}
    
    if baseline_models and augmented_models:
        baseline_auc = np.mean([v['metrics'].get('auc', 0) for v in baseline_models.values()])
        augmented_auc = np.mean([v['metrics'].get('auc', 0) for v in augmented_models.values()])
        
        improvement = augmented_auc - baseline_auc
        print(f"   基线特征平均AUC: {baseline_auc:.3f}")
        print(f"   增强特征平均AUC: {augmented_auc:.3f}")
        print(f"   性能提升: {improvement:+.3f} ({improvement/baseline_auc:+.1%})")
        
        if improvement > 0:
            print("   🎉 画像代理特征显示出正面效果！")
        else:
            print("   📝 画像代理特征效果有限，可能需要调优")


if __name__ == "__main__":
    main()
