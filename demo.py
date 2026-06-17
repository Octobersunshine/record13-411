import pandas as pd
from group_ranking import GroupRankingService


def build_sample_data():
    data = {
        '部门': ['技术部'] * 5 + ['销售部'] * 5 + ['市场部'] * 4,
        '姓名': [
            '张三', '李四', '王五', '赵六', '钱七',
            '孙八', '周九', '吴十', '郑十一', '王十二',
            '冯十三', '陈十四', '褚十五', '卫十六',
        ],
        'KPI分数': [95, 88, 95, 76, 92, 88, 88, 95, 80, 83, 90, 90, 85, 88],
        '出勤天数': [22, 20, 22, 21, 19, 22, 21, 22, 20, 18, 22, 21, 20, 22],
    }
    return pd.DataFrame(data)


def demo_basic_rank():
    print("=" * 60)
    print("【演示1】基础分组排名：部门内按 KPI 降序排名（average）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    result = service.rank(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        ascending=False,
        method='average',
    )
    print(result.sort_values(['部门', 'KPI分数_rank']).to_string(index=False))
    print()


def demo_all_methods():
    print("=" * 60)
    print("【演示2】并列处理方式对比（技术部 KPI 降序）")
    print("=" * 60)
    df = build_sample_data()
    df_tech = df[df['部门'] == '技术部'].copy()
    service = GroupRankingService()

    for method in ['average', 'min', 'max', 'dense', 'first']:
        ranked = service.rank(
            df=df_tech,
            group_cols='部门',
            value_col='KPI分数',
            rank_col=f'rank_{method}',
            ascending=False,
            method=method,
        )
        print(f"--- method = '{method}' ---")
        cols = ['姓名', 'KPI分数', f'rank_{method}']
        print(ranked[cols].sort_values(f'rank_{method}').to_string(index=False))
        print()


def demo_multi_rank():
    print("=" * 60)
    print("【演示3】同时对多个列进行排名（KPI降序、出勤升序）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    result = service.rank_multi(
        df=df,
        group_cols='部门',
        rank_configs=[
            {
                'value_col': 'KPI分数',
                'rank_col': 'KPI排名',
                'ascending': False,
                'method': 'dense',
            },
            {
                'value_col': '出勤天数',
                'rank_col': '出勤排名',
                'ascending': True,
                'method': 'min',
            },
        ],
    )
    print(result.sort_values(['部门', 'KPI排名']).to_string(index=False))
    print()


def demo_top_n_summary():
    print("=" * 60)
    print("【演示4】取每个部门 KPI 排名前 2 名（dense 方式）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    ranked = service.rank(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        rank_col='KPI排名',
        ascending=False,
        method='dense',
    )
    top2 = GroupRankingService.get_rank_summary(
        df=ranked,
        group_cols='部门',
        rank_col='KPI排名',
        top_n=2,
    )
    print(top2.to_string(index=False))
    print()


def demo_pct_rank():
    print("=" * 60)
    print("【演示5】百分比排名（部门内 KPI，降序）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    result = service.rank(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        rank_col='KPI排名百分比',
        ascending=False,
        method='average',
        pct=True,
    )
    print(result.sort_values(['部门', 'KPI排名百分比']).to_string(index=False))
    print()


def demo_percentile_basic():
    print("=" * 60)
    print("【演示6】基础分组百分位：部门内 KPI 百分位（升序）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    result = service.percentile(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        percentile_col='KPI百分位',
        method='average',
        ascending=True,
    )
    print(result.sort_values(['部门', 'KPI百分位']).to_string(index=False))
    print()


def demo_percentile_methods_compare():
    print("=" * 60)
    print("【演示7】百分位计算方式对比（技术部 KPI，降序，0-100）")
    print("=" * 60)
    df = build_sample_data()
    df_tech = df[df['部门'] == '技术部'].copy()
    service = GroupRankingService()

    all_methods = ['average', 'min', 'max', 'dense', 'linear', 'lower', 'higher', 'midpoint', 'nearest']
    result = df_tech.copy()
    for method in all_methods:
        result = service.percentile(
            df=result,
            group_cols='部门',
            value_col='KPI分数',
            percentile_col=f'pct_{method}',
            method=method,
            ascending=False,
            scale=100,
        )
    cols = ['姓名', 'KPI分数'] + [f'pct_{m}' for m in all_methods]
    print(result[cols].sort_values('KPI分数', ascending=False).to_string(index=False))
    print()


def demo_percentile_multi():
    print("=" * 60)
    print("【演示8】同时对多列计算百分位（KPI降序、出勤升序）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    result = service.percentile_multi(
        df=df,
        group_cols='部门',
        percentile_configs=[
            {
                'value_col': 'KPI分数',
                'percentile_col': 'KPI百分位',
                'ascending': False,
                'method': 'linear',
            },
            {
                'value_col': '出勤天数',
                'percentile_col': '出勤百分位',
                'ascending': True,
                'method': 'average',
            },
        ],
    )
    print(result.sort_values(['部门', 'KPI百分位']).to_string(index=False))
    print()


def demo_percentile_bucket():
    print("=" * 60)
    print("【演示9】百分位分桶（十分位）")
    print("=" * 60)
    df = build_sample_data()
    service = GroupRankingService()
    result = service.percentile(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        percentile_col='KPI百分位',
        method='average',
        ascending=False,
    )
    result = GroupRankingService.get_percentile_bucket(
        df=result,
        percentile_col='KPI百分位',
        bucket_col='KPI十分位',
        bins=4,
        labels=['Top25%', '中上25%', '中下25%', 'Bottom25%'],
    )
    print(result.sort_values(['部门', 'KPI百分位']).to_string(index=False))
    print()


def demo_percentile_summary():
    print("=" * 60)
    print("【演示10】分组百分位统计摘要（P10/P25/P50/P75/P90）")
    print("=" * 60)
    df = build_sample_data()
    summary = GroupRankingService.get_percentile_summary(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        percentiles=[10, 25, 50, 75, 90],
    )
    print(summary.to_string(index=False))
    print()


def demo_percentile_with_na():
    print("=" * 60)
    print("【演示11】百分位包含空值的处理（默认空值排最后）")
    print("=" * 60)
    data = {
        '部门': ['技术部'] * 6,
        '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八'],
        'KPI分数': [95, None, 88, None, 92, 76],
    }
    df = pd.DataFrame(data)
    service = GroupRankingService()
    result = service.percentile(
        df=df,
        group_cols='部门',
        value_col='KPI分数',
        percentile_col='KPI百分位',
        ascending=False,
        method='average',
        na_option='bottom',
    )
    print(result.sort_values('KPI百分位', na_position='last').to_string(index=False))
    print()


if __name__ == '__main__':
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    demo_basic_rank()
    demo_all_methods()
    demo_multi_rank()
    demo_top_n_summary()
    demo_pct_rank()
    demo_percentile_basic()
    demo_percentile_methods_compare()
    demo_percentile_multi()
    demo_percentile_bucket()
    demo_percentile_summary()
    demo_percentile_with_na()
