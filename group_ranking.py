import pandas as pd
from typing import List, Optional, Union


class GroupRankingService:
    """
    分组排名服务，支持按分组对数值列进行排名。

    支持的并列处理方式:
        - 'average': 并列取平均值（默认）
        - 'min': 并列取最小排名（跳跃式）
        - 'max': 并列取最大排名（跳跃式）
        - 'dense': 并列取相同排名，后续排名不间断
        - 'first': 按出现顺序排名，无并列
    """

    VALID_METHODS = {'average', 'min', 'max', 'dense', 'first'}

    def __init__(self):
        pass

    def rank(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        rank_col: Optional[str] = None,
        method: str = 'average',
        ascending: bool = True,
        na_option: str = 'keep',
        pct: bool = False,
    ) -> pd.DataFrame:
        """
        对 DataFrame 按分组列进行排名。

        Args:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表（如 '部门' 或 ['部门', '团队']）
            value_col: 需要排名的数值列名
            rank_col: 输出的排名字段名，默认在 value_col 后加 '_rank'
            method: 并列处理方式，可选 'average'/'min'/'max'/'dense'/'first'
            ascending: True 为升序（值越小排名越靠前），False 为降序
            na_option: NaN 的处理方式，'keep'/'top'/'bottom'
            pct: 是否以百分比形式显示排名

        Returns:
            新增排名字段后的 DataFrame（副本）
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method 必须是 {self.VALID_METHODS} 之一，当前传入: {method}"
            )

        if value_col not in df.columns:
            raise ValueError(f"数值列 '{value_col}' 不存在于 DataFrame 中")

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        for col in group_cols:
            if col not in df.columns:
                raise ValueError(f"分组列 '{col}' 不存在于 DataFrame 中")

        if rank_col is None:
            rank_col = f"{value_col}_rank"

        result = df.copy()

        result[rank_col] = (
            result.groupby(group_cols)[value_col]
            .rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )
        )

        return result

    def rank_multi(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        rank_configs: List[dict],
    ) -> pd.DataFrame:
        """
        同时对多个数值列进行排名。

        Args:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            rank_configs: 排名配置列表，每项为 dict，字段包括:
                - value_col: 数值列名（必填）
                - rank_col: 输出排名字段名（可选）
                - method: 并列处理方式（可选，默认 'average'）
                - ascending: 升序/降序（可选，默认 True）
                - na_option: NaN 处理方式（可选）
                - pct: 是否百分比排名（可选）

        Returns:
            新增多个排名字段后的 DataFrame（副本）
        """
        result = df.copy()
        for config in rank_configs:
            if 'value_col' not in config:
                raise ValueError("rank_configs 中的每项必须包含 'value_col'")
            result = self.rank(
                df=result,
                group_cols=group_cols,
                value_col=config['value_col'],
                rank_col=config.get('rank_col'),
                method=config.get('method', 'average'),
                ascending=config.get('ascending', True),
                na_option=config.get('na_option', 'keep'),
                pct=config.get('pct', False),
            )
        return result

    @staticmethod
    def get_rank_summary(
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        rank_col: str,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        按分组获取排名摘要，可选取每个分组的前 N 名。

        Args:
            df: 已包含排名字段的 DataFrame
            group_cols: 分组列名或列名列表
            rank_col: 排名字段名
            top_n: 每个分组取前 N 名，None 取全部

        Returns:
            排序后的摘要 DataFrame
        """
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        result = df.sort_values(group_cols + [rank_col]).reset_index(drop=True)

        if top_n is not None:
            result = (
                result.groupby(group_cols, group_keys=False)
                .apply(lambda x: x.nsmallest(top_n, rank_col))
                .reset_index(drop=True)
            )

        return result
