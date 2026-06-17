import pandas as pd
from typing import List, Optional, Union


class GroupRankingService:
    """
    分组排名服务，支持按分组对数值列进行排名和百分位计算。

    支持的并列处理方式:
        - 'average': 并列取平均值（默认）
        - 'min': 并列取最小排名（跳跃式）
        - 'max': 并列取最大排名（跳跃式）
        - 'dense': 并列取相同排名，后续排名不间断
        - 'first': 按出现顺序排名，无并列

    空值（NaN）处理方式:
        - 'bottom': 空值统一排最后（默认，推荐）
        - 'top': 空值统一排最前
        - 'keep': 空值排名保持为 NaN（不推荐，会导致排序异常）

    百分位计算方式（percentile method）:
        - 排名类: 'average'/'min'/'max'/'dense'/'first'（基于排名计算）
        - 插值类: 'linear'/'lower'/'higher'/'midpoint'/'nearest'（基于 quantile 插值）
    """

    VALID_METHODS = {'average', 'min', 'max', 'dense', 'first'}
    VALID_NA_OPTIONS = {'keep', 'top', 'bottom'}
    VALID_PERCENTILE_METHODS = {
        'average', 'min', 'max', 'dense', 'first',
        'linear', 'lower', 'higher', 'midpoint', 'nearest',
    }
    DEFAULT_NA_OPTION = 'bottom'

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
        na_option: Optional[str] = None,
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
            na_option: NaN 的处理方式，'bottom'（默认，排最后）/'top'（排最前）/'keep'（保持NaN）
            pct: 是否以百分比形式显示排名

        Returns:
            新增排名字段后的 DataFrame（副本）
        """
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method 必须是 {self.VALID_METHODS} 之一，当前传入: {method}"
            )

        if na_option is None:
            na_option = self.DEFAULT_NA_OPTION

        if na_option not in self.VALID_NA_OPTIONS:
            raise ValueError(
                f"na_option 必须是 {self.VALID_NA_OPTIONS} 之一，当前传入: {na_option}"
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
                na_option=config.get('na_option', self.DEFAULT_NA_OPTION),
                pct=config.get('pct', False),
            )
        return result

    @staticmethod
    def get_rank_summary(
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        rank_col: str,
        top_n: Optional[int] = None,
        na_position: str = 'last',
    ) -> pd.DataFrame:
        """
        按分组获取排名摘要，可选取每个分组的前 N 名。

        Args:
            df: 已包含排名字段的 DataFrame
            group_cols: 分组列名或列名列表
            rank_col: 排名字段名
            top_n: 每个分组取前 N 名，None 取全部
            na_position: 空值排名的位置，'last'（默认，排最后）/'first'（排最前）

        Returns:
            排序后的摘要 DataFrame
        """
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if na_position not in {'first', 'last'}:
            raise ValueError(
                f"na_position 必须是 'first' 或 'last'，当前传入: {na_position}"
            )

        result = df.sort_values(
            group_cols + [rank_col],
            na_position=na_position,
        ).reset_index(drop=True)

        if top_n is not None:
            def _get_top_n(group):
                has_na = group[rank_col].isna().any()
                if has_na and na_position == 'last':
                    valid_rows = group[group[rank_col].notna()]
                    na_rows = group[group[rank_col].isna()]
                    return pd.concat([
                        valid_rows.nsmallest(top_n, rank_col),
                        na_rows,
                    ])
                else:
                    return group.nsmallest(top_n, rank_col)

            result = (
                result.groupby(group_cols, group_keys=False)
                .apply(_get_top_n)
                .reset_index(drop=True)
            )

        return result

    def percentile(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        percentile_col: Optional[str] = None,
        method: str = 'average',
        ascending: bool = True,
        na_option: Optional[str] = None,
        scale: Union[int, float] = 100,
    ) -> pd.DataFrame:
        """
        计算每个值在其分组内的百分位数。

        Args:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            value_col: 需要计算百分位的数值列名
            percentile_col: 输出的百分位字段名，默认在 value_col 后加 '_percentile'
            method: 百分位计算方式:
                - 排名类: 'average'/'min'/'max'/'dense'/'first'（基于排名百分比）
                - 插值类: 'linear'/'lower'/'higher'/'midpoint'/'nearest'（基于 quantile 插值）
            ascending: True 为升序（值越小百分位越低，最小值≈0），False 为降序
            na_option: NaN 的处理方式，'bottom'（默认，排最后）/'top'（排最前）/'keep'（保持NaN）
            scale: 百分位缩放比例，默认 100（返回 0-100），设为 1 则返回 0-1

        Returns:
            新增百分位字段后的 DataFrame（副本）
        """
        if method not in self.VALID_PERCENTILE_METHODS:
            raise ValueError(
                f"method 必须是 {self.VALID_PERCENTILE_METHODS} 之一，当前传入: {method}"
            )

        if na_option is None:
            na_option = self.DEFAULT_NA_OPTION

        if na_option not in self.VALID_NA_OPTIONS:
            raise ValueError(
                f"na_option 必须是 {self.VALID_NA_OPTIONS} 之一，当前传入: {na_option}"
            )

        if value_col not in df.columns:
            raise ValueError(f"数值列 '{value_col}' 不存在于 DataFrame 中")

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        for col in group_cols:
            if col not in df.columns:
                raise ValueError(f"分组列 '{col}' 不存在于 DataFrame 中")

        if percentile_col is None:
            percentile_col = f"{value_col}_percentile"

        result = df.copy()
        rank_methods = {'average', 'min', 'max', 'dense', 'first'}

        if method in rank_methods:
            pct_rank = result.groupby(group_cols)[value_col].rank(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=True,
            )
            result[percentile_col] = pct_rank * scale
        else:
            def _calc_percentile(group):
                values = group[value_col].values
                non_null_mask = pd.notna(values)
                non_null_values = values[non_null_mask]

                output = pd.Series(index=group.index, dtype='float64')

                if len(non_null_values) == 0:
                    return output

                if na_option == 'keep':
                    output[~non_null_mask] = float('nan')
                elif na_option == 'top':
                    output[~non_null_mask] = 0.0 if ascending else float(scale)
                else:
                    output[~non_null_mask] = float(scale) if ascending else 0.0

                if len(non_null_values) == 1:
                    output[non_null_mask] = scale / 2
                    return output

                sorted_vals = sorted(non_null_values) if ascending else sorted(non_null_values, reverse=True)
                n = len(sorted_vals)

                for idx in group.index[non_null_mask]:
                    val = group.loc[idx, value_col]
                    if method == 'linear':
                        if ascending:
                            less_count = sum(v < val for v in sorted_vals)
                            equal_count = sum(v == val for v in sorted_vals)
                            if equal_count == 1:
                                pct = less_count / (n - 1)
                            else:
                                pct = (less_count + (equal_count - 1) / 2) / (n - 1)
                        else:
                            greater_count = sum(v > val for v in sorted_vals)
                            equal_count = sum(v == val for v in sorted_vals)
                            if equal_count == 1:
                                pct = greater_count / (n - 1)
                            else:
                                pct = (greater_count + (equal_count - 1) / 2) / (n - 1)
                    elif method == 'lower':
                        if ascending:
                            pct = sum(v < val for v in sorted_vals) / max(n - 1, 1)
                        else:
                            pct = sum(v > val for v in sorted_vals) / max(n - 1, 1)
                    elif method == 'higher':
                        if ascending:
                            pct = sum(v <= val for v in sorted_vals) / max(n - 1, 1) - (1 / n if n > 0 else 0)
                            pct = min(pct, 1.0)
                        else:
                            pct = sum(v >= val for v in sorted_vals) / max(n - 1, 1) - (1 / n if n > 0 else 0)
                            pct = min(pct, 1.0)
                    elif method == 'midpoint':
                        if ascending:
                            less = sum(v < val for v in sorted_vals)
                            greater_eq = sum(v >= val for v in sorted_vals)
                            pct_low = less / max(n - 1, 1)
                            pct_high = (n - greater_eq) / max(n - 1, 1)
                            pct = (pct_low + pct_high) / 2
                        else:
                            greater = sum(v > val for v in sorted_vals)
                            less_eq = sum(v <= val for v in sorted_vals)
                            pct_low = greater / max(n - 1, 1)
                            pct_high = (n - less_eq) / max(n - 1, 1)
                            pct = (pct_low + pct_high) / 2
                    elif method == 'nearest':
                        ranks = [i for i, v in enumerate(sorted_vals) if v == val]
                        avg_rank = sum(ranks) / len(ranks)
                        pct = avg_rank / max(n - 1, 1)
                    else:
                        pct = 0.0

                    output.loc[idx] = pct * scale

                return output

            pct_series = result.groupby(group_cols, group_keys=False).apply(_calc_percentile)
            if isinstance(pct_series, pd.DataFrame):
                pct_series = pct_series.squeeze()
            result[percentile_col] = pct_series.reindex(result.index)

        return result

    def percentile_multi(
        self,
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        percentile_configs: List[dict],
    ) -> pd.DataFrame:
        """
        同时对多个数值列计算分组百分位数。

        Args:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            percentile_configs: 百分位配置列表，每项为 dict，字段包括:
                - value_col: 数值列名（必填）
                - percentile_col: 输出百分位字段名（可选）
                - method: 百分位计算方式（可选，默认 'average'）
                - ascending: 升序/降序（可选，默认 True）
                - na_option: NaN 处理方式（可选）
                - scale: 缩放比例（可选，默认 100）

        Returns:
            新增多个百分位字段后的 DataFrame（副本）
        """
        result = df.copy()
        for config in percentile_configs:
            if 'value_col' not in config:
                raise ValueError("percentile_configs 中的每项必须包含 'value_col'")
            result = self.percentile(
                df=result,
                group_cols=group_cols,
                value_col=config['value_col'],
                percentile_col=config.get('percentile_col'),
                method=config.get('method', 'average'),
                ascending=config.get('ascending', True),
                na_option=config.get('na_option', self.DEFAULT_NA_OPTION),
                scale=config.get('scale', 100),
            )
        return result

    @staticmethod
    def get_percentile_bucket(
        df: pd.DataFrame,
        percentile_col: str,
        bucket_col: Optional[str] = None,
        bins: Union[int, List[float]] = 10,
        labels: Optional[List[str]] = None,
        right: bool = True,
    ) -> pd.DataFrame:
        """
        将百分位数值分桶（如十分位、四分位）。

        Args:
            df: 包含百分位列的 DataFrame
            percentile_col: 百分位列名
            bucket_col: 输出的分桶字段名，默认在 percentile_col 后加 '_bucket'
            bins: 分桶数量（int）或分箱边界（List[float]），默认 10（十分位）
            labels: 桶标签列表，长度应等于分桶数量
            right: 是否包含右边界，默认 True
            scale: 百分位的缩放比例，默认 100（配合 scale=100 的 percentile 使用）

        Returns:
            新增分桶字段后的 DataFrame（副本）
        """
        result = df.copy()
        if bucket_col is None:
            bucket_col = f"{percentile_col}_bucket"

        if isinstance(bins, int):
            n_bins = bins
            bin_edges = [i / n_bins * 100 for i in range(n_bins + 1)]
        else:
            bin_edges = list(bins)
            n_bins = len(bin_edges) - 1

        if labels is None:
            labels = [f"P{int(bin_edges[i])}-P{int(bin_edges[i+1])}" for i in range(n_bins)]

        if len(labels) != n_bins:
            raise ValueError(f"labels 数量 ({len(labels)}) 必须等于分桶数量 ({n_bins})")

        result[bucket_col] = pd.cut(
            result[percentile_col],
            bins=bin_edges,
            labels=labels,
            right=right,
            include_lowest=True,
        )

        return result

    @staticmethod
    def get_percentile_summary(
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_col: str,
        percentiles: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        按分组计算指定百分位数的统计摘要。

        Args:
            df: 输入的 DataFrame
            group_cols: 分组列名或列名列表
            value_col: 数值列名
            percentiles: 要计算的百分位数列表（0-100），默认 [10, 25, 50, 75, 90]

        Returns:
            分组百分位统计摘要 DataFrame
        """
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90]

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        for p in percentiles:
            if not 0 <= p <= 100:
                raise ValueError(f"百分位数 {p} 超出范围 [0, 100]")

        q = [p / 100 for p in percentiles]

        summary = (
            df.groupby(group_cols)[value_col]
            .quantile(q)
            .unstack()
            .reset_index()
        )
        summary.columns = group_cols + [f"P{p}" for p in percentiles]
        return summary
