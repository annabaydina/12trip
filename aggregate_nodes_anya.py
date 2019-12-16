from typing import List, Set, Dict, Optional, Any, Tuple, Type, Union
from .node import Node
from .pipeline import *
from ..io import *
import numpy as np
import os
from sklearn import preprocessing
from ..encoding import LabelEncoderPopularity
from ..aggregations.temporal import aggregate_with_time_local, aggregate_transaction_frequencies
import multiprocess as mp


class AddAggregatesTotalNode(Node):
    '''
    For every numerical feature makes to_mean and to_std
    '''
    params = {
        'features': [],
        'group_by': ''
    }

    def _run(self):
        data = self.input
        group_by_feature = self.params['group_by']

        for fname in self.params['features']:
            data[f'{fname}_to_mean_{group_by_feature}'] = data[fname] / data.groupby([group_by_feature])[
                fname].transform('mean').replace(-np.inf, np.nan).replace(np.inf, np.nan).astype(np.float32)
            data[f'{fname}_to_std_{group_by_feature}'] = data[fname] / data.groupby([group_by_feature])[
                fname].transform('std').replace(-np.inf, np.nan).replace(np.inf, np.nan).astype(np.float32)
            # num_cols.extend([f'{fname}_to_mean_{group_by_feature}', f'{fname}_to_std_{group_by_feature}'])

        self.output = data


class AddGroupNumericalAggregatesNode(Node):
    params = {
        'features': [],
        'group_by': [],
        'to_mean': True,
        'to_std': True,
        'to_minmax': True,
        'to_std_score': True
    }

    def _run(self):
        df = self.input

        groupby = self.params['group_by']
        cols = self.params['features']
        to_mean = self.params['to_mean']
        to_std = self.params['to_std']
        to_minmax = self.params['to_minmax']
        to_std_score = self.params['to_std_score']

        slice_df = df[groupby + cols]

        if len(groupby) == 1:
            gcname = groupby[0]
        else:
            gcname = "(" + '+'.join(groupby) + ")"

        for igc, gc in enumerate(groupby):
            if igc == 0:
                slice_df['group_col'] = slice_df[gc].astype(str)
            else:
                slice_df['group_col'] += '_' + slice_df[gc].astype(str)


        for col in cols:
            group_col = slice_df.groupby(['group_col'])[col]
            dmean = group_col.transform('mean')
            dstd = group_col.transform('std').replace(-np.inf, np.nan).replace(np.inf, np.nan)

            if to_mean:
                df[f'{col}_to_mean_groupby_{gcname}'] = (slice_df[col] / dmean).astype(np.float32)

            if to_std:
                df[f'{col}_to_std_groupby_{gcname}'] = (slice_df[col] / dstd).replace(
                    {np.inf: 0, -np.inf: 0, np.nan: 0}).astype(np.float32)

            if to_minmax:
                dmin = group_col.transform('min').astype(np.float32)
                dmax = group_col.transform('max').astype(np.float32)
                df[f'{col}_to_minmax_groupby_{gcname}'] = ((slice_df[col] - dmin) / (dmax - dmin)).replace(
                    {np.inf: 0, -np.inf: 0, np.nan: 0}).astype(np.float32)

            if to_std_score:
                df[f'{col}_to_stdscore_groupby_{gcname}'] = ((slice_df[col] - dmean) / dstd).replace(
                    {np.inf: 0, -np.inf: 0, np.nan: 0}).astype(np.float32)
        self.output = df

class AddGroupFrequencyEncodingNode(Node):
    params = {
        'features': [],
        'group_by': [],
        'count': True,
        'freq': False
    }

    def _run(self):
        df = self.input
        groupby = self.params['group_by']
        cols = self.params['features']
        to_count = self.params['count']
        to_freq = self.params['freq']

        slice_df = df[groupby + cols]
        if len(groupby) == 1:
            gcname = groupby[0]
        else:
            gcname = "(" + '+'.join(groupby) + ")"

        for igc, gc in enumerate(groupby):
            if igc == 0:
                slice_df['group_col'] = slice_df[gc].astype(str)
            else:
                slice_df['group_col'] += '_' + slice_df[gc].astype(str)

        if to_freq:
            count = slice_df['group_col'].map(slice_df['group_col'].value_counts(dropna=False))
        for col in cols:
            tempdf = (slice_df['group_col'] + '_' + slice_df[col].astype(str))
            tempdf = tempdf.map(tempdf.value_counts(dropna=False))
            if to_count:
                df[f'{col}_count_groupby_{gcname}'] = tempdf
            if to_freq:
                df[f'{col}_freq_groupby_{gcname}'] = tempdf / count

        self.output = df

class AddGlobalFrequencyEncodingNode(Node):
    params = {
        'features': [],
        'count': True,
        'freq': False
    }

    def _run(self):
        df = self.input
        cols = self.params['features']
        to_count = self.params['count']
        to_freq = self.params['freq']

        slice_df = df[cols]
        for col in cols:
            tempdf = slice_df[col].map(slice_df[col].value_counts(dropna=False))
            if to_count:
                df[f'{col}_global_count'] = tempdf
            if to_freq:
                df[f'{col}_global_freq'] = tempdf / tempdf.shape[0]

        self.output = df


class AddGroupFrequencyEncodingNode(Node):
    params = {
        'features': [],
        'group_by': [],
        'count': True,
        'freq': False
    }

    def _run(self):
        df = self.input
        groupby = self.params['group_by']
        cols = self.params['features']
        to_count = self.params['count']
        to_freq = self.params['freq']

        slice_df = df[groupby + cols]
        if len(groupby) == 1:
            gcname = groupby[0]
        else:
            gcname = "(" + '+'.join(groupby) + ")"

        for igc, gc in enumerate(groupby):
            if igc == 0:
                slice_df['group_col'] = slice_df[gc].astype(str)
            else:
                slice_df['group_col'] += '_' + slice_df[gc].astype(str)
        if to_freq:
            count = slice_df['group_col'].map(slice_df['group_col'].value_counts(dropna=False))
        for col in cols:
            tempdf = (slice_df['group_col'] + '_' + slice_df[col].astype(str))
            tempdf = tempdf.map(tempdf.value_counts(dropna=False))
            if to_count:
                df[f'{col}_count_groupby_{gcname}'] = tempdf
            if to_freq:
                df[f'{col}_freq_groupby_{gcname}'] = tempdf / count

        self.output = df



class AddTemporalAggregates(Node):
    params = {
        'features': [],
        'date_field': '',
        'window': [],
        'group_by': ''
    }

    def _run(self):
        data = self.input
        window = self.params['window']
        group_by_feature = self.params['group_by']
        date_field = self.params['date_field']

        # Fixing same data timestamps for same card_id
        # check_data = data.reset_index().set_index([group_by_feature, date_field])
        # duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        #
        # while len(duplicate_transactions) > 0:
        #     print(f"Found {len(duplicate_transactions)} duplicate transactions")
        #     for itid, tid in enumerate(duplicate_transactions):
        #         print(itid)
        #         q = data.loc[tid]
        #         date = q[date_field]
        #         card_id = q[group_by_feature]
        #         alldup = data[data[date_field] == date]
        #         alldup = alldup[alldup[group_by_feature] == card_id]
        #         #     print(alldup.index)
        #         for it, idx in enumerate(alldup.index):
        #             #         print(idx)
        #             data.loc[idx, date_field] += pd.Timedelta(seconds=it)
        #     check_data = data.reset_index().set_index([group_by_feature, date_field])
        #     duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values


        with mp.Pool() as Pool:

            self.output = pd.DataFrame(index=data.index)
            for nf in self.params['features']:
                if nf not in data.columns:
                    continue

                print(nf)
                df = pd.DataFrame(index=data.index)
                data_slice = data[[date_field, group_by_feature, nf]].reset_index()
                args = [(data_slice, group_by_feature, nf, ws) for ws in window]
                m = Pool.imap(aggregate_with_time_local, args)

                for i, df_agg in enumerate(m):
                    print('.')
                    assert df.shape[0] == df_agg.shape[0]
                    df = pd.concat([df, df_agg], axis=1)

                self.output = self.output.join(df)



class AddTransactionFrequenciesNode(Node):
    params = {
        'group_by': 'new_card_id'
    }

    def _run(self):
        data = self.input
        window = self.params['window']
        group_by_feature = self.params['group_by']
        date_field = self.params['date_field']

        # Fixing same data timestamps for same card_id

        # check_data = data.reset_index().set_index([group_by_feature, 'Date'])
        # duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        # while len(duplicate_transactions) > 0:
        #     print(f"Found {len(duplicate_transactions)} duplicate transactions")
        #     for itid, tid in enumerate(duplicate_transactions):
        #         print(itid)
        #         q = data.loc[tid]
        #         date = q['Date']
        #         card_id = q[group_by_feature]
        #         alldup = data[data['Date'] == date]
        #         alldup = alldup[alldup[group_by_feature] == card_id]
        #         #     print(alldup.index)
        #         for it, idx in enumerate(alldup.index):
        #             #         print(idx)
        #             data.loc[idx, 'Date'] += pd.Timedelta(seconds=it)
        #     check_data = data.reset_index().set_index([group_by_feature, 'Date'])
        #     duplicate_transactions = check_data[check_data.index.duplicated()]['TransactionID'].values
        #     print(data.loc[alldup.index])
        #     break

        with mp.Pool() as Pool:
            self.output = pd.DataFrame(index=data.index)
            df = pd.DataFrame(index=data.index)
            data_slice = data[[date_field, group_by_feature]].reset_index()

            data_slice.loc[:, 'hours'] = (data_slice.date_field - data_slice.date_field.iloc[0]).dt.total_seconds() / 3600

            args = [(data_slice, group_by_feature, ws) for ws in window]
            m = Pool.imap(aggregate_transaction_frequencies, args)
            for i, df_agg in enumerate(m):
                print('.')
                assert df.shape[0] == df_agg.shape[0]
                df = pd.concat([df, df_agg], axis=1)

            self.output = self.output.join(df)