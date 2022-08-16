#!/usr/bin/env python
# coding: utf-8

# # herondata exercise

# In[80]:


import calendar
from collections import Counter, defaultdict
import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import torch
from transformers import pipeline
import uuid


# ## load data

# In[36]:


def read_transactions(filename):
    with open(filename, 'rt') as fp:
        raw_data = json.load(fp)

    list_of_tx = raw_data['transactions']
    return list_of_tx

def to_df(transactions):
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    # df = df.sort_values('date')

    return df


# # explore data

# observation: maybe theres 4 or 5 recurring payments in the given dataset?

# observations:
# - acme corp salary has payment date in description
# - company lunch maybe recurring

# # brainstorming/ideas
# 
# define recurring
# - similar amt (does sign matter? refunds?)
# - same merchant
# - payment dates spaced uniformly
# 
# need features for ?each of? the above, 
# - should these be calculated independently
# - should we do:
#     - grp(amt, merchant) -> metric(periodicity)
#     - grp(amt, periodicity) -> metric(is_same_merchant)
#     - grp(amt) -> metric(is_same_merchant)
#     - ...
#     - then need to combine these somehow
# 
# add extra categories to strengthen confidence
# - entertainment, food, income, sports
# 
# assume this is for forecasting

# # cleaning: remove payment dates from description

# In[41]:


def _build_month_regex(month_idx):
    # prev.month, current.month, next.month in [1,12]-space
    # (((month_idx - 1) - 1) % 12) + 1, month_idx, ((((month_idx - 1) + 1) % 12) + 1)
    months_re = []
    month_lags = ((month_idx - 2) % 12) + 1, month_idx, (month_idx % 12) + 1
    for m in month_lags:
        months_re.append(calendar.month_name[m])
        months_re.append(calendar.month_abbr[m])

    return '|'.join(months_re).lower()

def clean_payment_dates(df, debug=False):
    df['month_regex'] = df.date.dt.month.apply(_build_month_regex)
    df['year_regex'] = df.date.dt.year.apply(lambda y: f"{y-1}|{y}|{y+1}")

    if debug:
        display(df.sample(n=3))

    df['description'] = (
        df[['description', 'month_regex', 'year_regex']]
        .apply(axis=1,
               func=lambda srs: re.sub(f"{srs.month_regex}|{srs.year_regex}", '', srs.description.lower()).strip())
        )

    df = df.drop(columns=['month_regex', 'year_regex'])
    return df


# # feature: determine periodicity/seasonality of tx
# 
# ideas
# - acf/autocorrelation
# - fourier transform

# In[43]:


def _align_timeseries(df_sparse, value_label, date_label='date'):
    df_sparse = df_sparse.sort_values(date_label)
    df_start = df_sparse[date_label].iloc[0].date()
    df_end = df_sparse[date_label].iloc[-1].date()

    df_aligned = (pd
        .DataFrame({date_label: pd.date_range(start=df_start, end=df_end)})
        .set_index(date_label)
        .join(df_sparse[[date_label, value_label]].set_index(date_label))
        .fillna(0)
    )
    
    return df_aligned

def _acf_helper(df_aligned, value_label, title, nlags=35, threshold=0.1, filter_multiples=False, debug=False):
    if debug:
        sm.graphics.tsa.plot_acf(df_aligned[value_label].squeeze(), lags=nlags, title=title)
    df_acf = (pd
        .DataFrame({'acf': sm.tsa.stattools.acf(df_aligned[value_label].squeeze(), nlags=nlags)})
        .sort_values(by='acf', key=abs, ascending=False)
        .query("abs(acf) >= @threshold")
        [1:]
    )
    
    if filter_multiples:
        f = df_acf.index < min(df_acf.index) * 2
        df_acf = df_acf[f]
    return df_acf


# In[45]:


def _lag_to_period_category(
    lags, 
    PERIODS = pd.DataFrame({'Q': 365/4, 'M': 365/12, 'F': 14, 'W': 7, 'D': 1}.items(), columns=["period", "lag"])
):
    mean_lag = sum(lags)/len(lags)
    sd_lag = np.std(lags)
 
    p = PERIODS[PERIODS.lag.between(mean_lag - sd_lag, mean_lag + sd_lag)]
    if len(p) == 1:
        return p.iloc[0].period
    
    return uuid.uuid1()


# In[46]:


def predict_periods(df, debug=False):
    periods = dict()
    for name, group in df.groupby('amount'):
        if group.shape[0] > 1:
            group_periodicity = _align_timeseries(group, 'amount')
            group_acf = _acf_helper(group_periodicity, 'amount', f"amount={name}", threshold=0.2, filter_multiples=True, debug=debug)
            periods[name] = _lag_to_period_category(group_acf.index)
        else:
            periods[name] = uuid.uuid1()

    periods_by_amount = pd.DataFrame(periods.items(), columns=['amount', 'amt_freq'])
    df = df.merge(periods_by_amount, how='left')

    return df


# In[48]:


def _add_freq_detail(grp):
    freq = grp.amt_freq.values[0]
    n = len(grp)
    res = 0
    if freq == 'M':
        res = np.average(grp.date.dt.day).astype(int)
        
    if freq == 'W':
        res = np.average(grp.date.dt.dayofweek).astype(int)

    return pd.DataFrame([res] * n, index=grp.index)

def predict_period_frequency(df):
    df['amt_freq_detail'] = (df
        [['date', 'amt_freq', 'amount']]
        .groupby(['amt_freq', 'amount'], as_index=False)
        .apply(_add_freq_detail)
    )
    return df


# # feature: categorise transactions

# In[50]:


def _setup_classifier(classifier=[]):
    if len(classifier) == 0:
        classifier.append(pipeline('zero-shot-classification', model='facebook/bart-large-mnli'))
    return classifier[0]    


# In[51]:


def _predict_category_helper(s):
    classifier = _setup_classifier()
    hypothesis_template = '{}.'
    labels = ["video streaming", "music streaming", "groceries", "sports", "food", "health", "income", "mortgage", "loan", "rent", "phone"]
    desc = s.iloc[0].values[0]
    prediction = classifier(desc, labels, hypothesis_template=hypothesis_template)
    
    n = s.shape[0]
    res = prediction['labels'][0]
    return pd.DataFrame([res] * n, index=s.index)

def predict_category(df):
    df['category'] = df[['description']].groupby('description').apply(_predict_category_helper)
    return df


# In[65]:


# def identify_recurring_transactions(transactions: List[Transaction]) -> List[Transaction.id]:
def identify_recurring_transactions(transactions, ids_only=True):
    df = to_df(transactions)
    df = clean_payment_dates(df)
    df = predict_periods(df)
    df = predict_period_frequency(df)
    df = predict_category(df)

    recurring_tx = []
    for n, grp in df.groupby(['amt_freq', 'amt_freq_detail', 'category']):
        if len(grp) >= 3:
            if ids_only:
                grp_res = list(grp.index)
            else:
                grp_res = grp[['date', 'description', 'amount']]

            recurring_tx.append(grp_res)

    return recurring_tx

