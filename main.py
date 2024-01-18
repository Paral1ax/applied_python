import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px


@st.cache_data
def work_with_data():
    clients = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_clients.csv")
    closed_loan = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_close_loan.csv")
    job_client = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_job.csv")
    last_credit = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_last_credit.csv")
    loan_client = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_loan.csv")
    pens_flag_description = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_pens.csv")
    salary = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_salary.csv")
    agreement_target = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_target.csv")
    work_flag_description = pd.read_csv(
        "https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/clients/D_work.csv")

    pens_flag_description.loc[0, 'FLAG'] = 1
    pens_flag_description.loc[1, 'FLAG'] = 0

    merge_loans = pd.merge(loan_client, closed_loan, on=['ID_LOAN'], how='left')
    merge_loans = merge_loans.groupby(['ID_CLIENT']).agg(['sum', 'count']).CLOSED_FL.reset_index().rename(
        columns={'sum': 'CLOSED_LOAN', 'count': 'LOAN_COUNT'})
    merge_loans[['CLOSED_LOAN', 'LOAN_COUNT']] = merge_loans[['CLOSED_LOAN', 'LOAN_COUNT']].fillna(0).replace(r'nan', 0,
                                                                                                              regex=True)

    dataframe = (clients. \
                 merge(job_client, left_on='ID', right_on='ID_CLIENT', how='left'). \
                 merge(last_credit, on=['ID_CLIENT'], how='left'). \
                 merge(loan_client, on=['ID_CLIENT'], how='left'). \
                 merge(merge_loans, on=['ID_CLIENT'], how='left'). \
                 merge(pens_flag_description, left_on='SOCSTATUS_PENS_FL', right_on='FLAG', how='left'). \
                 merge(salary, on=['ID_CLIENT'], how='left'). \
                 merge(agreement_target, on=['ID_CLIENT'], how='left'). \
                 drop(['ID_x', 'ID_y', 'FLAG'], axis=1))

    return dataframe


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with st.status("Обработка данных..."):
        df = work_with_data()

    describe = df.describe()
    important_columns = ['AGE', 'TARGET', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
                         'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME',
                         'LOAN_COUNT', 'CLOSED_LOAN']
    id_cols = ['ID_CLIENT', 'ID_LOAN', 'AGREEMENT_RK']

    with st.container(border=True):
        with st.chat_message("assistant"):
            st.code("<Это dataframe с банковскими данными/>")
            st.dataframe(df)

    with st.container(border=True):
        with st.chat_message("assistant"):
            st.code("<А это данные с числовыми характеристиками исходного dataframe/>")
            st.dataframe(describe)

    with st.container(border=True):
        with st.chat_message("assistant"):
            st.code("<Здесь отображается график попарных признаков/>")
            st.code("<Для того чтобы его увидеть, выберите признаки/>")

        multi_features = st.multiselect('', important_columns)
        if len(multi_features) < 1:
            st.warning('Выберите хотя бы 1 значение для отображения парных признаков')
        else:
            pair_plot = px.scatter_matrix(df[multi_features])
            st.plotly_chart(pair_plot)

    with st.container(border=True):

        with st.chat_message("assistant"):
            st.code("<А здесь изображена тепловая карта корреляций!/>")

        f, ax = plt.subplots()
        sns.heatmap(df[important_columns].corr(), ax=ax, annot=True)
        st.pyplot(f)

    with st.container(border=True):

        with st.chat_message("assistant"):
            st.code("<Данный график отображает распределение выбранного признака!/>")

        feature_spread = st.selectbox(label='Выберите признак для отображения распределения',
                                      options=important_columns, label_visibility='visible')

        spread_fig = plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x=df[feature_spread], fill=False, stat='count', common_norm=False, kde=True)
        st.pyplot(spread_fig)

    with st.container(border=True):
        with st.chat_message("assistant"):
            st.code("<А здесь отображается ящик с усами!/>")

        feature_box = st.selectbox(label='Признак для ящика с усами',
                                   options=important_columns, label_visibility='visible')

        box_plot = plt.figure(figsize=(10, 4))
        sns.boxplot(data=df, y=feature_box)
        st.pyplot(box_plot)

    with st.container(border=True):
        with st.chat_message("assistant"):
            st.code("<В данном разделе отображен линейный график ПРИЗНАК - ТАРГЕТ!/>")

        feature_line = st.selectbox(label='Признак для линейного графика', options=important_columns,
                                    label_visibility='visible')

        if feature_line == 'TARGET':
            st.warning('Выберите другой признак отличный от TARGET')
        else:
            reshape = df.groupby(feature_line)['TARGET'].agg('count').reset_index()
            fig = px.line(reshape, x=reshape[feature_line], y=reshape['TARGET'])
            st.plotly_chart(fig)
