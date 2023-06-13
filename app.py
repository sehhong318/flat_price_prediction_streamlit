import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from PIL import Image

df = pd.read_csv('./Data/data.csv')
eda_df = pd.read_csv('./Data/before_eda.csv')

before_after_finalData = df.copy()

# Sort town data alphabetically in ascending order
Town = before_after_finalData[['town','town_le']].copy()
Town.drop_duplicates(subset=['town'], inplace = True)    
Town.sort_values('town', inplace = True)
Town.reset_index(drop = True, inplace = True)

# Sort Flat Type data in ascending order
FlatType = before_after_finalData[['flat_type','flat_type_le']].copy()
FlatType.drop_duplicates(subset=['flat_type'], inplace = True)    
FlatType.sort_values('flat_type', inplace = True)
FlatType.reset_index(drop = True, inplace = True)

# Sort Storey Range data in ascending order
StoreyRange = before_after_finalData[['storey_range','storey_range_le']].copy()
StoreyRange.drop_duplicates(subset=['storey_range'], inplace = True)    
StoreyRange.sort_values('storey_range', inplace = True)
StoreyRange.reset_index(drop = True, inplace = True)

# Sort Flat Model data in ascending order
FlatModel = before_after_finalData[['flat_model','flat_model_le']].copy()
FlatModel.drop_duplicates(subset=['flat_model'], inplace = True)    
FlatModel.sort_values('flat_model', inplace = True)
FlatModel.reset_index(drop = True, inplace = True)

# Dashboard

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
tab1, tab2, tab3 = st.tabs(['About Us', 'Exploratory Data Analysis (EDA)', 'Prediction of Flat Sales Prices in Singapore'])

with tab1:
    st.title("About Us🧑🏻‍💻🧑🏻‍💻🧑🏻‍💻🧑🏻‍💻👩🏻‍💻")
    st.header("Group 3️⃣")
    data_df = pd.DataFrame(
        {
            "Name": ["🧑🏻‍💻LOW SEH HONG", "🧑🏻‍💻QIFANG ZOU", "🧑🏻‍💻MANG YU JIE", "🧑🏻‍💻LEE GUANG SHEN", "👩🏻‍💻LEE YEN WEN"],

            "Matric ID": ["S2178662", "S2195887", "22064556", "22052269", "17179615"],

            "Role": ["LEADER", "GENERALIST", "CODER", "PRESENTER", "SECRETARY"],

            "apps": [
                "https://drive.google.com/drive/folders/1h4lw4XWuRj8P9moGGq0m0gr5jibWySkd?usp=share_link",
                "https://drive.google.com/drive/folders/1czJhqbCZDiq-VXpv8nieh1X42_ff5Rd1?usp=share_link",
                "https://drive.google.com/drive/folders/1dJSoQ_ZFYBCQQst_hXvP-JNmpH_CrqiS?usp=share_link",
                "https://drive.google.com/drive/folders/1VVcrsXV6qL9_SHSDa4TYipOfw9vH1fU2?usp=share_link",
                "https://drive.google.com/drive/folders/1_eKXdb7oshv93aK0_qcXIEj1GDlcbSbI?usp=share_link"
            ],
        }
    )

    st.data_editor(
        data_df,
        column_config={
        "apps": st.column_config.LinkColumn(
            "E-PORTFOLIO",
            max_chars=100,
        )
    },
        hide_index=True,
    )

with tab2:
    st.title('Exploratory data analysis (EDA)💡')

    st.markdown('Before, we proceed to EDA, the purpose of the EDA is to answer the following questions:')
    st.markdown('1. How correlates between the numerical variables?')
    st.markdown('2. What is the relationship between the independent variables and dependent variable?')
    st.markdown('3. Which model is the best model to predict the sales prices of flats?')
    st.markdown('4. Which model is the best model to predict the sales category of flats?')

    st.subheader('1. Correlation between numeric features.')
    st.markdown('From this, we can see that `lease_commence_date` and `remaining_lease` is highly correlated, so we will drop `lease_commemce_date`.')
    fig = px.imshow(eda_df[['remaining_lease', 'lease_commence_date', 'floor_area_sqm', 'month', 'year']].corr().round(2), color_continuous_scale=px.colors.sequential.Cividis_r, text_auto=True)
    st.plotly_chart(fig)

    eda_df.drop(columns=['lease_commence_date'], inplace=True)

    st.subheader('2. Distribution of `price_category` & `resale_price`.') 
    st.markdown('From the result, we can see that it is very imbalanced, and it is skew to the right. We will proceed to drop the outlier using IQR method.')

    hist_data = [eda_df['resale_price'].values.tolist()]
    group_labels = ['resale_price']
    fig = ff.create_distplot(hist_data, group_labels,show_rug=False)
    fig.update_traces(nbinsx=30, autobinx=True, selector={'type':'histogram'}) 
    fig.update_layout(width=700, height=500,yaxis_title="density",
        xaxis_title="resale_price",showlegend=False)
    st.plotly_chart(fig)

    st.markdown('After removing outlier of resale_price, the dataset observations from 143421 to 139795. The distribution of resale_price is shown below:')

    q1 = eda_df['resale_price'].quantile(q=0.25)
    q3 = eda_df['resale_price'].quantile(q=0.75)  
    iqr = q3 - q1   
    lower_bound = q1 - (1.5 * iqr) 
    upper_bound = q3 + (1.5 * iqr)
    eda_df = eda_df.loc[(eda_df['resale_price'] >= lower_bound) &( eda_df['resale_price'] <= upper_bound)]

    hist_data = [eda_df['resale_price'].values.tolist()]
    group_labels = ['resale_price']
    fig = ff.create_distplot(hist_data, group_labels,show_rug=False)
    fig.update_traces(nbinsx=30, autobinx=True, selector={'type':'histogram'}) 
    fig.update_layout(width=700, height=500,yaxis_title="density",
        xaxis_title="resale_price",showlegend=False)
    st.plotly_chart(fig)

    st.subheader('3. Relationship of different numerical variables.')

    numericalData = df[['floor_area_sqm','remaining_lease','resale_price']].copy()

    image = Image.open('./Pic/1.png')
    st.image(image)

    fig = px.imshow(numericalData.corr().round(2), color_continuous_scale=px.colors.sequential.Cividis_r, text_auto=True)
    st.plotly_chart(fig)

    image = Image.open('./Pic/2.png')
    st.image(image)

    st.markdown('Based on the heatmap, both the numerical variables show a positive correlation with `resale_price`. The correlation between `floor_area_sqm` and `resale_price` is 0.64 while the correlation between `remaining_lease` and `resale_price` is 0.34. This means that the `floor_area_sqm` has a stronger correlation and linearity with `resale_price` as compared to the `remaining_lease`.')

    st.subheader('4. Relationship of different categorical variables vs `resale_price`.')

    fig = go.Figure()

    fig.add_trace(go.Box(x=df['town'],y=df['resale_price'], marker = {'color' : 'blue'}))
    fig.update_layout(yaxis_title="resale_price", xaxis_title="town",showlegend=False)
    st.plotly_chart(fig)

    st.markdown('By comparing the box plots for different towns, if the box plots for towns in highly desirable locations have higher medians and larger interquartile ranges than the box plots for towns in less desirable locations, this suggests that properties in more desirable towns tend to have higher resale prices.')

    fig = go.Figure()

    fig.add_trace(go.Box(x=df['flat_type'],y=df['resale_price'],marker = {'color' : 'red'}))
    fig.update_layout(yaxis_title="resale_price", xaxis_title="flat_type",showlegend=False)
    st.plotly_chart(fig)

    st.markdown('By comparing the box plots for different flat types, if the box plots for flats with more bedrooms, bigger area or multi generation apartment have higher medians and larger interquartile ranges than the box plots for flats with less bedrooms, smaller area or no balcony, this suggests that those types of flats tend to have higher resale prices.')

    fig = go.Figure()

    fig.add_trace(go.Box(x=df['storey_range'],y=df['resale_price'], marker = {'color' : 'green'}))
    fig.update_layout(yaxis_title="resale_price",xaxis_title="storey_range",showlegend=False)

    st.plotly_chart(fig)

    st.markdown('By evaluating the field plots for extraordinary storey levels, if the field plots for residences on better storeys have better medians and large interquartile tiers than the field plots for residences on decrease storeys, this shows that residences on better storeys generally tend to have better resale costs.')

    fig = go.Figure()

    fig.add_trace(go.Box(x=df['flat_model'],y=df['resale_price'], marker = {'color' : 'brown'}))
    fig.update_layout(yaxis_title="resale_price", xaxis_title="flat_model",showlegend=False)

    st.plotly_chart(fig)
    st.markdown('Flat models that are newer, more modern and energy efficient tend to have higher resale prices than older or less modern flat models.')

with tab3:
    st.title('Prediction of Flat Sales Prices in Singapore📈')
    st.markdown('Prediction of flat sales prices by applying regression and classification models.')
    
    st.header('Flat Features')
    town = st.selectbox(
        "Where is the location of the Flat?",
        (x for x in before_after_finalData.town.unique()),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
    flat_type = st.selectbox(
        "What is the type of the Flat?",
        (x for x in before_after_finalData.flat_type.unique()),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
    storey_range = st.selectbox(
        "What is the storey range of the Flat?",
        (x for x in before_after_finalData.storey_range.unique()),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
    flat_model = st.selectbox(
        "What is the model of the Flat?",
        (x for x in before_after_finalData.flat_model.unique()),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
    floor_area_sqm = st.slider('What is the Floor Area(sqm) of the Flat?', df.floor_area_sqm.min(), df.floor_area_sqm.max(), 0.5)
    remaining_lease = st.slider('What is the Remaining Lease (month) of the Flat?', df.remaining_lease.min(), df.remaining_lease.max(), 0.1)

    town_p = Town.loc[Town['town'] == town].town_le.values[0]

    flat_type_p = FlatType.loc[FlatType['flat_type'] == flat_type].flat_type_le.values[0] 

    storey_range_p = StoreyRange.loc[StoreyRange['storey_range'] == storey_range].storey_range_le.values[0] 

    flat_model_p = FlatModel.loc[FlatModel['flat_model'] == flat_model].flat_model_le.values[0] 

    if st.button('Predict Flat Sales Prices'):

        result = predict(np.array([[town_p, flat_type_p, storey_range_p,floor_area_sqm ,flat_model_p, remaining_lease]]))
        with st.spinner("Loading..."):

            color_price = ''

            st.code('Flat Sales Price using Random Forest Regressor: {} SGD'.format(result[0][0]))

            if result[1] == 'Low':
                color_price = '🟢'
                st.success('Flat Sales Price using XGBoost Classifier: {}'.format(result[1]), icon = color_price)
            elif result[1] == 'Medium':
                color_price = '🟠'
                st.warning('Flat Sales Price using XGBoost Classifier: {}'.format(result[1]), icon = color_price)
            else:
                color_price = '🔴'
                st.error('Flat Sales Price using XGBoost Classifier: {}'.format(result[1]), icon = color_price)

            hist_data = [eda_df['resale_price'].values.tolist()]
            group_labels = ['resale_price']
            fig = ff.create_distplot(hist_data, group_labels,show_rug=False)
            fig.update_traces(nbinsx=30, autobinx=True, selector={'type':'histogram'}) 
            fig.add_vline(x=result[0][0], line_width=2, line_dash="dash", line_color="gray")

            fig.update_layout(height = 300,yaxis_title="density",
                xaxis_title="resale_price",showlegend=False, margin=dict(t=0, b=0))
            st.plotly_chart(fig)



