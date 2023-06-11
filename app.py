import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prediction import predict
import time

df = pd.read_csv('resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv')

df = df.replace('years', '')

df['month'] = pd.to_datetime(df['month'], format='%Y-%m').dt.strftime('%Y-%m')

df['remaining_lease'] = df['remaining_lease'].str.replace(' years ', ' ')
df['remaining_lease'] = df['remaining_lease'].str.replace(' years', '')
df['remaining_lease'] = df['remaining_lease'].str.replace(' months', '')
df['remaining_lease'] = df['remaining_lease'].str.replace(' month', '')

df[['Leaseyears', 'Leasemonths']] = df.remaining_lease.str.split(" ", expand = True)

df['Leasemonths'].unique()

df['Leasemonths'] = df['Leasemonths'].fillna(0)
df['Leaseyears'] = df['Leaseyears'].astype('int')
df['Leasemonths'] = df['Leasemonths'].astype('int')

df['remaining_lease'] = (df['Leaseyears']*12) + df['Leasemonths']

df = df.drop(columns = ['Leaseyears', 'Leasemonths'])

le = LabelEncoder()

df['town_le'] = le.fit_transform(df['town'])
df['flat_type_le'] = le.fit_transform(df['flat_type'])
df['storey_range_le'] = le.fit_transform(df['storey_range'])
df['flat_model_le'] = le.fit_transform(df['flat_model'])

before_after_finalData = df.copy()

finalData = df[['town_le', 'flat_type_le', 'storey_range_le', 'flat_model_le', 'floor_area_sqm', 'resale_price']].copy()

finalData.columns = finalData.columns.str.replace('_le', '')

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
tab1, tab2, tab3 = st.tabs(['About Us', 'Exploratory data analysis (EDA)', 'Prediction of Flat Sales Prices in Singapore'])

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

with tab3:
    st.title('Prediction of Flat Sales Prices in Singapore📈')
    st.markdown('Predicting of flat sales prices by applying regression and classification models.')
    
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
        "What is the storey range of the Flat?",
        (x for x in before_after_finalData.flat_model.unique()),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
    floor_area_sqm = st.slider('What is the Floor Area(sqm) of the Flat?', df.floor_area_sqm.min(), df.floor_area_sqm.max(), 0.5)
    remaining_lease = st.slider('What is the Remaining Lease (month) of the Flat?', 517, 1173, 1)

    town_p = Town.loc[Town['town'] == town].town_le.values[0]

    flat_type_p = FlatType.loc[FlatType['flat_type'] == flat_type].flat_type_le.values[0] 

    storey_range_p = StoreyRange.loc[StoreyRange['storey_range'] == storey_range].storey_range_le.values[0] 

    flat_model_p = FlatModel.loc[FlatModel['flat_model'] == flat_model].flat_model_le.values[0] 

    if st.button('Predict Flat Sales Prices'):

        result = predict(np.array([[town_p, flat_type_p, storey_range_p,floor_area_sqm ,flat_model_p, remaining_lease]]))
        with st.spinner("Loading..."):
            time.sleep(2)
            st.text(result[0])

