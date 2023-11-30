import pickle

import pandas as pd
import streamlit as st

teams = ['South Africa',
 'India',
 'New Zealand',
 'West Indies',
 'Australia',
 'Sri Lanka',
 'England',
 'Bangladesh',
 'Pakistan']

cities = ['Sharjah Cricket Stadium', 'Saxton Oval', 'Hyderabad',
       'Sydney Cricket Ground', 'Abu Dhabi', 'Harare Sports Club',
       'Nagpur', 'The Rose Bowl', 'Visakhapatnam', 'Headingley',
       'Premadasa Stadium', 'Dubai', 'Dharamsala',
       'New Wanderers Stadium', 'Queen Park Oval', 'Kennington Oval',
       'University Oval', 'Kimberley', 'Ahmedabad', 'Ranchi', 'Malahide',
       'Senwes Park', 'Delhi', 'Shere Bangla', 'Boland Park', 'Bay Oval']

pipe = pickle.load(open('F:\\class\\Machine Learning\\project\\T20 Match Win Probalility\\model.pkl','rb'))
st.title('T20 Match Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    ball_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/ball_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_lef':[runs_left],'ball_left':[ball_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")