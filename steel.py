import pandas as pd
import pickle
import base64
import os
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


st.title('Titlting Furnace Casing Steelwork Predictor')
st.write('')

with st.expander('Unit casing Input'):
    form = st.form(key='my_form')
    capacity = form.number_input(label='Holding Capacity(Tonnes)', format="%.2f")
    door_height = form.number_input(label='Door Opening Height(mm)', format="%.2f")
    door_width = form.number_input(label='Door Opening Width(mm)', format="%.2f")
    metal_depth = form.number_input(label='Metal Bath Depth(mm)', format="%.2f")
    bath_length = form.number_input(label='Metal Bath Length(mm)', format="%.2f")
    submit_button = form.form_submit_button(label='Submit')

    
    
    if submit_button:
        d = {'Holding Capacity':[capacity], 'Door Opening Height':[door_height],'Door Opening Width': [door_width], 'Metal Bath Depth':[metal_depth],
            'Metal Bath Length':[bath_length]}
        data = pd.DataFrame(data=d)
        st.write('')
        st.write('Input Data')
        fig =  ff.create_table(data)
        fig.update_layout(width=670)
        st.write(fig)

        st.write('')
        st.write('')
        st.write('Predicted Output')

        def pred_model(model, scale):
            with open(model, 'rb') as f:
                model = pickle.load(f)
            
            with open(scale, 'rb') as f:
                scaler = pickle.load(f)

            pred_columns = ['Hearth Back Ramp', 'Hearth Centre Base', 'Hearth Front Ramp', 'Back Wall', 'Left Wall','Right Wall', 'Roof Beams', 
                'Lintel Beam', 'Overhead Door Shaft','Door Fabrication', 'Heat Shield', 'Door Surround Casting','Refractory','Total Steel']
            
            t_data = scaler.transform(data) 
            y_pred = model.predict(t_data)
            y_pred = y_pred.astype(np.int64)
    
            output_df = pd.DataFrame(y_pred, columns=pred_columns)
            d3 = {'Assembly Component': output_df.columns, 'Predicted Weight(Kgs)': output_df.iloc[0]}
            final = pd.DataFrame(data=d3)
            final['Predicted Weight(Kgs)'] = final['Predicted Weight(Kgs)'].abs()
            return final

        pred_data = pred_model('tuned_pkl', 'scaler_pkl')
        fig =  ff.create_table(pred_data)
        fig.update_layout(width=670)
        st.write(fig)

st.write('')

with st.expander('Batch Prediction'):
    # function to upload the file

    def file_upload(name):
        uploaded_file = st.file_uploader('%s' % (name),key='%s' % (name),accept_multiple_files=False)
        content = False
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                content = True
                return content, uploaded_df
            except:
                try:
                    uploaded_df = pd.read_excel(uploaded_file)
                    content = True
                    return content, uploaded_df
                except:
                    st.error('Please ensure file is .csv or .xlsx format and/or reupload file')
                    return content, None
        else:
            return content, None

        # make predictions on test data
    def download(df,filename): # Downloading DataFrame
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = (f'<a href="data:file/csv;base64,{b64}" download="%s.csv">Download csv file</a>' % (filename))
        return href
    
    status, df = file_upload('Please upload batch input data for Steelwork Prediction')

    if st.button('Submit'):
        def pred_model(model, scale):
            with open(model , 'rb') as f:
                model = pickle.load(f)
            
            with open(scale, 'rb') as f:
                scaler = pickle.load(f)

            pred_columns = ['Hearth Back Ramp', 'Hearth Centre Base', 'Hearth Front Ramp', 'Back Wall', 'Left Wall','Right Wall', 'Roof Beams', 
                'Lintel Beam', 'Overhead Door Shaft','Door Fabrication', 'Heat Shield', 'Door Surround Casting','Refractory','Total Steel']
            
            t_data = scaler.transform(df)
            y_pred = model.predict(t_data)
            output_df = pd.DataFrame(y_pred, columns=pred_columns)
            output_df =  output_df.astype(np.int64)
            return output_df.abs()

        st.write('')
        st.write('Predicted Output')
        pred_data = pred_model('tuned_pkl', 'scaler_pkl')
        st.write(pred_data)
        st.markdown(download(pred_data,'Predicted Output'), unsafe_allow_html=True)