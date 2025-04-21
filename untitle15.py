# fraud_detection.py
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    import sys
    print("Plotly is required but not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.express as px
    import plotly.graph_objects as go

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    st.title("Fraud Detection Dashboard")
    
    # Data loading
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("Fraud_Analysis_Dataset.csv")
        except Exception as e:
            st.error(f"Failed to load data: {str(e)}")
            return None
    
    df = load_data()
    if df is None:
        return
    
    # Preprocessing
   