import json
import numpy as np
import streamlit as st
from waferinsight.core import PATTERNS, synthetic_map, validate_map, map_records, yield_query, clusters
from waferinsight.plot import wafer_figure

st.set_page_config(page_title='WaferInsight AI', page_icon='🔬', layout='wide')
st.title('WaferInsight AI')
st.caption('WAFER MAP INTELLIGENCE · Yield analytics & spatial inspection')
st.sidebar.header('Inspection source')
pattern = st.sidebar.selectbox('Synthetic pattern', PATTERNS, index=2)
seed = st.sidebar.number_input('Random seed', min_value=0, value=42)
upload = st.sidebar.file_uploader('Upload a 2D map (.json / .csv)', type=['json','csv'])
try:
    if upload:
        a = validate_map(json.load(upload) if upload.name.endswith('.json') else np.loadtxt(upload, delimiter=','))
        st.info('User-supplied wafer map · 0 outside / 1 pass / 2 fail')
    else:
        a = synthetic_map(pattern, seed=seed)
        st.info('SYNTHETIC DEMO · Pattern is a generator setting, not a model prediction.')
    records = map_records(a)
    metrics = yield_query(records.lazy()).collect().row(0, named=True)
    cols = st.columns(4)
    for col, name, value in zip(cols, ['Valid dies','Failed dies','Yield','Clusters ≥3 dies'],
        [metrics['total_dies'], metrics['failed_dies'], f"{metrics['yield']:.2%}", len(clusters(a))]):
        col.metric(name, value)
    left, right = st.columns([3,2])
    with left:
        fig = wafer_figure(a)
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
    with right:
        st.subheader('Spatial defect clusters')
        st.dataframe(clusters(a), use_container_width=True)
        threshold = st.slider('Yield alert threshold (%)', 0, 100, 90)
        if metrics['yield']*100 < threshold:
            st.warning('Yield below threshold — engineering review required.')
        else:
            st.success('Yield meets the selected threshold.')
        st.caption('Connected-component boxes describe failed-die groups. Pattern classification below is optional; AOI segmentation and chamber attribution are not implemented.')
        if st.checkbox('Run trained wafer-pattern classifier',value=False):
            try:
                from waferinsight.predict import load_inspector,inspect_map
                @st.cache_resource
                def trained_inspector():
                    return load_inspector()
                model,card=trained_inspector()
                prediction=inspect_map(a,model)
                st.write('Predicted pattern:',prediction['pattern'])
                st.caption(f"Model score: {prediction['score']:.3f} (uncalibrated). Research model; engineering review required.")
                if not upload:st.caption('This input is synthetic and outside the real-data evaluation protocol.')
            except (ImportError,FileNotFoundError) as exc:
                st.info('Install requirements-ml.txt and ensure the verified model files are available in models/.')
            except ValueError as exc:
                st.error(str(exc))
        st.download_button('Export die records', records.write_csv(), 'die_records.csv', 'text/csv')
        st.download_button('Export inspection report', json.dumps({'source': upload.name if upload else 'synthetic',
            'metrics': metrics, 'clusters': clusters(a)}, indent=2), 'inspection.json', 'application/json')
except (ValueError, TypeError, json.JSONDecodeError) as exc:
    st.error(str(exc))
