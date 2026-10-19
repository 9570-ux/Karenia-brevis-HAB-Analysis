"""Phase 7 Streamlit research dashboard. Launch: streamlit run app.py."""
import altair as alt
import pandas as pd
import streamlit as st

import dashboard as data


@st.cache_data(show_spinner=False)
def artifacts():
    return data.load_artifacts()


@st.cache_data(show_spinner='Reading existing NOAA-derived records…')
def history():
    return data.load_history()


def bars(frame, x, y, *, title=None, percentage=False, horizontal=False):
    """One neutral category color; no health-risk color scale."""
    value = alt.X(y, title=y, axis=alt.Axis(format='.0%') if percentage else alt.Undefined) if horizontal else alt.Y(y, title=y, axis=alt.Axis(format='.0%') if percentage else alt.Undefined)
    category = alt.Y(x, sort=None, title=None) if horizontal else alt.X(x, sort=None, title=None)
    chart = alt.Chart(frame).mark_bar(color='#377e98', cornerRadiusEnd=3).encode(
        x=value if horizontal else category, y=category if horizontal else value,
        tooltip=[alt.Tooltip(x), alt.Tooltip(y, format='.4f' if percentage else ',')])
    if title:
        chart = chart.properties(title=title)
    st.altair_chart(chart.properties(height=260), width='stretch')


def overview(a):
    st.header('A recorded-data research workspace')
    st.write('Explore historical Karenia brevis observations, inspect the selected model, and run a single-observation demonstration.')
    cols = st.columns(4)
    cols[0].metric('Original NOAA records', f"{a['inspection']['row_count']:,}")
    cols[1].metric('Recorded categories', len(data.CLASSES))
    cols[2].metric('Test accuracy', f"{a['phase5']['test_metrics']['accuracy']:.2%}")
    cols[3].metric('Test macro F1', f"{a['phase5']['test_metrics']['macro_f1']:.2%}")
    left, right = st.columns([1.3,1])
    with left:
        st.subheader('From archive to inference')
        st.markdown('**NOAA HABSOS** → quality handling → approved features → **Phase 5 balanced Random Forest** → **Phase 6 inference** → **Phase 7 dashboard**')
        st.write('Target: source-recorded contemporaneous abundance category. Historical coverage: 1953-08-19 through 2024-03-25.')
        st.write(' · '.join(data.CLASSES))
    with right:
        st.subheader('Read the limitations first')
        st.warning(data.HIGH_WARNING)
        st.caption('High-category recall: 3.33%; 3 of 90 recorded high observations correctly classified in the locked Phase 5 test.')
    st.info('Start with Prediction for one recorded observation, or Historical Data to explore the original sampling record.')


def prediction():
    st.header('Single-observation prediction')
    st.write('Enter recorded measurements and calendar context. Blank environmental fields are handled only by the saved fitted imputer.')
    example = data.genuine_example()
    if st.button('Load genuine NOAA example', key='load_example'):
        for field, value in example['observation'].items():
            st.session_state['input_' + field] = value
        st.session_state.pop('prediction_result', None)
        st.session_state.pop('submitted_observation', None)
    st.caption(f"Optional example: SOURCE_ROW_NUMBER {example['SOURCE_ROW_NUMBER']}, recorded {example['recorded_date']}; actual recorded category: {example['recorded_category_for_context_only']}. Values come from existing project data.")
    with st.form('observation_form'):
        st.subheader('Location and recorded calendar')
        cols = st.columns(2)
        observation = {}
        for i, (field,label) in enumerate([('latitude','Latitude (decimal degrees)'),('longitude','Longitude (decimal degrees)')]):
            observation[field] = cols[i].number_input(label, value=None, key='input_'+field, format='%.6f', placeholder='Required')
        cols = st.columns(3)
        for i, (field,label) in enumerate([('year','Recorded year'),('month','Recorded month'),('day_of_year','Recorded day of year')]):
            observation[field] = cols[i].number_input(label, value=None, step=1, key='input_'+field, placeholder='Required')
        st.subheader('Environmental measurements · optional')
        cols = st.columns(3)
        for i, (field,label) in enumerate([('temperature','Temperature (degrees C)'),('salinity','Salinity (ppt)'),('depth','Depth (m)')]):
            observation[field] = cols[i].number_input(label, value=None, key='input_'+field, format='%.6f', placeholder='Leave blank if unavailable')
        st.caption('Use source-approved units and quality handling. These inputs are never saved as scientific observations.')
        submitted = st.form_submit_button('Run Prediction', type='primary')
    if submitted:
        st.session_state.pop('prediction_result', None)
        st.session_state.pop('submitted_observation', None)
        try:
            with st.spinner('Running the saved Phase 6 inference pipeline…'):
                result = data.predict_for_dashboard(observation)
            st.session_state['prediction_result'] = result
            st.session_state['submitted_observation'] = observation
        except ValueError as exc:
            st.error(f'Input validation: FAIL — {exc}')
        except (RuntimeError, OSError) as exc:
            st.error(f'Inference unavailable — {exc}')
    result = st.session_state.get('prediction_result')
    if result:
        st.divider()
        st.subheader('Predicted Recorded Abundance Category')
        st.metric('Source-category model output', result['predicted_class'])
        st.caption('Result belongs to the last submitted input shown below. Submit again after editing the form.')
        probs = pd.DataFrame({'Category':list(result['class_probabilities']),
                              'Predicted class probability':list(result['class_probabilities'].values())})
        left, right = st.columns([1.5,1])
        with left:
            bars(probs, 'Category:N', 'Predicted class probability:Q', percentage=True, horizontal=True)
        with right:
            st.dataframe(probs, hide_index=True, width='stretch', column_config={
                'Predicted class probability':st.column_config.NumberColumn(format='%.6f')})
            st.success('Input validation: ' + result['input_validation']['status'])
            missing = result['input_validation']['missing_environmental_fields']
            st.write('Missing environmental fields: ' + (', '.join(missing) or 'none'))
        st.info(data.PROBABILITY_NOTE)
        st.caption('Model: Phase 5 selected model · ' + result['model']['identity'])
        st.code(result['model']['artifact'], language=None)
        with st.expander('Last submitted input and model metadata'):
            st.json({'submitted_input':st.session_state['submitted_observation'], 'model':result['model']})
    st.warning(data.HIGH_WARNING)


def historical():
    st.header('Historical recorded observations')
    frame = history()
    controls = st.columns(2)
    limits = (int(frame.observation_year.min()), int(frame.observation_year.max()))
    years = controls[0].slider('Recorded years', *limits, value=limits)
    states = controls[1].multiselect('Recorded states', sorted(frame.STATE_ID.dropna().unique()), default=sorted(frame.STATE_ID.dropna().unique()))
    subset = data.filter_history(frame, years, states)
    st.caption(f'{len(subset):,} of {len(frame):,} source records selected. Charts describe sampling observations, not bloom prevalence. No values are imputed for visualization.')
    if subset.empty:
        st.info('No observations match these filters.')
        return
    cols = st.columns(3)
    cols[0].metric('Selected observations', f'{len(subset):,}')
    cols[1].metric('Missing recorded category', f'{subset.CATEGORY.isna().sum():,}')
    cols[2].metric('Eligible existing labels', f'{subset.observed_abundance_category.notna().sum():,}')
    left, right = st.columns(2)
    with left:
        bars(data.category_counts(subset), 'Recorded category:N', 'Observations:Q', title='Original source categories')
        annual = subset.groupby('observation_year').size().reset_index(name='Observations')
        st.line_chart(annual, x='observation_year', y='Observations', x_label='Recorded year', color='#377e98')
    with right:
        monthly = subset.groupby('observation_month').size().reindex(range(1,13), fill_value=0).rename_axis('Recorded month').reset_index(name='Observations')
        bars(monthly, 'Recorded month:O', 'Observations:Q', title='Observations by calendar month')
        st.caption('Category chart uses original recorded CATEGORY, including categories on rows excluded from supervised evaluation. Missing categories are counted above and are not assigned to another class. Years/months are recorded collection dates; uneven effort and incomplete 2024 affect counts.')
    st.subheader('Recorded environmental distributions')
    for col, (field,(label,unit)) in zip(st.columns(3), data.ENV_UNITS.items()):
        with col:
            observed = subset[field].dropna()
            st.caption(f'{label}: {len(observed):,} available; {subset[field].isna().sum():,} missing after existing Phase 3 QA handling.')
            if len(observed):
                groups, edges = pd.cut(observed, bins=25, retbins=True)
                histogram = pd.DataFrame({'Bin start':edges[:-1], 'Bin end':edges[1:],
                                          'Observations':groups.value_counts(sort=False).to_numpy()})
                chart = alt.Chart(histogram).mark_bar(color='#377e98').encode(
                    x=alt.X('Bin start:Q',title=f'{label} ({unit})'), x2=alt.X2('Bin end:Q'),
                    y=alt.Y('Observations:Q'), tooltip=['Bin start:Q','Bin end:Q','Observations:Q'])
                st.altair_chart(chart.properties(height=200), width='stretch')
    st.caption('Each histogram excludes only missing values in its own chart subset. Values are not clipped; caution/source-converted readings remain under the established QA policy. Underlying files are unchanged.')
    st.subheader('Historical recorded observation locations')
    points, stride = data.map_subset(subset)
    st.caption(f'Map: {len(points):,} actual coordinates, every {stride}-th coordinate-complete row in source order (maximum 4,000). Display subset only; repeated locations can overlap. These are historical collection locations, not real-time HAB locations.')
    # Coordinate scatter avoids external map tiles and works without a network.
    chart = alt.Chart(points).mark_circle(size=18,opacity=.35,color='#377e98').encode(
        longitude='LONGITUDE:Q', latitude='LATITUDE:Q',
        tooltip=['SOURCE_ROW_NUMBER:Q','SAMPLE_DATE_ISO:N','STATE_ID:N','CATEGORY:N']).project(type='mercator')
    st.altair_chart(chart.properties(height=400), width='stretch')
    st.caption('Geographic coordinate plot uses Mercator display projection only; no coordinates or geographic model features are generated.')


def performance(a):
    st.header('Locked model performance')
    metrics = a['phase5']['test_metrics']
    for col, (label,key) in zip(st.columns(3), [('Test accuracy','accuracy'),('Test macro F1','macro_f1'),('Balanced accuracy','balanced_accuracy')]):
        col.metric(label, f'{metrics[key]:.2%}')
    st.warning(data.HIGH_WARNING)
    st.write(f"High-category recall: **{metrics['per_class']['high']['recall']:.2%}** · **3 of 90** recorded high observations correctly classified; **87 missed**.")
    st.caption('Metrics are read from locked Phase 5 reports. The Phase 4/5 comparison reuses the same test cohort; it is not independent prospective validation.')
    st.subheader('Phase 4 vs Phase 5')
    st.dataframe(a['comparison'], hide_index=True, width='stretch')
    comparison = a['comparison'].melt(id_vars='phase',value_vars=['accuracy','macro_f1','balanced_accuracy'],var_name='Metric',value_name='Score')
    chart = alt.Chart(comparison).mark_bar().encode(x='Metric:N',xOffset='phase:N',y=alt.Y('Score:Q',scale=alt.Scale(domain=[0,1])),
             color=alt.Color('phase:N',scale=alt.Scale(range=['#aab8c4','#377e98'])),tooltip=['phase','Metric',alt.Tooltip('Score:Q',format='.2%')])
    st.altair_chart(chart.properties(height=260), width='stretch')
    st.subheader('Phase 5 confusion matrix · existing locked figure')
    st.image(str(data.ROOT / 'reports/figures/phase5_final_confusion_matrix.png'), width='stretch')
    st.subheader('Model-associated feature importance')
    st.image(str(data.ROOT / 'reports/figures/phase5_feature_importance.png'), width='stretch')
    st.caption('Forest impurity importance describes model associations; it does not establish environmental causality. Missingness indicators are fitted internal preprocessing signals.')
    with st.expander('Full feature-importance table'):
        st.dataframe(a['importance'], hide_index=True, width='stretch')


def dataset_information(a):
    st.header('Dataset information and provenance')
    st.write('NOAA National Centers for Environmental Information · HABSOS · accession **0120767**, version **8.8** · Karenia brevis observations.')
    st.write(f"Original records: **{a['inspection']['row_count']:,}** · raw bytes: **{a['inspection']['size_bytes']:,}** · 1953-08-19–2024-03-25 · recorded states AL, FL, MS, TX.")
    st.markdown('[Official NOAA metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0120767) · [Original archived CSV](https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv)')
    st.caption('DOI: not provided in the inspected metadata. The exact archived title and supplied dataset citation are reproduced below from docs/dataset.md.')
    st.write('Raw: data/raw/habsos_20240430.csv · Processed: data/processed/habsos_cleaned.csv (all source rows preserved) · Approved predictors/targets: data/ml/.')
    st.subheader('Approved source predictors')
    st.dataframe(pd.DataFrame({'Public input':list(data.inference.FIELDS),'Existing predictor':list(data.inference.FIELDS.values()),
                  'Unit':['degrees C','ppt','m','decimal degrees','decimal degrees','calendar year','calendar month','calendar day']}),hide_index=True,width='stretch')
    frame = history()
    missing = pd.DataFrame([{'Predictor':field,'Missing':int(frame[field].isna().sum()),'Missing percentage':float(frame[field].isna().mean()*100)} for field in data.inference.FIELDS.values()])
    st.subheader('Existing feature missingness · no dashboard imputation')
    st.dataframe(missing, hide_index=True, width='stretch')
    st.write('Classes: ' + ', '.join(data.CLASSES))
    st.write('pH, turbidity, TDS, dissolved oxygen and chlorophyll-a: Not available in this dataset.')
    st.info('Environmental missingness, source QA cautions/conversions, missing source times, undocumented coordinate datum and uneven sampling remain. Salinity is not a measured TDS value.')
    with st.expander('Verified source documentation and citation'):
        st.markdown((data.ROOT / 'docs/dataset.md').read_text(encoding='utf-8'))


def about():
    st.header('About this research demonstration')
    phases = [('1','NOAA/HABSOS dataset collection'),('2','Preprocessing and data-quality handling'),
              ('3','Feature engineering and target definition'),('4','Baseline machine learning'),
              ('5','Model improvement and error analysis'),('6','Final inference pipeline'),
              ('7','Software dashboard'),('8','Hardware demonstration — NOT IMPLEMENTED')]
    st.dataframe(pd.DataFrame(phases, columns=['Phase','Scope']), hide_index=True, width='stretch')
    st.warning(data.LIMITATION)
    st.write('Phase 8 hardware integration is not part of this dashboard implementation.')
    st.write('No model is trained here. User inputs and predictions stay in the current session; they are never appended to scientific datasets. Historical chart subsets do not change the archive.')
    st.write('The target is a source-reported contemporaneous abundance category. Toxicity, future bloom onset, health risk, unseen-site generalization and operational alerting have not been validated.')
    st.caption('Dataset lineage, software versions, input handling and evaluation limitations are documented in docs/dataset.md and docs/phase3–phase7 documents.')


def main():
    st.set_page_config(page_title='HABSOS | Abundance research', page_icon='🌊', layout='wide')
    st.markdown('''<style>
    .block-container {padding-top:2rem;max-width:1440px;}
    h1,h2,h3 {letter-spacing:-.025em;}
    [data-testid="stMetric"] {background:#edf3f7;border:1px solid #dae4ec;border-radius:12px;padding:18px;}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"] {color:#16354b;}
    </style>''',unsafe_allow_html=True)
    st.sidebar.markdown('## HABSOS Research\nNOAA archive · Software demonstration')
    page = st.sidebar.radio('Workspace', data.PAGES, key='navigation')
    st.sidebar.divider()
    st.sidebar.caption('Phase 5 model · Phase 6 inference · Phase 7 interface')
    st.sidebar.info('Historical archive through March 2024. No live monitoring feed.')
    st.title('HAB Abundance Monitoring & Prediction Dashboard')
    st.caption('NOAA HABSOS-based Machine Learning Demonstration')
    st.warning(data.LIMITATION)
    a = artifacts()
    pages = {'Overview':lambda:overview(a),'Prediction':prediction,'Historical Data':historical,
             'Model Performance':lambda:performance(a),'Dataset Information':lambda:dataset_information(a),
             'About / Limitations':about}
    pages[page]()
    st.divider()
    st.caption('NOAA HABSOS accession 0120767 · Recorded abundance research · Phase 8 not implemented')


if __name__ == '__main__':
    main()
