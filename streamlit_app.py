import leafmap.foliumap as leafmap
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import folium
import json

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Reports", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Plastic detection reports!!")

df = pd.read_csv("synthetic_data.csv")

col1, col2 = st.columns((2))
df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")
df["latitude"] = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)

# Getting the min and max date 
startDate = pd.to_datetime(df["time"]).min()
endDate = pd.to_datetime(df["time"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["time"] >= date1) & (df["time"] <= date2)].copy()

st.sidebar.header("Choose your filter: ")
# Create for Region
region = st.sidebar.multiselect("Pick your Region", df["region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["region"].isin(region)]

# Create for State
state = st.sidebar.multiselect("Pick the State", df2["state"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["state"].isin(state)]

# Create for City
city = st.sidebar.multiselect("Pick the City",df3["city"].unique())

# Filter the data based on Region, State and City

if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["state"].isin(state)]
elif state and city:
    filtered_df = df3[df["state"].isin(state) & df3["city"].isin(city)]
elif region and city:
    filtered_df = df3[df["region"].isin(region) & df3["city"].isin(city)]
elif region and state:
    filtered_df = df3[df["region"].isin(region) & df3["state"].isin(state)]
elif city:
    filtered_df = df3[df3["city"].isin(city)]
else:
    filtered_df = df3[df3["region"].isin(region) & df3["state"].isin(state) & df3["city"].isin(city)]
    
# Convert string columns to numeric
numeric_columns = ["plastic", "slippers", "plastic wrapper", "plastic spoon", "plastic bag", "thermocol",
                   "plastic plate", "plastic bottle", "plastic cup"]

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

# Group by plastic category and calculate total counts
plastic_category_df = df[numeric_columns].sum().reset_index()
plastic_category_df.columns = ["Plastic Category", "Count"]

"""

        Plot 1 Bar chart
        
"""

with col1:
    st.subheader("Plastic Category Distribution")
    
    # Group by plastic category and calculate total counts based on the filtered data
    filtered_category_df = filtered_df[numeric_columns].sum().reset_index()
    filtered_category_df.columns = ["Plastic Category", "Count"]
    
    fig_category = px.bar(filtered_category_df, x="Plastic Category", y="Count",
                          text=filtered_category_df["Count"].apply(lambda x: f"{x:,}"))

    fig_category.update_traces(texttemplate="%{text}", textposition="outside")
    fig_category.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig_category, use_container_width=True, height=300)
    
"""

        Plot 2 Pie chart
        
"""

with col2:
    st.subheader("Region-wise Plastic Count")
    fig_region = px.pie(filtered_df, values="plastic", names="region",
                        hole=0.5)
    fig_region.update_traces(textinfo="percent+label", textposition="outside")
    # fig_region.update_layout(title="Region-wise Plastic Count")
    st.plotly_chart(fig_region, use_container_width=True)

"""

        Plot 1 Bar chart - data download 
        
"""

cl1, cl2 = st.columns((2))
with cl1:
    with st.expander("Category wise plastics data"):
        st.write(filtered_category_df.T.style.background_gradient(cmap="Blues"))
        csv = filtered_category_df.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Category.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

"""

        Plot 2 Pie chart - data download 
        
"""

with cl2:
    with st.expander("Region wise plastic data"):
        region = filtered_df.groupby(by = "region", as_index = False)["count"].sum()
        st.write(region.T.style.background_gradient(cmap="Oranges"))
        csv = region.to_csv(index = False).encode('utf-8')
        st.download_button("Download Data", data = csv, file_name = "Region.csv", mime = "text/csv",
                        help = 'Click here to download the data as a CSV file')
        
"""

        Plot 3 line chart - comparison of amount of plastics categories
        
"""     

# Add month and year columns to the DataFrame
filtered_df["year"] = filtered_df["time"].dt.year
filtered_df["month"] = filtered_df["time"].dt.month
filtered_df["month_year"] = filtered_df["time"].dt.strftime('%b-%Y')  # Format as "Jan-2023"

st.subheader('Over time plastic growth by category')

# Define the plastic category columns
plastic_categories = ["plastic", "slippers", "plastic wrapper", "plastic spoon", "plastic bag", "thermocol",
                   "plastic plate", "plastic bottle", "plastic cup"]

# Create a multi-line chart for each plastic category
fig2 = go.Figure()

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()

for category in plastic_categories:
    category_data = filtered_df.groupby(["year", "month", "month_year"])[category].sum().reset_index()
    fig2.add_trace(go.Scatter(x=category_data["month_year"], y=category_data[category], mode='lines', name=category))
    
    # Merge the data for this category into the merged_data DataFrame
    if merged_data.empty:
        merged_data = category_data
    else:
        merged_data = pd.merge(merged_data, category_data, on=["year", "month", "month_year"])
merged_data = merged_data.drop(['year','month'],axis=1)
fig2.update_layout(
    xaxis_title="Time period",
    yaxis_title="Amount of plastic",
    # template="gridon",
    height=580,
    width=1100
)

fig2.update_xaxes(type='category')  # Ensure x-axis is treated as a category
st.plotly_chart(fig2, use_container_width=True)

"""

        Plot 3 line chart - data download 
        
"""     

with st.expander("View Data"):
    st.write(merged_data.style.background_gradient(cmap="Blues"))
    csv = merged_data.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')
    
"""

        Plot 4 geo plot - heat map - hotspots of plastics
        
"""  
   
# Create a Streamlit app
st.title('Heatmap of Latitude and Longitude Data')

filepath = "synthetic_data.csv"
m = leafmap.Map(height="400px", width="800px",center=(20.9238878,79.6734611), zoom=6,draw_control=False, measure_control=False, fullscreen_control=False, attribution_control=True)
m.add_tile_layer(url="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", name="Google Satellite", attribution="Google")
m.add_heatmap(
    filtered_df,
    latitude="latitude",
    longitude="longitude",
    value="count",
    name="Heat map",
    radius=20,
)
m.fit_bounds([[31.8100486,75.9797811], [12.9539454,77.4657859]])
m.to_streamlit(scrolling=True)

"""

        Plot 5 TreeMap - hirachy of all states containing different level of plastics
        
"""     

# Create a TreeMap
st.subheader("Hierarchical view of plastic category")
heat = filtered_df[filtered_df['count'] > 0]
# for c in col:
fig3 = px.treemap(heat, path=["region", "state", "city"], values="count", hover_data=["plastic"],
                color="count", color_continuous_scale='rdbu_r')

# fig3.update_layout(width=800, height=650)
fig3.update_traces(root_color="white",marker=dict(cornerradius=1))
fig3.update_layout(width=800, height=650,margin = dict(t=50, l=25, r=25, b=25))
st.plotly_chart(fig3, use_container_width=True)

"""

        Plot 6 heatmap - weather and plastics trend analysis
        
"""     

# Create histograms for the "plastic_area" variable
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=filtered_df["plastic_area"],
    histnorm='percent',
    name='Plastic Area',
    xbins=dict(
        start=filtered_df["plastic_area"].min(),
        end=filtered_df["plastic_area"].max(),
        size=5000  # Adjust the bin size as needed
    ),
    marker_color='#571B7E',
    opacity=0.85
))

fig.update_layout(
    title_text='Distribution of Plastic Area',
    xaxis_title_text='Plastic Area',
    yaxis_title_text='Percentage',
    bargap=0.1,
    bargroupgap=0.5,
    template='plotly_dark'  # Use a dark theme
)

# Subheader for the second plot
st.subheader('Pollution Trend in Different Weather Conditions')

# Pivot the data to create a grid
heatmap_data = filtered_df.pivot_table(index='weather', columns='temperature', values='count', aggfunc='sum')

# Create a heatmap trace
heatmap = go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Viridis',  # You can choose another colorscale if desired
)

# Create a layout for the heatmap
layout = go.Layout(
    xaxis=dict(title='Temperature'),
    yaxis=dict(title='Weather'),
)

# Create a figure and add the heatmap trace
fig_heatmap = go.Figure(data=[heatmap], layout=layout)

# Use st.columns to create two columns
col1, col2 = st.columns(2)

# Add the first plot to the first column
col1.plotly_chart(fig, use_container_width=True)

# Add the second plot to the second column
col2.plotly_chart(fig_heatmap, use_container_width=True)

"""

        Data summary
        
"""     

st.subheader("A short summary about month wise detected plastic data")

with st.expander("Click me to view"):
    num_rows_to_display = 10  # Adjust the number of rows to display as needed
    df_sample = df[:num_rows_to_display][["city", "state", "region", "plastic", "slippers", "plastic wrapper"]]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month-wise Sub-Category Table")
    # Assuming you have a filtered_df DataFrame based on your data
    filtered_df["time"] = pd.to_datetime(filtered_df["time"], format="%d-%m-%Y")
    sub_category_Year = pd.pivot_table(data=filtered_df, values="plastic", index=["city"], columns=filtered_df["time"].dt.month_name())
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))
    
"""

        Plot 7 Goe plot - state wise plastic count
        
"""     

st.title('State wise plastics detected')
# Load GeoJSON data
with open('india_state_geo.json') as file:
    geojsonData = json.load(file)
statecases = filtered_df.groupby('state')['count'].sum().reset_index()

# Modify state names in GeoJSON data
for i in geojsonData['features']:
    if(i['properties']['NAME_1'] == 'Orissa'):
        i['properties']['NAME_1'] = 'Odisha'
    elif(i['properties']['NAME_1'] == 'Uttaranchal'):
        i['properties']['NAME_1'] = 'Uttarakhand'
    i['id'] = i['properties']['NAME_1']

# Create a dictionary to map state names to counts
state_count_dict = dict(zip(filtered_df['state'], filtered_df['count']))

# Create a Folium map
map_choropleth = folium.Map(location=[20.5937, 78.9629], zoom_start=4, tiles="cartodbpositron")

# Add Choropleth layer to the map
folium.Choropleth(
    geo_data=geojsonData,
    data=statecases,
    name='CHOROPLETH',
    key_on='feature.id',
    columns=['state', 'count'],
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    overlay=filtered_df['state'],
    legend_name='Total count of plastic',
    highlight=True,
    nan_fill_color="white",
    reset=True
).add_to(map_choropleth)

# Add tooltips to show state name and count
tooltip = folium.GeoJsonTooltip(fields=['NAME_1'], aliases=['State'], sticky=True)
folium.GeoJson(
    geojsonData,
    tooltip=tooltip,
    style_function=lambda x: {
        'fillColor': 'transparent',
        'color': 'transparent'
    },
    highlight_function=lambda x: {
        'weight': 3,
        'fillColor': 'YlOrRd',
        'color': 'YlOrRd',
        'fillOpacity': 0.7
    }
).add_to(map_choropleth)

# Save the map as an HTML file
map_choropleth.save("choropleth_map.html")

# Display the Folium map in Streamlit
st.components.v1.html(open("choropleth_map.html", "r").read(), height=600)

"""

        Plot 3 line chart - model comparison
        
"""     

st.title('Model Scores Comparison')

# Allow the user to select which scores to compare
selected_columns = st.multiselect('Select scores to compare:', filtered_df.columns[18:23])

# If nothing is selected, compare all scores
if not selected_columns:
    selected_columns = filtered_df.columns[18:23]

# Filter the DataFrame based on user selection
selected_df = filtered_df[selected_columns]
selected_df['time'] = filtered_df['time']

selected_df = selected_df.sort_values(by='time')

# Create an interactive line chart
fig = go.Figure()

num_points = 150  # Adjust the number of points for smoother lines

for col in selected_columns:
    x_timestamp = pd.to_numeric(selected_df['time'])  # Convert timestamp to numerical value
    x_smooth = np.linspace(x_timestamp.min(), x_timestamp.max(), num_points)
    y_smooth = np.interp(x_smooth, x_timestamp, selected_df[col])
    
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(x_smooth),  # Convert back to timestamp for x-axis
        y=y_smooth,
        mode='lines+markers',
        name=col
    ))

fig.update_layout(
    title='Model scores monitoring for Drift and other possible factors for model degraation',
    xaxis_title='Time',
    yaxis_title='Scores',
    height=600,  # Adjust the height as needed
    width=1000   # Adjust the width as needed
)

fig.update_yaxes(range=[40, 120])

st.plotly_chart(fig, use_container_width=True)


st.subheader("Download the complete csv used here")
with st.expander("View Data"):
    st.write(filtered_df.iloc[:5, :].style.background_gradient(cmap="Oranges"))

csv = df.to_csv(index = False).encode('utf-8')
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")