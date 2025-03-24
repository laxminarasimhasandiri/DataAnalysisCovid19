#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:58:26 2023

@author: ravi
"""
# Import necessary libraries

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Creating dataframe
covid_day_wise = pd.read_csv('day_wise.csv')

covid_full_grouped = pd.read_csv('full_grouped.csv')

covid_country_wise_latest = pd.read_csv('country_wise_latest.csv')

covid_worldometer_data = pd.read_csv('worldometer_data.csv')

# Convert the 'Date' column to a datetime object
covid_full_grouped['Date'] = pd.to_datetime(covid_full_grouped['Date'])

covid_day_wise['Date'] = pd.to_datetime(covid_day_wise['Date'])

np.bool=np.bool_
np.float=np.float64

#########################Scenario-1#########################

## Line chart for Death per 100 cases over time

# Set the style for the plot
sns.set(style="whitegrid")

# Plotting the line chart
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Deaths / 100 Cases', data=covid_day_wise, marker='o')

# Adding labels and title
plt.title('Death Per 100 Cases Over Time')
plt.xlabel('Time')
plt.ylabel('No of Death per 100 cases')

# Display the plot
plt.show()

#########################Scenario-2#########################

## Line chart for Recovered / 100 Cases

# Plotting the area chart
plt.figure(figsize=(10, 6))


# Fill the area under the line
sns.lineplot(x='Date', y='Recovered / 100 Cases', data=covid_day_wise, marker='o', color='blue')
plt.fill_between(covid_day_wise['Date'], covid_day_wise['Recovered / 100 Cases'], color='skyblue', alpha=0.4)

# Adding labels and title
plt.title('Recovered / 100 Cases Over Time (Area Chart)')
plt.xlabel('Time')
plt.ylabel('No of Recovered / 100 Cases')

# Display the plot
plt.show()


#########################Scenario-3#########################

## Line chart for active and Deaths

# Set the style for the plot
sns.set(style="whitegrid")

# Plotting the line chart
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Active', data=covid_day_wise, marker='o', color='red', label='Active Cases')

sns.lineplot(x='Date', y='Deaths', data=covid_day_wise, marker='p', color='blue', label='Deaths')

# Adding labels and title
plt.title('Active Cases and Deaths Over Time')
plt.xlabel('Time')
plt.ylabel('No of Active Cases and Deaths')

plt.legend()

# Display the plot
plt.show()

#########################Scenario-4#########################

## Line chart for Active, confirmed cases and Deaths

# Set the style for the plot
sns.set(style="whitegrid")

# Plotting the line chart
plt.figure(figsize=(10, 6))

sns.lineplot(x='Date', y='Active', data=covid_day_wise, marker='o', color='green', label='Active Cases')

sns.lineplot(x='Date', y='Confirmed', data=covid_day_wise, marker='p', color='blue', label='Confirmed')

sns.lineplot(x='Date', y='Deaths', data=covid_day_wise, marker='p', color='red', label='Deaths')

# Adding labels and title
plt.title('Active, Confirmed and Deaths Over Time')
plt.xlabel('Time')
plt.ylabel('No of Active cases, Confirmed Cases and Deaths(Million)')

plt.legend()

# Display the plot
plt.show()

#########################Scenario-5#########################

## Canada line chart

country = 'Canada'
country_data = covid_full_grouped[covid_full_grouped['Country/Region'] == country]

country_data['Date'] = pd.to_datetime(country_data['Date'])

plt.figure(figsize=(10, 6))

#Line chart for death/confirmed/Active/recovered
sns.lineplot(x='Date', y=country_data['Confirmed'], data=covid_full_grouped, color='red', label='Confirmed Cases')
sns.lineplot(x='Date', y=country_data['Deaths'], data=covid_full_grouped, color='blue', label='Deaths')

sns.lineplot(x='Date', y=country_data['Active'], data=covid_full_grouped, color='green', label='Active')
sns.lineplot(x='Date', y=country_data['Recovered'], data=covid_full_grouped, color='black', label='Recovered')

# Adding labels and title
plt.title('COVID 19 in  Canada')
plt.xlabel('Time')
plt.ylabel('No of Covid 19 cases in Canada')

plt.show()

#########################Scenario-6#########################

# Create a pie chart about Distribution of highest confirmed cases by WHO Region
plt.figure(figsize=(10, 6))

# Define colors for the pie chart
cases_by_region = covid_full_grouped.groupby("WHO Region")["Confirmed"].sum()

colors = sns.color_palette('Set2', len(cases_by_region))

# Create the pie chart
plt.pie(cases_by_region, labels=cases_by_region.index, autopct='%1.1f%%', colors=colors, startangle=140)

plt.title("Highest confirmed cases by WHO Region")

plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

plt.figure(figsize=(12, 6))  # Optional: Set the figure size

plt.show()

#########################Scenario-7#########################

# Create a pie chart about Distribution of total test performed by WHO Region
plt.figure(figsize=(10, 6))

# Define colors for the pie chart
test_by_region = covid_worldometer_data.groupby("WHO Region")["TotalTests"].sum()

colors = sns.color_palette('Set2', len(test_by_region))

# Create the pie chart
plt.pie(test_by_region, labels=test_by_region.index, autopct='%1.1f%%', colors=colors, startangle=140)

plt.title("COVID Test Performed in WHO Region")

plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

plt.figure(figsize=(12, 6))  # Optional: Set the figure size

plt.show()

#########################Scenario-8#########################

#Craete a Bar plot
covid_melted = pd.melt(covid_full_grouped, id_vars=['Date', 'Country/Region', 'WHO Region'], value_vars=['Deaths', 'Active', 'Recovered'])
plt.figure(figsize=(10, 6))


ax = sns.barplot(y="WHO Region",x="value", hue="variable", data=covid_melted, palette="muted",ci=None)

# Add labels above each bar
for p in ax.patches:
    ax.annotate(f'{p.get_width():.0f}', (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height()), ha='left', va='baseline')

ax.set_title("Total No of Cases in WHO Region")

plt.xlabel('Total No of COVID Cases(Million)')

plt.show()

#########################Scenario-9#########################

# Create the bar chart for all who regions for New cases
plt.figure(figsize=(10, 6))

plt.xticks(rotation=90)

cases_by_region = covid_country_wise_latest.groupby("WHO Region")["New cases"].sum()

df_sorted = cases_by_region.sort_values()
print(df_sorted)

ax1 = sns.barplot(x=df_sorted.index, y=df_sorted.values, ci=None)
ax1.set_xlabel("WHO Region")
ax1.set_ylabel("New cases")
ax1.set_title("Number of New cases by WHO Region")


# Add labels above each bar
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.show()

#########################Scenario-10#########################

## covid 19 bar chart for India

plt.figure(figsize=(10, 6))

india_data = covid_country_wise_latest[covid_country_wise_latest['Country/Region']=='India']

# Get values for each category
india_Deaths = india_data['Deaths'].sum()
india_Confirmed = india_data['Confirmed'].sum()
india_Active = india_data['Active'].sum()
india_Recovered = india_data['Recovered'].sum()

India_info = [india_Confirmed, india_Recovered, india_Active, india_Deaths]

ax=sns.barplot(x=['Confirmed', 'Recovered', 'Active', 'Deaths'], y=India_info)

sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)

ax.set_title('Covid 19 in India')

# Add labels above each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.ylabel('Total No of COVID Cases(Million)')
plt.show()

#########################Scenario-11#########################

## covid 19 bar chart for US

plt.figure(figsize=(10, 6))

US_data = covid_country_wise_latest[covid_country_wise_latest['Country/Region']=='US']

# Get values for each category
US_Deaths = US_data['Deaths'].sum()
US_Confirmed = US_data['Confirmed'].sum()
US_Active = US_data['Active'].sum()
US_Recovered = US_data['Recovered'].sum()

US_info = [US_Confirmed, US_Active, US_Recovered, US_Deaths]

ax=sns.barplot(x=['Confirmed', 'Active', 'Recovered', 'Deaths'], y=US_info)

sns.color_palette("mako", as_cmap=True)

ax.set_title('Covid 19 in US')

# Add labels above each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.ylabel('Total No of COVID Cases(Million)')
plt.show()

#########################Scenario-12#########################

#20 less affected countires in COVID 19

plt.figure(figsize=(10, 6))

# Create a function to find and display the top 20 countries with the lowest number of deaths
def get_lowest_countries_by_Active(covid_country_wise_latest, top_n=20, custom_color='purple'):
    lowest_Active = covid_country_wise_latest.nsmallest(top_n, 'Active')

    plt.xticks(rotation=90)
    plt.figure(figsize=(30, 20))
    ax = sns.barplot(
        y='Country/Region',
        x='Active',
        data=lowest_Active,
        palette=[custom_color] * top_n
        
    )

    # Add labels to the top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.0f}', (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height()), ha='left', va='baseline')

    plt.title(f'Top {top_n} Countries with the Lowest Number of Active')
    plt.xlabel('Number of Active')
    plt.ylabel('Country/Region')
    

# Example usage to find and display the top 20 countries with the lowest deaths using a custom color
get_lowest_countries_by_Active(covid_country_wise_latest, top_n=20, custom_color='pink')

# Add labels above each bar

plt.show()

#########################Scenario-13#########################

#correlation Heatmap

plt.figure(figsize=(10, 6))

# Calculate the correlation matrix
correlation_matrix = covid_worldometer_data[['TotalCases', 'TotalDeaths', 'Serious,Critical','Population']].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Set3', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#########################Scenario-14#########################

#Create a Bar plot for continents Vs Tot Cases/1M pop', 'Tests/1M pop
plt.figure(figsize=(10, 6))

covid_melted = pd.melt(covid_worldometer_data, id_vars=['Continent','Country/Region', 'WHO Region'], value_vars=['Tot Cases/1M pop', 'Tests/1M pop'])

ax = sns.barplot(y="Continent",x="value", hue="variable", data=covid_melted, palette="muted",ci=None)

ax.set_title("Number of Cases by WHO Region")

# Add labels above each bar
for p in ax.patches:
    ax.annotate(f'{p.get_width():.0f}', (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height()), ha='left', va='baseline')

plt.xlabel('No of COVID cases')
plt.show()

#########################Scenario-15#########################
# create stacked bar chart for totalcases and activecases

plt.figure(figsize=(10, 6))

# Calculate the total cases and active cases for all WHO regions
continent_totals = covid_worldometer_data.groupby('WHO Region')[['TotalCases', 'ActiveCases']].sum().reset_index()

# Sort the DataFrame by 'TotalCases' in descending order
continent_totals_sorted = continent_totals.sort_values(by='TotalCases', ascending=False)

ax = sns.barplot(x='WHO Region', y='TotalCases', data=continent_totals_sorted, color='skyblue', label='TotalCases')
ax1 = sns.barplot(x='WHO Region', y='ActiveCases', data=continent_totals_sorted, color='r', label='ActiveCases', bottom=continent_totals_sorted['TotalCases'])

# Set labels and title
plt.title("Total cases and ActiveCases by WHO Region")
plt.xlabel("WHO Region")
plt.ylabel("No of COVID Cases(Million)")

# Display legend
plt.legend(title="Metric", loc="upper right")

# Add labels above each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.show()

#########################Scenario-16#########################

## combination chart for confirmed and recovered cases
plt.figure(figsize=(10, 6))

ax = sns.barplot(x="WHO Region", y="Confirmed", data=covid_full_grouped, ci=None, color='skyblue')
ax.set_xlabel("WHO Region")
ax.set_ylabel("Confirmed")
ax.set_title("Number of confirmed by WHO Region")

# Add labels above each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')

plt.xticks(rotation=90)  # Optional: Rotate x-axis labels for better readability

#create multiple line chart

sns.set(style="whitegrid")

sns.lineplot(x='WHO Region', y='Recovered', data=covid_full_grouped, label='Recovered', color='red')

plt.xlabel('WHO Region')
plt.ylabel('Confirmed Cases')
plt.title('Comparison of confirmed cases and recovered cases by WHO Region')

######################### Predictive Analysis - Scenario-17#########################

# Extract features (X) and target variable (y)
X = np.array(covid_full_grouped['Date']).astype(np.int64) // 10**9  # Convert dates to timestamps in seconds
X = X.reshape(-1, 1)  # Reshape to a 2D array
y = covid_full_grouped['New cases']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set and future dates
y_pred = model.predict(X_test)
future_dates = pd.date_range(start=covid_full_grouped['Date'].max(), periods=30)  # Adjust the number of periods as needed
future_timestamps = np.array(future_dates).astype(np.int64) // 10**9
future_predictions = model.predict(future_timestamps.reshape(-1, 1))

# Convert numeric representation back to datetime for plotting
X_test_dates = X_test.flatten() * np.timedelta64(1, 's') + np.datetime64('1970-01-01')
future_dates = future_timestamps * np.timedelta64(1, 's') + np.datetime64('1970-01-01')

# Plot the line chart
plt.figure(figsize=(10, 6))

# Actual cases
plt.plot(covid_full_grouped['Date'], covid_full_grouped['New cases'], label='Actual Cases', color='skyblue')

# Predicted cases on the test set
plt.plot(X_test_dates, y_pred, label='Predicted Cases (Test Set)', color='black')

# Predicted cases for future dates
plt.plot(future_dates, future_predictions, label='Predicted Cases (Future Dates)', linestyle='dashed', color='red')

plt.xlabel('Date')
plt.ylabel('New Cases')
plt.title('COVID-19 Predictive Analysis - Line Chart for New Cases')
plt.legend()

# Format x-axis ticks to display dates properly
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

plt.show()

######################### Predictive Analysis - Scenario-18 #########################

# Feature selection
features = covid_day_wise[['Confirmed', 'Deaths']]
target = covid_day_wise['Active']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# Plotting the predicted values against the actual values as a line plot
plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Active Cases', color='green', linestyle='-', linewidth=2)
plt.plot(predictions, label='Predicted Active Cases', color='black', linestyle='--', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Active Cases')
plt.title('Actual vs Predicted Active Cases')
plt.legend()
plt.show()

