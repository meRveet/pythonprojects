import matplotlib.pyplot as plt # for plotting
import pandas as pd # for wrangling data
import numpy as np # managing numbers
import collections as c
import seaborn as sns

# Reading the Dataset
load("D:/Dropbox/R/123.RData")
setwd("D:/Dropbox/R")

df= pd.read_csv("nhanes_2015_2016.txt")

#Exploring the Datasets
df.shape
df.head()
df.info #
df.describe() # Overview on Mean, Count, Std, Min, Max and Interquatile range

#Checking for missing data
# dropna=False to keep NaN values in count, sort_index() to sort in order
df.value_counts(dropna=False).sort_index() 
df.ALQ101.isna.sum() #verfiying that missing data matches dict
(df.BPXDI2>df.BPXSY2).sum() # Verifying if any BP measures are unusual when Di>SY

#Visualising Dataset
plt.hist(df.BPXSY2) # Quick Plotting to Visualise dataset
plt.show() #Shows plot
plt.clf()  #Clears plot

#Creating new columns
df['drink'] = 0


#Dichotomise the drinking data
df.drink[((df.ALQ101 ==1) | (df.ALQ110==1) | (df.ALQ130 == any(range(1,16))))] = 1
df.drink.sum()

#Dichotomise the smoking data
df.SMQ020[(df.SMQ020>2)] = 0
df.SMQ020[(df.SMQ020<3)] #check if data has been edited. 


#Visualising Smoking vs Drinking
fig.ax = plt.subplots(figsize=(12,6))
x= df['drink'][df.DMDCITZN<9]
y= df['SMQ020'][df.DMDCITZN<9]
z= df.DMDCITZN[df.DMDCITZN<9]
plt.scatter(x,y,z)
plt.show()
plt.clf()


#Visualising Drinking Range
a = df.ALQ130[(df.ALQ130 >=1) & (df.ALQ130<=15)] # Filtering Drinkers
a
i = c.Counter(df.ALQ130[(df.ALQ130 >=1) & (df.ALQ130<=15)]) # Counting Freq
i = sorted(i.items())  # Sorting the counted values
i
b= [x[0] for x in i] # setting Bin categories
a=c.Counter(df.ALQ130[(df.ALQ130 >=1) & (df.ALQ130<=15)]).values()
fig, ax=plt.subplots(figsize=(8,6), (1,2,1)) # Setting figure size
fig = plt.figure(constrained_layout=True)
plt.bar(b,a) # Plotting Bar chart
plt.title("Number of Drinks per month", loc='left')
plt.xlabel("Number of Drinks")
plt.ylabel("Number of People")
plt.show() 
plt.clf()


#Visualising the distribution of Race (method using dict to create cat)
race_label= ["Mexican American","Other Hispanic","Non-Hispanic White","Non-Hispanic Black","Other Race - Including Multi-Racial"]
race_n= dict(df.RIDRETH1.value_counts().sort_index())
race_cat = list(race_n.keys())
race_label
race_cat
race_freq = list(race_n.values())
fig, ax=plt.subplots(1,2,2)
plt.bar(race_cat, race_freq, tick_label=race_label)
plt.title("Distribution of Race")
plt.clf()
plt.show()

#Visualising weight distribution according to race
#Filtering out the dataset that has na
bwtdf = df[(~df.BMXWT.isna()) & (df.RIDRETH1<6)]
plt.clf()
fig, ax = plt.subplots()
scatter = ax.scatter(bwtdf.RIDAGEYR, bwtdf.BMXWT, c=bwtdf.RIDRETH1, s=bwtdf.RIDRETH1)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Race")
ax.add_artist(legend1)
# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
plt.show()

#Visualising data using boxplot to explore difference btween age intervals
plt.clf()
properties = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax1 = sns.boxplot(x='RIDRETH1', y='BMXWT', data = df, ax=axes[0])
ax1.set_xticks([0,1,2,3,4], labels= ['Mexican American','Other Hispanic','Non-Hispanic White','Non-Hispanic Black','Other Race - Including Multi-Racial'])
ax1.set_xlabels("Ethnic Groups of Sample Population")
ax1.set_ylabel("Body Weight (kg)")
ax1.set_title("Boxplot")


ax2 = sns.stripplot(x='RIDRETH1', y='BMXWT', data = df, ax=axes[1])
ax2.set_xticks([0,1,2,3,4], labels= ['Mexican American','Other Hispanic','Non-Hispanic White','Non-Hispanic Black','Other Race - Including Multi-Racial'])
ax2.set_xlabels("Ethnic Groups of Sample Population")
ax2.set_title("Stripplot")

#Mergine first and second plot for better visualisation
ax3 = sns.boxplot(x='RIDRETH1', y='BMXWT', data = df, **properties, ax=axes[2],showfliers = False) # fliters are hidden for second plot overlay
ax3 = sns.stripplot(x='RIDRETH1', y='BMXWT', data=df, ax=axes[2]) 
ax3.set_xticks([0,1,2,3,4], labels= ['Mexican American','Other Hispanic','Non-Hispanic White','Non-Hispanic Black','Other Race - Including Multi-Racial'])
ax3.set_xlabels("Ethnic Groups of Sample Population")
ax3.set_title("Boxplot w/ Stripplot overlay")

for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)



plt.show()


#Creating bins on age to explore difference between age groups
bins= np.linspace(min(bwtdf['RIDAGEYR']), max(bwtdf['RIDAGEYR']), 4)
group_names = ['Young', "Middle", "Older"]
g=bwtdf.groupby(np.digitize(bwtdf.RIDAGEYR, bins))
bwtdf['agebins']= pd.cut(bwtdf.RIDAGEYR,bins, labels= group_names, include_lowest= True)
bwtdf['agebins']
fig = plt.figure(constrained_layout=True)
plt.hist(bwtdf.agebins)
plt.show()

plt.clf()

#Processing Data to remove noresponse, missing data and refused data. 
dfcorr = df.filter(["SEQN", "drink", "SMQ020", "RIAGENDR", "RIDAGEYR","RIDRETH1", "DMDCITZN", "DMDEDUC2", "DMDMARTL", "DMDHHSIZ", "INDFMPIR", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BMXWT", "BMXHT", "BMXBMI", "BMXLEG", "BMXARML", "BMXWAIST", "HIQ210"], axis=1)
dfcorr.DMDCITZN[dfcorr.DMDCITZN>2] = np.nan
dfcorr.DMDEDUC2[dfcorr.DMDEDUC2>7] = np.nan
dfcorr.DMDMARTL[dfcorr.DMDMARTL>6] = np.nan
dfcorr.DMDHHSIZ[dfcorr.DMDHHSIZ>7] = np.nan
dfcorr.HIQ210[dfcorr.HIQ210>2] = np.nan
dfcorr.dropna().corr()
dfcorr.corr()


#Continuous Dataset for Corr
ccorr= dfcorr.loc[:,('RIDAGEYR','INDFMPIR','BPXSY1','BPXDI1','BPXSY2','BPXDI2' ,'BMXWT' ,'BMXHT','BMXBMI')]
eth1 = dfcorr[(df.RIDRETH1 == 1)].loc[:,('RIDAGEYR','INDFMPIR','BPXSY1','BPXDI1','BPXSY2','BPXDI2' ,'BMXWT' ,'BMXHT','BMXBMI')]
eth2 = dfcorr[(df.RIDRETH1 == 2)].loc[:,('RIDAGEYR','INDFMPIR','BPXSY1','BPXDI1','BPXSY2','BPXDI2' ,'BMXWT' ,'BMXHT','BMXBMI')]
eth3 = dfcorr[(df.RIDRETH1 == 3)].loc[:,('RIDAGEYR','INDFMPIR','BPXSY1','BPXDI1','BPXSY2','BPXDI2' ,'BMXWT' ,'BMXHT','BMXBMI')]
eth4 = dfcorr[(df.RIDRETH1 == 4)].loc[:,('RIDAGEYR','INDFMPIR','BPXSY1','BPXDI1','BPXSY2','BPXDI2' ,'BMXWT' ,'BMXHT','BMXBMI')]
eth5 = dfcorr[(df.RIDRETH1 == 5)].loc[:,('RIDAGEYR','INDFMPIR','BPXSY1','BPXDI1','BPXSY2','BPXDI2' ,'BMXWT' ,'BMXHT','BMXBMI')]

#Pairplots for each ethinicity
plt.clf()
plt.show()
sns.pairplot(eth1, dropna= True)
plt.title("Mexican American")
plt.show()

plt.clf()
plt.show()
sns.pairplot(eth2, dropna= True)
plt.title("Other Hispanic")
plt.show()

plt.clf()
plt.show()
sns.pairplot(eth3, dropna= True)
plt.title("None Hispanic White")
plt.show()

plt.clf()
plt.show()
sns.pairplot(eth4, dropna= True)
plt.title("None Hispanic Black")
plt.show()

plt.clf()
plt.show()
sns.pairplot(eth5, dropna= True)
plt.title("Others")
plt.show()

#Pairplots with all continuous data
plt.clf()
sns.pairplot(ccorr, dropna= True)
plt.show()

# Plotting of Corr Heatmap
mask = np.zeros_like(dfcorr.corr())
mask
mask[np.triu_indices_from(mask)]= True
fig = plt.figure(constrained_layout=True)

with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=(7,5))
        ax = sns.heatmap(dfcorr.corr(), mask=mask, vmax =.3, square = True, cmap= sns.diverging_palette(10,220, as_cmap=True))
        plt.title("Correlation Matrix Heatmap")
        plt.show()

#Uni- and Bi- varate Analysis
def categorical_summarized(df, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    df: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    plt.clf()
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = df[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=df, palette=palette)
    plt.show()

#Univarate
c_palette= ['tab:blue', 'tab:orange']
categorical_summarized(df, y='RIAGENDR', palette= c_palette)


#Bivarate
c_palette= ['tab:blue', 'tab:orange']
categorical_summarized(df, y='RIAGENDR', hue= 'ALQ101' , palette= c_palette)


# Quantitative Analysis

def quantitative_summarized(df, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data
    Arguments
    =========
    df: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed
    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    plt.clf()
    series = df[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=df, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=df,
                      palette=palette, ax=ax)

    plt.show()


#Boxplot comparision between gender, ethnic for BMI
quantitative_summarized(df, x='RIDRETH1', y='BMXBMI', hue='RIAGENDR')

