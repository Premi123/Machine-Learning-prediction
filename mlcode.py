#!/usr/bin/env python
# coding: utf-8

# In[100]:


###importing libraries
import numpy as np ### statistical calculation
import pandas as pd ### pandas for pre processing
import matplotlib.pyplot as plt ### data Visualization
import seaborn as sns ### adding more style on visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[101]:


###loading the dataset
dataset=pd.read_csv("mental_health.csv")
dataset.head(10) ### dispaying first 10 rows in our dataset


# In[102]:


###listing out the missing values
dataset.isnull().sum()


# In[103]:


###using mode value to fill the missing values in the work_interfere
dataset["work_interfere"].fillna('Sometimes',inplace = True)


# In[104]:


###counting the values
dataset.self_employed.value_counts()


# In[105]:


###using mode value to fill the missing values in the self_employed
dataset['self_employed'].fillna('No', inplace=True)


# In[106]:


###listing out to check if the null values are filled for the required columns
dataset.isnull().sum()


# In[107]:


###dropping the columns that are not necessary for predicting our target value
dataset = dataset.drop(['comments'], axis= 1)
dataset = dataset.drop(['state'], axis= 1)
dataset = dataset.drop(['Timestamp'], axis= 1)
dataset = dataset.drop(['Country'], axis= 1)


# In[108]:


print(dataset.isnull().sum().max()) ##### to check if there are any null values.


# In[109]:


dataset.head(5) ### disploying first 5 rows for checking droped columns


# In[110]:


###Finding unique values in Gender column
dataset['Gender'].unique()


# In[111]:


###replacing similar entities with one attribute
dataset['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

dataset['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

dataset["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace = True)


# In[112]:


###checking the output of above code 
dataset['Gender'].unique()


# In[113]:


dataset.head()


# In[114]:


###getting information about our columns and datatype
dataset.info()


# In[115]:


###importing label encoder for replacing the categories into numericals
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset['self_employed']= label_encoder.fit_transform(dataset['self_employed']) ###label encoding for self_employed
  
dataset['self_employed'].unique()
  


# In[116]:


###finding unique values for family_history
dataset.family_history.unique()


# In[117]:


###label encoding for family_history
label_encoder = preprocessing.LabelEncoder()
dataset['family_history']= label_encoder.fit_transform(dataset['family_history'])
  
dataset['family_history'].unique()


# In[118]:


###finding unique values for treatment
dataset.treatment.unique()


# In[119]:


###label encoding for treatment
label_encoder = preprocessing.LabelEncoder()
dataset['treatment']= label_encoder.fit_transform(dataset['treatment'])
  
dataset['treatment'].unique()


# In[120]:


###finding unique values for work_interfere
dataset.work_interfere.unique()


# In[121]:


###label encoding for work_interfere
dataset['work_interfere']= label_encoder.fit_transform(dataset['work_interfere'])
  
dataset['work_interfere'].unique()


# In[122]:


###finding unique values for no_employees
dataset.no_employees.unique()


# In[123]:


###label encoding for no_employees
dataset['no_employees']= label_encoder.fit_transform(dataset['no_employees'])
  
dataset['no_employees'].unique()


# In[124]:


###finding unique values for remote_work
dataset.remote_work.unique()


# In[125]:


###label encoding for remote_work
dataset['remote_work']= label_encoder.fit_transform(dataset['remote_work'])
  
dataset['remote_work'].unique()


# In[126]:


dataset.head() ## displaying dataset for checking  the output oflabel encoding 


# In[127]:


###finding unique values for tech_company
dataset.tech_company.unique()


# In[128]:


###label encoding for tech_company
dataset['tech_company']= label_encoder.fit_transform(dataset['tech_company'])
  
dataset['tech_company'].unique()


# In[129]:


###finding unique values for benefits
dataset.benefits.unique()


# In[130]:


###label encoding for benefits
dataset['benefits']= label_encoder.fit_transform(dataset['benefits'])
  
dataset['benefits'].unique()


# In[131]:


###finding unique values for benefits
dataset.anonymity.unique()


# In[132]:


###label encoding for anonymity
dataset['anonymity']= label_encoder.fit_transform(dataset['anonymity'])
  
dataset['anonymity'].unique()


# In[133]:


###finding unique values for leave
dataset.leave.unique()


# In[134]:


###label encoding for leave
dataset['leave']= label_encoder.fit_transform(dataset['leave'])
  
dataset['leave'].unique()


# In[135]:


###finding unique values for mental_health_consequence
dataset.mental_health_consequence.unique()


# In[136]:


###label encoding for mental_health_consequence
dataset['mental_health_consequence']= label_encoder.fit_transform(dataset['mental_health_consequence'])
  
dataset['mental_health_consequence'].unique()


# In[137]:


###checking the dataset with the labelled values
dataset.head()


# In[138]:


###finding unique values in phys_health_consequence
dataset.phys_health_consequence	.unique()


# In[139]:


###label encoding for phys_health_consequence
dataset['phys_health_consequence']= label_encoder.fit_transform(dataset['phys_health_consequence'])
  
dataset['phys_health_consequence'].unique()


# In[140]:


###finding unique values for supervisor
dataset.supervisor.unique()


# In[141]:


###label encoding for supervisor
dataset['supervisor']= label_encoder.fit_transform(dataset['supervisor'])
  
dataset['supervisor'].unique()


# In[142]:


###finding unique values for coworkers
dataset.coworkers.unique()


# In[143]:


###label encoding for coworkers
dataset['coworkers']= label_encoder.fit_transform(dataset['coworkers'])
  
dataset['coworkers'].unique()


# In[144]:


###finding unique values for mental_health_interview
dataset.mental_health_interview.unique()


# In[145]:


###label encoding for mental_health_interview
dataset['mental_health_interview']= label_encoder.fit_transform(dataset['mental_health_interview'])
  
dataset['mental_health_interview'].unique()


# In[146]:


###finding unique values for phys_health_interview
dataset.phys_health_interview.unique()


# In[147]:


###label encoding for phys_health_interview
dataset['phys_health_interview']= label_encoder.fit_transform(dataset['phys_health_interview'])
  
dataset['phys_health_interview'].unique()


# In[148]:


### finding unique value of mental_vs_physical
dataset.mental_vs_physical.unique()


# In[149]:


## label encoding for mental_vs_physical
dataset['mental_vs_physical']= label_encoder.fit_transform(dataset['mental_vs_physical'])
  
dataset['mental_vs_physical'].unique()


# In[150]:


## unique values of obs_consequence
dataset.obs_consequence.unique()


# In[151]:


#### label encoding for obs_consequence
dataset['obs_consequence']= label_encoder.fit_transform(dataset['obs_consequence'])
  
dataset['obs_consequence'].unique()


# In[152]:


## unique values for gender
dataset.Gender.unique()


# In[153]:


## label encoding for gender
dataset['Gender']= label_encoder.fit_transform(dataset['Gender'])
  
dataset['Gender'].unique()


# In[154]:


### unique value for care_option

dataset.care_options.unique()


# In[155]:


### label encoding for care_option
dataset['care_options']= label_encoder.fit_transform(dataset['care_options'])
  
dataset['care_options'].unique()


# In[156]:


### label encoding for wellness program
dataset['wellness_program']= label_encoder.fit_transform(dataset['wellness_program'])
  
dataset['wellness_program'].unique()


# In[157]:


### label encoding for seek_help
dataset['seek_help']= label_encoder.fit_transform(dataset['seek_help'])
  
dataset['seek_help'].unique()


# In[158]:


### checking datasets
### changing categories to numericals
dataset.head()


# In[159]:


dataset.Age.unique() ### getting unique values of age


# In[160]:


## eliminating the unwanted data in age colum using replace methods 
dataset['Age'].replace([dataset['Age'][dataset['Age'] < 15]], np.nan, inplace = True)
dataset['Age'].replace([dataset['Age'][dataset['Age'] > 100]], np.nan, inplace = True)
## the age between 15 to 100 only display, others change into nan
dataset['Age'].unique()


# In[161]:


dataset.Age.isnull().sum() ### getting total number of null values in age


# In[162]:


data=dataset.dropna() ### dropping the null values


# In[163]:


data.head() ## displaying the dataset


# In[164]:


data.isnull().sum() ### checking if null values are still there


# In[165]:


data.Age.unique() ### there is no unusual data in the age column


# In[166]:


data.Age.min() ### getting minimum value in age for checking


# In[167]:


data.Age.max() #### getting maximum value in age for checking


# In[168]:


#### assigning the age values to 0,1,2
data.loc[data['Age']<= 30, 'Age']= 0
data.loc[(data['Age']> 30) & (data['Age'] <= 50), 'Age'] = 1
data.loc[ data['Age']> 50, 'Age']= 2


# In[169]:


# getting unique value for age 
data.Age.unique()


# In[170]:


data.head()


# In[171]:


### 'treatment' is our target variable, so dropping it from X 
X=data.drop(['treatment'], axis = 1)


# In[172]:


y=data['treatment']


# In[173]:


### importing libraries for training, testing and splitting of the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3)


# # feature selection

# In[175]:


### coworkers and supervisor are correlated by 58%, thus, dropping either coworkers or supervisor before fitting  the model
### dropping 'supervisor' column


# In[182]:


### plotting the heatmap determing the correlation
plt.figure(figsize=(17,17))
sns.heatmap(data.corr(),annot=True,cmap ='twilight_shifted')


# In[870]:


### finding constant varaiables to drop as the dependency factor on our target variable is very less
for col in ["Age","Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work","tech_company", "benefits", "care_options","wellness_program", "seek_help", "anonymity", "leave", "mental_health_consequence","phys_health_consequence", "coworkers", "supervisor", "mental_health_interview", "phys_health_interview", "mental_vs_physical", "obs_consequence"]:
    sns.catplot(x=col, y=y, data=data, kind='point', aspect=2)
    plt.ylim(0, 0.7)


# In[ ]:


xx=data


# # FEATURE SELECTION

# In[950]:


#### importing libraries for train and test data
from sklearn.feature_selection import mutual_info_classif
mut_info=mutual_info_classif(X_train, y_train, random_state=42)
mut_info


# In[951]:


mut_info = pd.Series(mut_info)
mut_info.index = X_train.columns
mut_info.sort_values(ascending = False)


# In[952]:


mut_info.sort_values(ascending = False).plot.bar(figsize=(20,8))


# In[117]:


data.columns


# In[118]:


### dropping columns in xx as they have low dependency on the target variable and treatment and supervisor are not applicable
xx = data.drop(['treatment','supervisor','phys_health_consequence','tech_company','no_employees','Gender'], axis = 1)
yy= data['treatment']


# In[119]:


xx.head(2)


# In[120]:


yy.head()


# In[121]:


### importing libraries for random forest algorithm and it's train and test data
from sklearn.model_selection import train_test_split
xx_train, xx_test, yy_train,yy_test = train_test_split(xx,yy,test_size = 0.3)


# In[122]:


### random forest model
### using all features for random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, plot_roc_curve


# In[123]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[124]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[125]:


rf=RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 8, verbose=2, random_state=42, n_jobs = -1)


# In[126]:


rf_random.fit(xx_train, yy_train)


# In[127]:


rf_random.best_score_


# In[128]:


rf_random.best_params_


# In[129]:


rf_random.best_estimator_


# In[130]:


r_forest = RandomForestClassifier(max_depth=90, max_features='sqrt', min_samples_leaf=4,
                       min_samples_split=10, n_estimators=400)
r_forest.fit(xx_train,yy_train)


# In[131]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred_rf = r_forest.predict(xx_test)
print(confusion_matrix(yy_test,y_pred_rf))
print(classification_report(yy_test,y_pred_rf))
print(accuracy_score(yy_test,y_pred_rf))


# In[ ]:


### SVM


# In[140]:


from sklearn import svm
clf = svm.SVC()
clf.fit(xx_train, yy_train)


# In[141]:


y_predsvc = clf.predict(xx_test)
print(confusion_matrix(yy_test,y_predsvc))
print(classification_report(yy_test,y_predsvc))


# In[142]:


print(accuracy_score(yy_test,y_predsvc))


# In[154]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,xx_test,yy_test,cv=10)
    knn_scores.append(score.mean())


# In[155]:


plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[167]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=14)
neigh.fit(xx_test, yy_test)


# In[168]:


y_predknn = neigh.predict(xx_test)
print(confusion_matrix(yy_test,y_predknn))
print(classification_report(yy_test,y_predknn))


# In[169]:


print(accuracy_score(yy_test,y_predknn))


# In[1058]:


#### plz load xg boost and decision tree ####


# In[146]:


from xgboost.sklearn import XGBClassifier
boost = XGBClassifier(random_state = 42)
boost.fit(xx_train, yy_train)


# In[150]:


import xgboost as xgb


# In[151]:


y_predxg = boost.predict(xx_test)
print(confusion_matrix(yy_test,y_predxg))
print(classification_report(yy_test,y_predxg))


# In[152]:


print(accuracy_score(yy_test,y_predxg))


# In[ ]:




