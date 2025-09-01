import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)




lift = pd.read_csv('new_male_test.csv', encoding = 'cp949')
lifting = lift[['bweight', 'age','event','nation', 'con','snatch','bornyear', 'jerk', 'total', 'doping','dateyear']]
lifting['tw'] = lifting['total'] / lifting['bweight']
lifting.dropna(axis = 0, inplace = True )
lifting.drop(lifting[lifting['total']== 0].index, inplace = True)

def assign_olympic_category(weight):
    if weight <= 61:
        return "≤61kg"
    elif weight <= 73:
        return "≤73kg"
    elif weight <= 89:
        return "≤89kg"
    elif weight <= 102:
        return "≤102kg"
    else:
        return ">102kg"


lifting["olympic_category"] = lifting["bweight"].apply(assign_olympic_category)

ndp = lifting[lifting['doping'] == 0]
dp = lifting[lifting['doping'] == 1]

lifting.data = lifting[['bweight', 'age','total','olympic_category']]
lifting.target = lifting['doping']

lifting_data = lifting[['age','total', 'olympic_category']].dropna()


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# 3. age와 bweight 기준 이상치 제거
lifting_data = remove_outliers_iqr(lifting_data, 'age')

encoder = LabelEncoder()
lifting_data['olympic_category'] = encoder.fit_transform(lifting_data['olympic_category'])

# 6. 수치형 변수 정규화 (total과 age)
scaler = StandardScaler()
lifting_data[[ 'age','total']] = scaler.fit_transform(lifting_data[[ 'age', 'total']])



def visualize_silhouette(cluster_lists, X_features, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    if n_cols == 1:
        axs = [axs]

    for ind, n_cluster in enumerate(cluster_lists):
        clusterer = KMeans(n_clusters=n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title(f'Number of Clusters: {n_cluster}\nSilhouette Score: {round(sil_avg, 3)}')
        axs[ind].set_xlabel("Silhouette Coefficient Values")
        axs[ind].set_ylabel("Cluster Label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper),
                                   0, ith_cluster_sil_values,
                                   facecolor=color, edgecolor=color, alpha=0.7)

            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

    plt.tight_layout()
#     plt.savefig('kmeans.png', dpi=300, bbox_inches='tight')
    plt.show()



from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



visualize_silhouette([ 2,3,4,5 ], lifting_data)





from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# 모델 선언
model = KMeans(random_state=0)

# 시각화 객체 설정: inertia 기준, K=2~10 테스트
visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)

# 학습 및 시각화
visualizer.fit(lifting_data)  # PCA 결과 사용
visualizer.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

X_features = lifting_data.values if hasattr(lifting_data, 'values') else lifting_data

fig = plt.figure(figsize=(20, 14))

# ----- (A) Silhouette Plots -----
cluster_range = [2, 3, 4, 5]
for idx, n_clusters in enumerate(cluster_range):
    ax = plt.subplot2grid((3, 4), (0, idx), rowspan=1)
    clusterer = KMeans(n_clusters=n_clusters, max_iter=500, random_state=0)
    cluster_labels = clusterer.fit_predict(X_features)
    silhouette_avg = silhouette_score(X_features, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Blues(float(i + 1) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.9)
        ax.text(-0.08, y_lower + 0.5 * size_cluster_i, str(i), fontsize=8)
        y_lower = y_upper + 10

    ax.set_title(f'Number of clusters = {n_clusters}\nSilhouette = {silhouette_avg:.3f}', fontsize=10, pad=8)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X_features) + (n_clusters + 1) * 10])
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Silhouette Coefficient", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Cluster Label", fontsize=9)

# (A) 라벨
fig.text(0.5, 0.97, "(A)", ha='center', va='center', fontsize=18, weight='bold')

# ----- (B) Distortion Plot -----
ax_dist = plt.subplot2grid((3, 4), (1, 1), colspan=2, rowspan=2)
distortions = []
kvals = range(2, 10)
for k in kvals:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X_features)
    distortions.append(model.inertia_)

elbow_k = 4
elbow_score = distortions[elbow_k - 2]

ax_dist.plot(kvals, distortions, marker='o', linestyle='-', color='royalblue')
ax_dist.axvline(x=elbow_k, linestyle='--', color='black')
ax_dist.set_title("Distortion Score Elbow for KMeans Clustering", fontsize=13)
ax_dist.set_xlabel("Number of Clusters", fontsize=12)
ax_dist.set_ylabel("Distortion Score", fontsize=12)
ax_dist.legend([f"elbow at k = {elbow_k}, score = {elbow_score:.3f}"], loc='upper right', fontsize=11)

# (B) 라벨
fig.text(0.5, 0.58, "(B)", ha='center', va='center', fontsize=18, weight='bold')

plt.subplots_adjust(left=0.06, right=0.98, hspace=1.4, wspace=0.3, top=0.94, bottom=0.08)

# 저장 원할 시 주석 해제
# plt.savefig("final_combined_silhouette_distortion.png", dpi=300, bbox_inches='tight')

plt.show()


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# 데이터 준비
X_features = lifting_data.values if hasattr(lifting_data, 'values') else lifting_data

# 실루엣 플롯 4개를 하나의 그림으로 저장
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
cluster_range = [2, 3, 4, 5]

for ax, n_clusters in zip(axes, cluster_range):
    clusterer = KMeans(n_clusters=n_clusters, max_iter=500, random_state=0)
    cluster_labels = clusterer.fit_predict(X_features)
    silhouette_avg = silhouette_score(X_features, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Blues(float(i + 1) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.9)
        # 클러스터 레이블을 1부터 시작해서 표시
        ax.text(-0.08, y_lower + 0.5 * size_cluster_i, str(i + 1), fontsize=8)
        y_lower = y_upper + 10

    ax.set_title(f'Number of Clusters: {n_clusters}\nSilhouette Score: {silhouette_avg:.3f}', fontsize=11)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X_features) + (n_clusters + 1) * 10])
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Silhouette Coefficient")
    if n_clusters == 2:
        ax.set_ylabel("Cluster Label")

plt.tight_layout()
# 저장하려면 아래 주석 해제
plt.savefig("silhouette_all.png", dpi=300)
plt.show()


# In[52]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

distortions = []
kvals = range(2, 10)
for k in kvals:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(X_features)
    distortions.append(model.inertia_)

elbow_k = 4

plt.figure(figsize=(6, 5))
plt.plot(kvals, distortions, marker='o', linestyle='-', color='dodgerblue')
plt.axvline(x=elbow_k, linestyle='--', color='black', label=f"Optimal Clusters")  # 선에 라벨 직접 지정
# plt.title("Distortion Score Elbow for KMeans Clustering")  # 제목 생략 시 주석 처리
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion Score")
plt.legend(loc='upper right')  # 위에서 지정된 label이 여기에 표시됨
plt.tight_layout()
plt.savefig("distortion_elbow.png", dpi=300)
plt.show()



grouped = clut_3.groupby('cluster')

# 실제 도핑률 계산
observed_rates = grouped['doping'].mean().values
observed_stat = np.var(observed_rates)

# 퍼뮤테이션 테스트
n_permutations = 10000
permuted_stats = []
doping_array = clut_3['doping'].values
cluster_sizes = grouped.size().values

for _ in range(n_permutations):
    shuffled = np.random.permutation(doping_array)
    permuted_means = []
    start = 0
    for size in cluster_sizes:
        end = start + size
        permuted_means.append(np.mean(shuffled[start:end]))
        start = end
    stat = np.var(permuted_means)
    permuted_stats.append(stat)

# p-value 계산
p_value = np.mean([s >= observed_stat for s in permuted_stats])

# 결과 출력
print(f"Observed variance in doping rates across clusters: {observed_stat:.5f}")
print(f"P-value from permutation test: {p_value:.5f}")


# In[24]:


ndp1_2 = group1_2[group1_2['doping'] ==0 ]
dp1_2 = group1_2[group1_2['doping'] ==1 ]
ndp2_2 = group2_2[group2_2['doping'] ==0 ]
dp2_2 = group2_2[group2_2['doping'] ==1 ]
ndp3_2 = group3_2[group3_2['doping'] ==0 ]
dp3_2 = group3_2[group3_2['doping'] ==1 ]
ndp4_2 = group4_2[group4_2['doping'] ==0 ]
dp4_2 = group4_2[group4_2['doping'] ==1 ]



ndop1 = pd.read_csv('nondop_1.csv', encoding = 'cp949')
ndop2 = pd.read_csv('nondop_2.csv', encoding = 'cp949')
ndop3 = pd.read_csv('nondop_3.csv', encoding = 'cp949')
ndop4 = pd.read_csv('nondop_4.csv', encoding = 'cp949')

dop1 = pd.read_csv('dop_1.csv', encoding = 'cp949')
dop2 = pd.read_csv('dop_2.csv', encoding = 'cp949')
dop3 = pd.read_csv('dop_3.csv', encoding = 'cp949')
dop3 = pd.read_csv('dop_4.csv', encoding = 'cp949')



y_target = ndop1['tw']
X_features = ndop1[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=0)


# In[7]:


rf = KNeighborsRegressor()
param_dist = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],        # Number of neighbors to use
    'weights': ['uniform', 'distance'],            # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute neighbors
    'leaf_size': [10, 20, 30, 40, 50],              # Leaf size for BallTree or KDTree
    'p': [1, 2]                                     # Power parameter for Minkowski metric (1 = manhattan, 2 = euclidean)
}


# In[8]:


def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# 사이킷런의 mean_square_error() 를 이용하여 RMSE 계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))

# MAE, RMSE, RMSLE 를 모두 계산 
def evaluate_regr(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    r2_val = r2_score(y,pred)
    print('RMSLE: {0:.3f}, RMSE: {1:.3F}, R2: {2:.3F}'.format(rmsle_val, rmse_val, r2_val))


# In[9]:


random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,  # number of parameter settings sampled
    cv=5,       # 5-fold cross-validation
    scoring='neg_mean_squared_error',  # regression metric
    verbose=1,
    random_state=0,
    n_jobs=-1
)


# In[10]:


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


# In[11]:


y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", random_search.best_params_)
print("Test MSE:", mse)
print("Test MAE:", mae)
print('R2:' , r2)


# In[12]:


result_n_1 = pd.DataFrame(y_test.values, columns=['real_count'])
result_n_1['predicted_count']= np.round(y_pred)
result_n_1['diff'] = np.abs(result_n_1['real_count'] - result_n_1['predicted_count'])


# In[13]:


dy_pred = best_model.predict(dop1[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']])
result_d_1 = pd.DataFrame(dop1['tw'].values, columns=['real_count'])
result_d_1['predicted_count']= np.round(dy_pred)
result_d_1['diff'] = np.abs(result_d_1['real_count'] - result_d_1['predicted_count'])


# In[14]:


from scipy.stats import mannwhitneyu
mannwhitneyu(result_n_1['diff'] , result_d_1['diff'])


# In[15]:


from scipy.stats import mannwhitneyu, norm
import numpy as np

# 데이터
x = result_n_1['diff']
y = result_d_1['diff']

# 1. Mann–Whitney U test
u_stat, p_val = mannwhitneyu(result_n_1['diff'] , result_d_1['diff'], alternative='two-sided')

# 2. Z-score 계산
n1, n2 = len(x), len(y)
mean_u = n1 * n2 / 2
std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z = (u_stat - mean_u) / std_u

# 3. Effect size r 계산
r = abs(z) / np.sqrt(n1 + n2)

# 4. 출력
print(f"Mann–Whitney U: {u_stat:.3f}, p-value: {p_val:.4f}")
print(f"Effect size (r): {r:.3f}")


# In[16]:


import numpy as np

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)
    delta = (greater - less) / (nx * ny)
    return delta

# 사용 예시:
delta = cliffs_delta(result_n_1['diff'], result_d_1['diff'])
print(f"Cliff's Delta: {delta:.4f}")


# In[17]:


#group2


# In[18]:


y_target = ndop2['tw']
X_features = ndop2[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=100)


# In[19]:


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


# In[20]:


y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", random_search.best_params_)
print("Test MSE:", mse)
print("Test MAE:", mae)
print('R2:' , r2)


# In[21]:


result_n_2 = pd.DataFrame(y_test.values, columns=['real_count'])
result_n_2['predicted_count']= np.round(y_pred)
result_n_2['diff'] = np.abs(result_n_2['real_count'] - result_n_2['predicted_count'])


# In[22]:


dy_pred = best_model.predict(dop2[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']])
result_d_2 = pd.DataFrame(dop2['tw'].values, columns=['real_count'])
result_d_2['predicted_count']= np.round(dy_pred)
result_d_2['diff'] = np.abs(result_d_2['real_count'] - result_d_2['predicted_count'])


# In[23]:


x = result_n_2['diff']
y = result_d_2['diff']

# 1. Mann–Whitney U test
u_stat, p_val = mannwhitneyu(result_n_1['diff'] , result_d_1['diff'], alternative='two-sided')

# 2. Z-score 계산
n1, n2 = len(x), len(y)
mean_u = n1 * n2 / 2
std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z = (u_stat - mean_u) / std_u

# 3. Effect size r 계산
r = abs(z) / np.sqrt(n1 + n2)

# 4. 출력
print(f"Mann–Whitney U: {u_stat:.3f}, p-value: {p_val:.4f}")
print(f"Effect size (r): {r:.3f}")


# In[24]:


import numpy as np

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)
    delta = (greater - less) / (nx * ny)
    return delta

# 사용 예시:
delta = cliffs_delta(result_n_2['diff'], result_d_2['diff'])
print(f"Cliff's Delta: {delta:.4f}")


# In[25]:


#group3


# In[26]:


y_target = ndop3['tw']
X_features = ndop3[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=100)


# In[27]:


random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


# In[28]:


y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", random_search.best_params_)
print("Test MSE:", mse)
print("Test MAE:", mae)
print('R2:' , r2)


# In[29]:


result_n_3 = pd.DataFrame(y_test.values, columns=['real_count'])
result_n_3['predicted_count']= np.round(y_pred)
result_n_3['diff'] = np.abs(result_n_3['real_count'] - result_n_3['predicted_count'])


# In[30]:


dy_pred = best_model.predict(dop3[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']])
result_d_3 = pd.DataFrame(dop3['tw'].values, columns=['real_count'])
result_d_3['predicted_count']= np.round(dy_pred)
result_d_3['diff'] = np.abs(result_d_3['real_count'] - result_d_3['predicted_count'])


# In[31]:


x = result_n_3['diff']
y = result_d_3['diff']

# 1. Mann–Whitney U test
u_stat, p_val = mannwhitneyu(result_n_3['diff'] , result_d_3['diff'], alternative='two-sided')

# 2. Z-score 계산
n1, n2 = len(x), len(y)
mean_u = n1 * n2 / 2
std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z = (u_stat - mean_u) / std_u

# 3. Effect size r 계산
r = abs(z) / np.sqrt(n1 + n2)

# 4. 출력
print(f"Mann–Whitney U: {u_stat:.3f}, p-value: {p_val:.4f}")
print(f"Effect size (r): {r:.3f}")


# In[32]:


delta = cliffs_delta(result_n_3['diff'], result_d_3['diff'])
print(f"Cliff's Delta: {delta:.4f}")


# In[33]:


# group4


# In[34]:


y_target = ndop4['tw']
X_features = ndop4[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=100)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


# In[35]:


y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", random_search.best_params_)
print("Test MSE:", mse)
print("Test MAE:", mae)
print('R2:' , r2)


# In[36]:


result_n_4 = pd.DataFrame(y_test.values, columns=['real_count'])
result_n_4['predicted_count']= np.round(y_pred)
result_n_4['diff'] = np.abs(result_n_4['real_count'] - result_n_4['predicted_count'])

dy_pred = best_model.predict(dop3[['bweight', 'age','olympic_category', 'event_type_encoded','nation_encoded']])
result_d_4 = pd.DataFrame(dop3['tw'].values, columns=['real_count'])
result_d_4['predicted_count']= np.round(dy_pred)
result_d_4['diff'] = np.abs(result_d_4['real_count'] - result_d_4['predicted_count'])


# In[37]:


x = result_n_4['diff']
y = result_d_4['diff']

# 1. Mann–Whitney U test
u_stat, p_val = mannwhitneyu(result_n_4['diff'] , result_d_4['diff'], alternative='two-sided')

# 2. Z-score 계산
n1, n2 = len(x), len(y)
mean_u = n1 * n2 / 2
std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z = (u_stat - mean_u) / std_u

# 3. Effect size r 계산
r = abs(z) / np.sqrt(n1 + n2)

# 4. 출력
print(f"Mann–Whitney U: {u_stat:.3f}, p-value: {p_val:.4f}")
print(f"Effect size (r): {r:.3f}")


# In[38]:


delta = cliffs_delta(result_n_4['diff'], result_d_4['diff'])
print(f"Cliff's Delta: {delta:.4f}")


# In[39]:


import seaborn as sns



# 클러스터별 데이터 통합
dfs = []
for i in range(1, 5):
    df_n = globals()[f"result_n_{i}"].copy()
    df_d = globals()[f"result_d_{i}"].copy()

    df_n["Group"] = "Negative"
    df_d["Group"] = "Positive"

    df_n["Cluster"] = f"Cluster {i}"
    df_d["Cluster"] = f"Cluster {i}"

    dfs.append(pd.concat([df_n, df_d], axis=0))

# 전체 병합
plot_df = pd.concat(dfs, axis=0)

# 스타일 설정
sns.set(style="whitegrid", font_scale=1.2)

# 박스플롯 + 점그래프
g = sns.catplot(
    data=plot_df,
    x="Group", y="diff", col="Cluster",
    kind="box", notch=True,
    palette=["#3498db", "#e67e22"],
    height=5, aspect=0.8,
    linewidth=2, fliersize=0
)

# 점 데이터 (stripplot) 추가 및 스타일 조정
for i, ax in enumerate(g.axes.flat):
    cluster = f"Cluster {i+1}"
    sub_df = plot_df[plot_df["Cluster"] == cluster]
    sns.stripplot(
        data=sub_df,
        x="Group", y="diff",
        ax=ax,
        palette=["#3498db", "#e67e22"],
        jitter=0.25, size=4,
        alpha=0.25, dodge=True
    )
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Doping Status", fontsize=13, weight='bold')  # ← 여기 수정
    ax.set_ylabel("Prediction Error", fontsize=13, weight='bold')
    ax.set_title(cluster, fontsize=13, weight='bold')

plt.tight_layout()
# 저장 시:
# plt.savefig("cluster_diff_boxplot_final.png", dpi=300, bbox_inches='tight')
plt.show()


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# 클러스터별 데이터프레임 리스트 구성
cluster_auc_dict = {}
plt.figure(figsize=(8, 6))

for i in range(1, 5):
    # 클러스터별 도핑/비도핑 데이터
    df_n = globals()[f"result_n_{i}"].copy()
    df_d = globals()[f"result_d_{i}"].copy()

    df_n["doping"] = 0
    df_d["doping"] = 1

    df_cluster = pd.concat([df_n, df_d], axis=0)
    df_cluster = df_cluster.dropna(subset=["diff", "doping"])

    # AUC 계산
    y_true = df_cluster["doping"]
    y_score = df_cluster["diff"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    cluster_auc_dict[f"Cluster {i}"] = auc
    plt.plot(fpr, tpr, label=f"Cluster {i} (AUC = {auc:.3f})")

# 기준선
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title("Cluster-wise ROC Curve (Prediction Error = actual tw - predicted tw)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('AUC_curve.png', dpi= 300)
plt.show()

# AUC 수치 출력
for name, auc in cluster_auc_dict.items():
    print(f"{name}: AUC = {auc:.4f}")






