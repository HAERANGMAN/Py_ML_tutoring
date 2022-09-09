# `분류`

# 1. 로지스틱 회귀 
# 2. 신경망 모형
# 3. 의사결정트리(분류) : iris_data()
# 4. KNN
# 5. SVM
# 6. 앙상블
#     1. 보팅
#     2. 베깅
#     3. 부스팅
#         1. ADA
#         2. Gradient(GBM)
#         3. XGB
#         4. lightGBM


#붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리
iris_data = load_iris()
(iris_data.data,iris_data.target ,test_size=0.2 ,random_state=11)
X_train,X_test,y_train,y_test = train_test_split

# 의사결정트리(분류) : iris_data()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def DTC(X_train,y_train,X_test):
        #DecisionTree Classifier생성
        dt_clf = DecisionTreeClassifier(random_state=156)        

        #DecisionTreeClassifier 학습
        dt_clf.fit(X_train,y_train)
        
        #예측정확도
        accuracy_score(y_test,DecisionTreeClassifier(random_state=156).predict(X_test))
        
