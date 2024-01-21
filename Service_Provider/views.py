
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,insurance_claim_status,detection_accuracy,detection_ratio


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":

            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')



def Find_Insurance_Claim_Prediction_Details_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Fraud Claim'
    print(kword)
    obj = insurance_claim_status.objects.all().filter(Q(PREDICTION=kword))
    obj1 = insurance_claim_status.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Real Claim'
    print(kword1)
    obj1 = insurance_claim_status.objects.all().filter(Q(PREDICTION=kword1))
    obj11 = insurance_claim_status.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()

    return render(request, 'SProvider/Find_Insurance_Claim_Prediction_Details_Ratio.html', {'objs': obj})

def View_Insurance_Claim_Prediction_Details(request):

    obj = insurance_claim_status.objects.all()
    return render(request, 'SProvider/View_Insurance_Claim_Prediction_Details.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = detection_accuracy.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Train_Test_View_Results_Details(request):
    detection_accuracy.objects.all().delete()
    data = pd.read_csv("Insurance_Claim_Datasets.csv", encoding='latin-1')

    def apply_results(results):
        if (results == 'Fraud'):
            return 0
        elif (results == 'Real'):
            return 1

    data['Results'] = data['Claim_Staus'].apply(apply_results)

    x = data['POLICY_NO'].apply(str)
    y = data['Results']

    # data.drop(['Type_of_Breach'],axis = 1, inplace = True)
    cv = CountVectorizer()

    print(x)
    print(y)

    cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
    # x = cv.fit_transform(data['POLICY_NO'].apply(lambda x: np.str_(x)))
    x = cv.fit_transform(x)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB

    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print("ACCURACY")
    print(naivebayes)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_nb))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_nb))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, dtcpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, dtcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, dtcpredict))
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

    print("SGD Classifier")
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
    sgd_clf.fit(X_train, y_train)
    sgdpredict = sgd_clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, sgdpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, sgdpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, sgdpredict))
    detection_accuracy.objects.create(names="SGD Classifier", ratio=accuracy_score(y_test, sgdpredict) * 100)


    labeled = 'labeled_data.csv'
    data.to_csv(labeled, index=False)
    data.to_markdown


    obj =detection_accuracy.objects.all()
    return render(request, 'SProvider/Train_Test_View_Results_Details.html', {'objs': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = insurance_claim_status.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Account_Code, font_style)
        ws.write(row_num, 1, my_row.DATE_OF_INTIMATION, font_style)
        ws.write(row_num, 2, my_row.DATE_OF_ACCIDENT, font_style)
        ws.write(row_num, 3, my_row.CLAIM_Real, font_style)
        ws.write(row_num, 4, my_row.AGE, font_style)
        ws.write(row_num, 5, my_row.TYPE, font_style)
        ws.write(row_num, 6, my_row.DRIVING_LICENSE_ISSUE, font_style)
        ws.write(row_num, 7, my_row.BODY_TYPE, font_style)
        ws.write(row_num, 8, my_row.MAKE, font_style)
        ws.write(row_num, 9, my_row.MODEL, font_style)
        ws.write(row_num, 10, my_row.YEAR, font_style)
        ws.write(row_num, 11, my_row.CHASIS_Real, font_style)
        ws.write(row_num, 12, my_row.REG, font_style)
        ws.write(row_num, 13, my_row.SUM_INSURED, font_style)
        ws.write(row_num, 14, my_row.POLICY_NO, font_style)
        ws.write(row_num, 15, my_row.POLICY_START, font_style)
        ws.write(row_num, 16, my_row.POLICY_END, font_style)
        ws.write(row_num, 17, my_row.INTIMATED_AMOUNT, font_style)
        ws.write(row_num, 18, my_row.INTIMATED_SF, font_style)
        ws.write(row_num, 19, my_row.EXECUTIVE, font_style)
        ws.write(row_num, 20, my_row.PRODUCT, font_style)
        ws.write(row_num, 21, my_row.POLICYTYPE, font_style)
        ws.write(row_num, 22, my_row.NATIONALITY, font_style)
        ws.write(row_num, 23, my_row.PREDICTION, font_style)

    wb.save(response)
    return response

















