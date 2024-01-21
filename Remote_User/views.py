from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import datetime
import re
import string

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier


# Create your views here.
from Remote_User.models import ClientRegister_Model,insurance_claim_status

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    val=''
    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": val})


def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Insurance_Claim_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Account_Code=request.POST.get('Account_Code')
            DATE_OF_INTIMATION=request.POST.get('DATE_OF_INTIMATION')
            DATE_OF_ACCIDENT=request.POST.get('DATE_OF_ACCIDENT')
            CLAIM_Real=request.POST.get('CLAIM_Real')
            AGE=request.POST.get('AGE')
            TYPE=request.POST.get('TYPE')
            DRIVING_LICENSE_ISSUE=request.POST.get('DRIVING_LICENSE_ISSUE')
            BODY_TYPE=request.POST.get('BODY_TYPE')
            MAKE=request.POST.get('MAKE')
            MODEL=request.POST.get('MODEL')
            YEAR=request.POST.get('YEAR')
            CHASIS_Real=request.POST.get('CHASIS_Real')
            REG = request.POST.get('REG')
            SUM_INSURED=request.POST.get('SUM_INSURED')
            POLICY_NO=request.POST.get('POLICY_NO')
            POLICY_START=request.POST.get('POLICY_START')
            POLICY_END=request.POST.get('POLICY_END')
            INTIMATED_AMOUNT=request.POST.get('INTIMATED_AMOUNT')
            INTIMATED_SF=request.POST.get('INTIMATED_SF')
            EXECUTIVE=request.POST.get('EXECUTIVE')
            PRODUCT=request.POST.get('PRODUCT')
            POLICYTYPE=request.POST.get('POLICYTYPE')
            NATIONALITY=request.POST.get('NATIONALITY')


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

            labeled = 'labeled_data.csv'
            data.to_csv(labeled, index=False)
            data.to_markdown

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
            print(naivebayes)
            print(confusion_matrix(y_test, predict_nb))
            print(classification_report(y_test, predict_nb))
            models.append(('naive_bayes', NB))

            # SVM Model
            print("SVM")
            from sklearn import svm

            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            print(svm_acc)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, predict_svm))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, predict_svm))
            models.append(('svm', lin_clf))

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
            models.append(('logistic', reg))

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
            models.append(('SGDClassifier', sgd_clf))


            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)


            POLICY_NO2 = [POLICY_NO]
            vector1 = cv.transform(POLICY_NO2).toarray()
            predict_text = classifier.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")

            prediction = int(pred1)

            if prediction == 0:
                val = 'Fraud Claim'
            elif prediction == 1:
                val = 'Real Claim'

            print(prediction)
            print(val)

            insurance_claim_status.objects.create(
            Account_Code=Account_Code,
            DATE_OF_INTIMATION=DATE_OF_INTIMATION,
            DATE_OF_ACCIDENT=DATE_OF_ACCIDENT,
            CLAIM_Real=CLAIM_Real,
            AGE=AGE,
            TYPE=TYPE,
            DRIVING_LICENSE_ISSUE=DRIVING_LICENSE_ISSUE,
            BODY_TYPE=BODY_TYPE,
            MAKE=MAKE,
            MODEL=MODEL,
            YEAR=YEAR,
            CHASIS_Real=CHASIS_Real,
            REG=REG,
            SUM_INSURED=SUM_INSURED,
            POLICY_NO=POLICY_NO,
            POLICY_START=POLICY_START,
            POLICY_END=POLICY_END,
            INTIMATED_AMOUNT=INTIMATED_AMOUNT,
            INTIMATED_SF=INTIMATED_SF,
            EXECUTIVE=EXECUTIVE,
            PRODUCT=PRODUCT,
            POLICYTYPE=POLICYTYPE,
            NATIONALITY=NATIONALITY,
            PREDICTION=val)

        return render(request, 'RUser/Predict_Insurance_Claim_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Insurance_Claim_Type.html')





