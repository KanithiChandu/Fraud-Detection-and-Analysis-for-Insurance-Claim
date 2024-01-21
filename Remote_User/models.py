from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address = models.CharField(max_length=300)
    gender = models.CharField(max_length=30)


class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class insurance_claim_status(models.Model):

    Account_Code= models.CharField(max_length=300)
    DATE_OF_INTIMATION= models.CharField(max_length=300)
    DATE_OF_ACCIDENT= models.CharField(max_length=300)
    CLAIM_Real= models.CharField(max_length=300)
    AGE= models.CharField(max_length=300)
    TYPE= models.CharField(max_length=300)
    DRIVING_LICENSE_ISSUE= models.CharField(max_length=300)
    BODY_TYPE= models.CharField(max_length=300)
    MAKE= models.CharField(max_length=300)
    MODEL= models.CharField(max_length=300)
    YEAR= models.CharField(max_length=300)
    CHASIS_Real= models.CharField(max_length=300)
    REG = models.CharField(max_length=300)
    SUM_INSURED= models.CharField(max_length=300)
    POLICY_NO= models.CharField(max_length=300)
    POLICY_START= models.CharField(max_length=300)
    POLICY_END= models.CharField(max_length=300)
    INTIMATED_AMOUNT= models.CharField(max_length=300)
    INTIMATED_SF= models.CharField(max_length=300)
    EXECUTIVE= models.CharField(max_length=300)
    PRODUCT= models.CharField(max_length=300)
    POLICYTYPE= models.CharField(max_length=300)
    NATIONALITY= models.CharField(max_length=300)
    PREDICTION= models.CharField(max_length=300)
