# Practical Meaching Learning Course Project
Gal Peled  
May 3, 2017  


##Pre analysis

### Repreducbility 

I set the seed to (185) so R will generate psedo random datasets for the split which will enable a easy repreduce of the split to training and validation data set 

### What are we predicting 

We try to get from each variable if the participate did it training workout currect and if not what was the errore 
this is represent in the classe variable that each value mean the following things 

- exactly according to the specification (Class A)
- throwing the elbows to the front (Class B)
- lifting the dumbbell only halfway (Class C)
- lowering the dumbbell only halfway (Class D)
- throwing the hips to the front (Class E) 
this data was made available from [1]

we will test 2 models Decision Tree and Random Forests and we will take the best of them as our result if none of the will give me result of more then 95% we will look into more algoritem 

### Cross - validation 
  
  I will split the training data set to 2 sets one for training(80%) and one for testing (20%)
  we will use the training set for training the model and the testing set for testing 
  
###what is the Expected out of sample errore 

well that is suppose to be 1-accuracy of the model we train 
in reality the Expected out of sample errore is corrisponding to the expected number of missclasifcation observation 

### Resose to my calculation 

we have a large sample size 19622 observation this allow me to tack 20 % of the data for testing 

## Data Analisis and Meaching Learning Code 

### Packeges 
loading library if you dont have one of the packeges you can install it by removing the # sing 

```r
#install.packages("caret")
#install.packages("randomForest")
#install.packages("rpart")
#install.packages("rpart.plot")
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
```
### Seed 
we set a specific see so it will be easy to repreduce 

```r
set.seed(185)
```


### loading the data and cleaning the data 
We want to load the data sets to R and make sure all the missing value are markt as NA
irrelevant variable will be deleted (empty column and date name and row nums columns)
Result will be hidden from this report for clarity and space constrain 

```r
#while looking at the data 3 value that mean na where found "NA","#DIV/0!", ""
TestSet     <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
TrainingSet <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""))

#remove column that all of there value are null
TestSet     <- TestSet[,colSums(is.na(TestSet))==0]
TrainingSet <- TrainingSet[,colSums(is.na(TrainingSet))==0]

#remove cloumn 1-7 as they contains data that will not help the prediction algoritem like name rownume date...
TestSet     <- TestSet[,-c(1:7)]
TrainingSet <- TrainingSet[,-c(1:7)]

str(TestSet)
str(TrainingSet)
```

###Spliting the training set to training and validation sets for cross validation 

so after seeing that the training set have 19622 observation and 53 variable 
and the test set have only 20 observation and it have no class value we will need to split our training set 
to a training part (80%) and validation / test part (20%) 
we will do it by using random sampaling with no return 


```r
splitBy <- createDataPartition(y= TrainingSet$classe,p=0.8,list = FALSE)
SubTrainingSet    <- TrainingSet[splitBy,]
SubValidationSet  <- TrainingSet[-splitBy,]
```

### A look at the classe variable

Lets see the frequency of eac class that we try to predict 


```r
plot(SubTrainingSet$classe,col ="green",main = "Frequency of each class",xlab="classe",ylab="Frequency")
```

![](ML_Project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

as we can see all the class are at the same scale when A as the most observation and D the least 

### Decision Tree model 


```r
DecisionTreeModel <- rpart(classe~.,data = SubTrainingSet,method = "class")

PredictionTreeModel <- predict(DecisionTreeModel,SubValidationSet,type = "class")

# show the decision Tree

rpart.plot(DecisionTreeModel,main= "Decision Tree")
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![](ML_Project_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
confusionMatrix(data = PredictionTreeModel, reference = SubValidationSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1033   87    9   66   38
##          B   44  549  117   40   49
##          C    2   72  487   95   57
##          D   29   45   45  427   55
##          E    8    6   26   15  522
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7693          
##                  95% CI : (0.7558, 0.7824)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.707           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9256   0.7233   0.7120   0.6641   0.7240
## Specificity            0.9287   0.9210   0.9302   0.9470   0.9828
## Pos Pred Value         0.8378   0.6871   0.6830   0.7105   0.9047
## Neg Pred Value         0.9691   0.9328   0.9386   0.9350   0.9405
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2633   0.1399   0.1241   0.1088   0.1331
## Detection Prevalence   0.3143   0.2037   0.1817   0.1532   0.1471
## Balanced Accuracy      0.9272   0.8222   0.8211   0.8055   0.8534
```

this model gice us  0.74 accuracy that is nice but lets see if we can get better result 

### Random Forest model


```r
RandomForestModel <- randomForest(classe~.,data = SubTrainingSet,method ="class")
PredictionForestModel <- predict(RandomForestModel,SubValidationSet,method = "class")
confusionMatrix(PredictionForestModel,SubValidationSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    2    0    0    0
##          B    1  754    7    0    0
##          C    0    3  677    6    2
##          D    0    0    0  637    3
##          E    0    0    0    0  716
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9909, 0.9961)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9934   0.9898   0.9907   0.9931
## Specificity            0.9993   0.9975   0.9966   0.9991   1.0000
## Pos Pred Value         0.9982   0.9895   0.9840   0.9953   1.0000
## Neg Pred Value         0.9996   0.9984   0.9978   0.9982   0.9984
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1922   0.1726   0.1624   0.1825
## Detection Prevalence   0.2847   0.1942   0.1754   0.1631   0.1825
## Balanced Accuracy      0.9992   0.9954   0.9932   0.9949   0.9965
```
This model give us 0.994 accuracy that is pretty good and there is no need to try more model with this result 

### Desicion 
Apperntly Random Forest algoritem preforms great in this kind of data and gave us outstending result 

Accuracy for RF is 0.994 against the 0.74 of decision Tree and so we will chose the RF algoritem 

with an accuracy of 99% and the expected out of sample error is 1- accuarcy = 0.06% we should expect almost no errore in the test set


```r
testSetResult <- predict(RandomForestModel,TestSet,method = "class")
testSetResult
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

[1]Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013
