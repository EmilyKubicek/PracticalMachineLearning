---
title: "Pracitcal Machine Learning Course Project"
author: "Emily Kubicek"
date: "2/27/2020"
output: rmarkdown::github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(corrplot)
source("http://www.sthda.com/upload/rquery_cormat.r")
require("corrplot")
library(caret)
```

## R Markdown

They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Load Data
```{r}
# Load in training/test sets
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

```

### Explore/clean data
```{r, echo=FALSE}
# Remove variables with near zero variance in both sets
# identify column index where there is near zero variance
nzv <- nearZeroVar(training)

# make new training and test sets without these column indices
training2 <- training[,-nzv]

# check work
dim(training2)
dim(training)

# same for testing
testing2 <- testing[,-nzv]

# check for NA
sum(is.na(training2))
sum(is.na(testing2))

# remove variables that are mostly NA
lotsna <- sapply(training2, function(x) mean(is.na(x))) > .95
training3 <- training2[,lotsna == FALSE]

# Check work
dim(training3)
dim(training2)

# same for testing
testing3 <- testing2[,lotsna == FALSE]

# Remove identity/time related data
training4 <- training3[,-(1:5)]

# check work
dim(training4)

# apply to testing
testing4 <- testing3[,-(1:5)]

```

The above cleaning has changed our data significantly. We now only have 54 variables to work with for both training and testing, with all variables contianing mostly relevant (i.e. not NA data). However, in order to see the accuracies of our model, we must further break apart the training set. Leaving the inital test set for validation.

```{r}
inTrain <- createDataPartition(training4$classe, p = 3/4, list = FALSE)
training5 <- training4[inTrain,]
testing5 <- training4[-inTrain,]
```

### Conduct inital correlation analysis
```{r, echo = FALSE}
# we want everything in the corrlation matrix except what we are trying to predict
cormat <- cor(training5[,-54])

# plot the correlation matrix
corrplot(cormat, type = "lower", method = "color", order = "FPC", tl.cex = 0.5)
```
The above correlation matrix gives us an overview of our variables. We can see that many of our variables are highly correlated (represented by dark colors in the matrix). Given the high amount of variables, PCA could be performed to create powerful components. However, for this analysis we will be attempting to fit multiple models instead to which performs best.


### Fit models
#### Random Forest
```{r}
# fit cleaned data to random forest and find accuracy
set.seed(11111)
rfmod <- randomForest(classe ~ ., data = training5)
rfmodpred <- predict(rfmod, testing5)
confusionMatrix(rfmodpred, testing5$classe)$overall[1]

```

Above we can see that our random forest model yielded an extremely high accuracy rate. While this may seem great, this highly suggests that we overfit our model. Let's see what some other models look like.

#### Decision Tree
```{r}
set.seed(11111)
dtmod <- train(classe ~ ., method = "rpart", data = training5)
dtmodpred <- predict(dtmod, testing5)
confusionMatrix(dtmodpred, testing5$classe)$overall[1]

# get visual of deicision tree
library(rattle)
fancyRpartPlot(dtmod$finalModel)
```
Our decision tree model gives us an exceptionally low accuracy (~50%). Let's check out one more model before deciding which to apply to our test set.

#### Generalized Boosted Model
```{r}
control <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
gbmod  <- train(classe ~ ., data = training5, method = "gbm",
                    trControl = control, verbose = FALSE)

gbmodpred <- predict(gbmod, testing5)
confusionMatrix(gbmodpred, testing5$classe)$overall[1]

```

Much like our random forest model, we get a 98% accuracy. While that may seem good, it is more than likley we are overfitting. Let's see what model had the highest accuracy and fit our test data to it.

Accuracies for tested models:
- Random Forest: 99.8%
- Decision Tree: 49.0%
- GBM: 98.6%

```{r}
# Apply random forest model to our validation set
finaltest <- predict(rfmod, testing)
finaltest

```

