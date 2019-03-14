# Install Packages
install.packages("Amelia")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("plotly")

install.packages("rminer")
install.packages("e1071")

#Load Libraries
library(Amelia)
library(ggplot2)
library(randomForest)
library(plotly)
library(rminer)
library(e1071)


# Set working directory and import datasets
setwd("C:\\Users\\monis\\Documents\\NEU-CPS\\Data Mining\\Titanic")
titanic.train <- read.csv(file = "train.csv", stringsAsFactors = F, header = T)
titanic.test <- read.csv(file = "test.csv", stringsAsFactors = F, header = T)


titanic.train$IsTrain <- TRUE
titanic.test$IsTrain <- FALSE

titanic.test$Survived <- NA

#merge train and test dataset
titanic <- rbind.data.frame(titanic.train, titanic.test)

summary(titanic)

####################DATA CLEANING########################

#view NA
missmap(titanic.train, col = c('yellow', 'black'), legend = T)

#Replace missing values in Embarked by mode
titanic[titanic$Embarked == '', "Embarked"] <- "S"


#replacing fare with median
titanic[is.na(titanic$Fare), "Fare"] <- median(titanic$Fare, na.rm = T)


hist(titanic$Age, main = "Before age prediction")

#Age prediction

upper.whisker <- boxplot.stats(titanic$Age)$stats[5]
outlier.filter <- titanic$Age < upper.whisker
cor(titanic.train[,sapply(titanic.train, is.numeric)], use = "complete.obs")


age.formula <- "Age ~ Fare + Parch"
age.model <- lm(formula = age.formula, data = titanic[outlier.filter,])
age.rows <- titanic[is.na(titanic$Age), c("Fare", "Parch")]
age.prediction <- predict(age.model, newdata = age.rows)
titanic[is.na(titanic$Age), "Age"] <- age.prediction
summary(titanic)
hist(titanic$Age, main = "after age prediction")

################## VISUALIZATION ############################

#Total survival rate
ggplot(titanic.train, aes(x = titanic.train$Survived, group = factor(titanic.train$Survived))) +
  geom_bar(show.legend = T)+
  theme_bw()+
  labs(x = "Survived", y = "Passenger Count")

prop.table(table(titanic.train$Survived))


#Survival rate by sex
a1 <- ggplot(titanic.train, aes(x = titanic.train$Sex, fill =factor(titanic.train$Survived))) + 
  theme_bw() +
  geom_bar() 
  
ggplotly(a1)

#Survival rate by Pclass
a2 <- ggplot(titanic.train, aes(x = titanic.train$Pclass, fill =factor(titanic.train$Survived))) + 
  theme_bw() +
  geom_bar() 

ggplotly(a2)

#Survival rate by sex and pclass
a3 <- ggplot(titanic.train, aes(x = titanic.train$Sex, fill =factor(titanic.train$Survived))) + 
  theme_bw() +
  facet_wrap( ~ Pclass) +
  geom_bar() 

ggplotly(a3)

#Distribution of age
a4 <- ggplot(titanic.train, aes(x = titanic.train$Age, fill = factor(titanic.train$Survived))) + 
  theme_bw() +
  geom_histogram(binwidth = 5) 

ggplotly(a4)

#Density plot
a5 <- ggplot(titanic.train, aes(x = titanic.train$Age, fill = factor(titanic.train$Survived))) +
  facet_wrap(titanic.train$Sex ~ titanic.train$Pclass ) +
  geom_density(alpha = 0.5) +
  theme_bw()

ggplotly(a5)
###################### categorical casting ############################

titanic$Pclass <- as.factor(titanic$Pclass)
table(titanic$Pclass)


titanic$Sex <- as.factor(titanic$Sex)
titanic$SibSp <- as.factor(titanic$SibSp)
titanic$Parch <- as.factor(titanic$Parch)
titanic$Embarked <- as.factor(titanic$Embarked)


titanic.train <- titanic[titanic$IsTrain == TRUE,]
titanic.test <- titanic[titanic$IsTrain == FALSE,]

titanic.train$Survived <- as.factor(titanic.train$Survived)

#################### FINDING CORRELATION##################

cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Age))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Sex))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Sex == "female"))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Pclass))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$SibSp))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Parch))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Fare))
cor(as.numeric(titanic.train$Survived), as.numeric(titanic.train$Embarked))

####################### FITTING MODELS ################################

survive.eq <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survive.formula <- as.formula(survive.eq)

#Random Forest

rf.model <- randomForest(formula = survive.formula , data = titanic.train, ntree = 50)
rf.Survived <- predict(rf.model, titanic.test)
plot(rf.Survived, ylim = c(0,300))
prop.table(table(predict(rf.model, titanic.test)))
#Summary
summary(rf.Survived)


#Naive Bayes

nb.model <- naiveBayes(formula = survive.formula , data = titanic.train)
nb.Survived <- predict(nb.model, titanic.test)
plot(nb.Survived,  ylim = c(0,350))
prop.table(table(predict(nb.model, titanic.test)))
#Summary 
summary(nb.Survived)

#########################
# Check accuracy by submitting on kaggle

PassengerId <- titanic.test$PassengerId
output.df<- as.data.frame(PassengerId)
output.df$Survived <- rf.Survived

write.csv(output.df, file = "Titanickaggle1.csv", row.names = FALSE)


